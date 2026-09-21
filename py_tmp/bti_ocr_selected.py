#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Выборочный, параллельный и кешируемый OCR для уже созданных BTI-дампов.

Вход: CSV с колонкой «путь» (как исходный список PDF).
Ищет для каждого PDF его папку в --dump-root, созданную bti_dump_pdfs.py:
  <dump-root>/<safe_stem>__<hash>/manifest.json
  <dump-root>/<safe_stem>__<hash>/dump.xlsx

Выход, добавляемый в папку конкретного PDF:
  ocr/page_0001.txt        — OCR-текст страницы
  ocr/page_0001.json       — метрики и статус страницы
  ocr_index.csv            — постраничный кеш / журнал
  ocr_manifest.json        — сводка OCR конкретного PDF

По умолчанию OCR запускается ТОЛЬКО для PDF, чей manifest.json имеет
"document_mode": "scan". PDF типа native и mixed будут пропущены.

Зависимости:
  py -m pip install pymupdf pytesseract pillow openpyxl tqdm

Tesseract:
  папка tesseract должна лежать рядом со скриптом:
    .\tesseract\tesseract.exe
    .\tesseract\tessdata\rus.traineddata
    .\tesseract\tessdata\eng.traineddata

Примеры:
  # Проверка отбора страниц без OCR и без записи результата
  py bti_ocr_selected.py --csv ocr_candidates.csv --sep ";" --dump-root bti_dumps --dry-run

  # Основной быстрый массовый запуск: только scan-PDF, 3 процесса
  py bti_ocr_selected.py --csv ocr_candidates.csv --sep ";" --dump-root bti_dumps --workers 3

  # Добавить mixed-PDF
  py bti_ocr_selected.py --csv ocr_candidates.csv --sep ";" --dump-root bti_dumps --include-mixed --workers 3

  # Повторить только страницы с timeout/error
  py bti_ocr_selected.py --csv ocr_candidates.csv --sep ";" --dump-root bti_dumps --retry-failed --workers 2

  # Принудительно сделать OCR конкретных страниц, включая плановые
  py bti_ocr_selected.py --csv one_pdf.csv --sep ";" --dump-root bti_dumps --force-pages "17,18,42" --workers 1
"""

import argparse
import csv
import hashlib
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import fitz
import pytesseract
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items, **kwargs):
        return items


KEYWORDS_RE = re.compile(
    r"эксплик|площад|помещен|квартир|нежил|общая\s+площад|жилая\s+площад",
    re.IGNORECASE,
)
PLAN_WORDS_RE = re.compile(
    r"поэтажн|план\s+(?:этажа|помещен)|схема|масштаб|условн(?:ые|ых)\s+обознач",
    re.IGNORECASE,
)
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

DONE_STATUSES = {
    "ok",
    "skipped_plan_suspect",
    "skipped_too_large",
    "skipped_native_text",
    "skipped_not_bad",
    "skipped_low_text",
}
RETRYABLE_STATUSES = {"error", "timeout"}
OCR_COLUMNS = [
    "page", "status", "dpi", "megapixels", "chars", "seconds",
    "native_chars", "plan_score", "reason", "text_file", "meta_file", "updated_at",
]


def norm_text(text):
    text = (text or "").replace("\xa0", " ").replace("\u202f", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def native_is_bad(text):
    text = norm_text(text)
    if len(text) < 80:
        return True
    cid = text.casefold().count("(cid:")
    cyr = len(CYRILLIC_RE.findall(text))
    return cid >= 5 or cyr < max(8, len(text) // 80)


def safe_name(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", " ", text).strip(" ._")
    return text[:80] or "pdf"


def short_hash(path):
    return hashlib.sha1(str(path).encode("utf-8", errors="replace")).hexdigest()[:10]


def dump_folder_for(pdf_path, dump_root):
    return dump_root / f"{safe_name(pdf_path.stem)}__{short_hash(pdf_path)}"


def read_paths(csv_path, column, delimiter):
    paths, seen = [], set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise SystemExit(f"В CSV нет колонки «{column}». Найдены: {reader.fieldnames}")
        for row in reader:
            raw = (row.get(column) or "").strip().strip('"')
            if raw and raw not in seen:
                paths.append(Path(raw))
                seen.add(raw)
    return paths


def find_tesseract(script_dir, supplied):
    if supplied:
        candidate = Path(supplied)
        return candidate if candidate.is_file() else None
    for candidate in [
        script_dir / "tesseract" / "tesseract.exe",
        script_dir / "Tesseract-OCR" / "tesseract.exe",
        script_dir / "tesseract.exe",
    ]:
        if candidate.is_file():
            return candidate
    return None


def validate_tesseract(exe, lang):
    if not exe:
        raise SystemExit("Tesseract не найден. Положите tesseract.exe в .\\tesseract\\ или передайте --tesseract-cmd.")
    pytesseract.pytesseract.tesseract_cmd = str(exe)
    tessdata = exe.parent / "tessdata"
    if tessdata.is_dir():
        os.environ.setdefault("TESSDATA_PREFIX", str(tessdata))
    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception as exc:
        raise SystemExit(f"Не удалось запустить Tesseract: {exc}")
    missing = [x for x in lang.split("+") if x not in available]
    if missing:
        raise SystemExit(f"В Tesseract нет языков: {', '.join(missing)}")


def load_json(path, default=None):
    if not path.is_file():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def load_ocr_index(path):
    rows = {}
    if not path.is_file():
        return rows
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rows[int(row.get("page", ""))] = row
                except ValueError:
                    continue
    except Exception:
        return {}
    return rows


def write_ocr_index(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OCR_COLUMNS)
        writer.writeheader()
        for page in sorted(rows):
            writer.writerow({key: rows[page].get(key, "") for key in OCR_COLUMNS})


def page_metrics(page):
    text = norm_text(page.get_text("text", sort=True))
    blocks = page.get_text("blocks", sort=True)
    image_blocks = [b for b in blocks if len(b) >= 7 and b[6] == 1]
    area = max(1, page.rect.width * page.rect.height)
    image_area = sum(max(0, b[2] - b[0]) * max(0, b[3] - b[1]) for b in image_blocks)
    try:
        drawings = len(page.get_drawings())
    except Exception:
        drawings = 0
    score, signals = 0, []
    if len(text) < 35:
        score += 25
        signals.append("мало текста")
    if image_area / area >= 0.55:
        score += 30
        signals.append("крупное изображение")
    if drawings >= 80:
        score += 30
        signals.append("много векторной графики")
    elif drawings >= 25:
        score += 10
        signals.append("векторная графика")
    if PLAN_WORDS_RE.search(text):
        score += 20
        signals.append("слова плана")
    if KEYWORDS_RE.search(text):
        score -= 35
        signals.append("слова таблицы площадей")
    return {
        "native_text": text,
        "native_chars": len(text),
        "native_bad": native_is_bad(text),
        "plan_score": max(0, score),
        "plan_signals": "; ".join(signals),
        "width": page.rect.width,
        "height": page.rect.height,
    }


def render_page(page, dpi):
    scale = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    try:
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        pix = None
    return image


def ocr_one_page(page, dpi, lang, psm, timeout):
    image = render_page(page, dpi)
    megapixels = (image.width * image.height) / 1_000_000
    try:
        text = pytesseract.image_to_string(
            image,
            lang=lang,
            config=f"--oem 1 --psm {psm}",
            timeout=timeout,
        )
        return norm_text(text), megapixels
    finally:
        image.close()


def write_page_result(ocr_dir, row, text):
    page = int(row["page"])
    text_path = ocr_dir / f"page_{page:04d}.txt"
    meta_path = ocr_dir / f"page_{page:04d}.json"
    if text is not None:
        text_path.write_text(text, encoding="utf-8")
        row["text_file"] = str(text_path.name)
    meta_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    row["meta_file"] = str(meta_path.name)


def should_process_page(existing, args):
    if not existing:
        return True
    status = existing.get("status", "")
    if args.force_pages:
        return True
    if args.retry_failed and status in RETRYABLE_STATUSES:
        return True
    return status not in DONE_STATUSES and status not in RETRYABLE_STATUSES


def process_pdf_worker(task):
    pdf_text = task["pdf_path"]
    pdf_path = Path(pdf_text)
    dump_folder = Path(task["dump_folder"])
    args = task["args"]
    tesseract_cmd = task["tesseract_cmd"]

    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    ocr_dir = dump_folder / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    index_path = dump_folder / "ocr_index.csv"
    rows = load_ocr_index(index_path)
    started = time.time()
    outcome = {"pdf": pdf_text, "status": "ok", "processed": 0, "ok": 0, "skipped": 0, "errors": 0, "folder": str(dump_folder)}

    if not pdf_path.is_file():
        outcome.update({"status": "missing", "errors": 1})
        return outcome

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        outcome.update({"status": "error", "errors": 1, "reason": f"open PDF: {type(exc).__name__}: {exc}"})
        return outcome

    try:
        selected = set(args["force_pages"])
        for index in range(doc.page_count):
            page_no = index + 1
            existing = rows.get(page_no)
            if not should_process_page(existing, SimpleArgs(args)):
                continue
            if selected and page_no not in selected:
                continue

            page = doc.load_page(index)
            metrics = page_metrics(page)
            row = {
                "page": page_no,
                "status": "",
                "dpi": args["dpi"],
                "megapixels": "",
                "chars": "",
                "seconds": "",
                "native_chars": metrics["native_chars"],
                "plan_score": metrics["plan_score"],
                "reason": "",
                "text_file": "",
                "meta_file": "",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            outcome["processed"] += 1

            if not args["force_pages"] and not metrics["native_bad"]:
                row.update({"status": "skipped_native_text", "reason": "нормальный нативный текст"})
                write_page_result(ocr_dir, row, None)
                rows[page_no] = row
                outcome["skipped"] += 1
                continue

            if not args["force_pages"] and metrics["plan_score"] >= args["plan_skip_score"] and not KEYWORDS_RE.search(metrics["native_text"]):
                row.update({"status": "skipped_plan_suspect", "reason": f"plan score {metrics['plan_score']}: {metrics['plan_signals']}"})
                write_page_result(ocr_dir, row, None)
                rows[page_no] = row
                outcome["skipped"] += 1
                continue

            predicted_mp = ((metrics["width"] * args["dpi"] / 72) * (metrics["height"] * args["dpi"] / 72)) / 1_000_000
            if predicted_mp > args["max_megapixels"]:
                row.update({"status": "skipped_too_large", "megapixels": round(predicted_mp, 2), "reason": f"{predicted_mp:.2f} MP > limit {args['max_megapixels']} MP"})
                write_page_result(ocr_dir, row, None)
                rows[page_no] = row
                outcome["skipped"] += 1
                continue

            t0 = time.time()
            try:
                text, megapixels = ocr_one_page(page, args["dpi"], args["ocr_lang"], args["ocr_psm"], args["timeout"])
                elapsed = time.time() - t0
                row.update({"megapixels": round(megapixels, 2), "seconds": round(elapsed, 2), "chars": len(text)})
                if len(text) < args["min_chars"] and not KEYWORDS_RE.search(text):
                    row.update({"status": "skipped_low_text", "reason": f"OCR text: {len(text)} chars"})
                    write_page_result(ocr_dir, row, text)
                    outcome["skipped"] += 1
                else:
                    row["status"] = "ok"
                    write_page_result(ocr_dir, row, text)
                    outcome["ok"] += 1
            except RuntimeError as exc:
                elapsed = time.time() - t0
                reason = str(exc)
                status = "timeout" if "timeout" in reason.casefold() else "error"
                row.update({"status": status, "seconds": round(elapsed, 2), "reason": f"{type(exc).__name__}: {reason}"})
                write_page_result(ocr_dir, row, None)
                outcome["errors"] += 1
            except Exception as exc:
                elapsed = time.time() - t0
                row.update({"status": "error", "seconds": round(elapsed, 2), "reason": f"{type(exc).__name__}: {exc}"})
                write_page_result(ocr_dir, row, None)
                outcome["errors"] += 1
            rows[page_no] = row
            write_ocr_index(index_path, rows)

        write_ocr_index(index_path, rows)
        ocr_manifest = {
            "source_pdf": pdf_text,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.time() - started, 2),
            "page_count": doc.page_count,
            "processed_this_run": outcome["processed"],
            "ok_this_run": outcome["ok"],
            "skipped_this_run": outcome["skipped"],
            "errors_this_run": outcome["errors"],
            "settings": args,
        }
        (dump_folder / "ocr_manifest.json").write_text(json.dumps(ocr_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return outcome
    except Exception as exc:
        outcome.update({"status": "error", "errors": outcome["errors"] + 1, "reason": f"worker: {type(exc).__name__}: {exc}"})
        return outcome
    finally:
        doc.close()


class SimpleArgs:
    def __init__(self, values):
        self.__dict__.update(values)


def eligible_pdf(pdf_path, dump_root, include_mixed):
    folder = dump_folder_for(pdf_path, dump_root)
    manifest = load_json(folder / "manifest.json")
    if not manifest or manifest.get("status") != "ok":
        return False, folder, "no_native_dump"
    mode = manifest.get("document_mode", "")
    if mode == "scan":
        return True, folder, mode
    if include_mixed and mode == "mixed":
        return True, folder, mode
    return False, folder, mode or "unknown"


def main():
    ap = argparse.ArgumentParser(description="Быстрый выборочный OCR сканированных БТИ-PDF с постраничным кешем.")
    ap.add_argument("--csv", required=True, help="CSV с колонкой «путь»")
    ap.add_argument("--path-column", default="путь")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--dump-root", required=True, help="Папка результата bti_dump_pdfs.py")
    ap.add_argument("--workers", type=int, default=3, help="Одновременные PDF-процессы; начните с 3")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--ocr-lang", default="rus+eng")
    ap.add_argument("--ocr-psm", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=60, help="Максимум секунд на одну страницу")
    ap.add_argument("--max-megapixels", type=float, default=14.0, help="Пропустить слишком большие страницы")
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--plan-skip-score", type=int, default=55)
    ap.add_argument("--include-mixed", action="store_true", help="Также OCR-ить PDF с режимом mixed")
    ap.add_argument("--retry-failed", action="store_true", help="Повторить только timeout/error в существующем OCR-кеше")
    ap.add_argument("--force-pages", default="", help="Номера страниц через запятую: 17,18,42; игнорирует плановый фильтр")
    ap.add_argument("--tesseract-cmd", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Показать PDF-кандидаты, не запускать OCR")
    args = ap.parse_args()

    if len(args.sep) != 1:
        raise SystemExit("--sep должен быть одним символом")
    if args.workers < 1:
        raise SystemExit("--workers должен быть не меньше 1")
    if args.dpi < 100:
        raise SystemExit("--dpi слишком низкий")
    try:
        force_pages = sorted({int(x.strip()) for x in args.force_pages.split(",") if x.strip()})
    except ValueError:
        raise SystemExit("--force-pages: только номера через запятую, например 17,18,42")

    csv_path = Path(args.csv)
    dump_root = Path(args.dump_root)
    if not csv_path.is_file():
        raise SystemExit(f"Нет CSV: {csv_path}")
    if not dump_root.is_dir():
        raise SystemExit(f"Нет папки дампов: {dump_root}")
    paths = read_paths(csv_path, args.path_column, args.sep)
    if not paths:
        raise SystemExit("В CSV нет непустых путей")

    script_dir = Path(__file__).resolve().parent
    tesseract = find_tesseract(script_dir, args.tesseract_cmd)
    validate_tesseract(tesseract, args.ocr_lang)

    tasks, skipped = [], []
    public_args = {
        "dpi": args.dpi,
        "ocr_lang": args.ocr_lang,
        "ocr_psm": args.ocr_psm,
        "timeout": args.timeout,
        "max_megapixels": args.max_megapixels,
        "min_chars": args.min_chars,
        "plan_skip_score": args.plan_skip_score,
        "retry_failed": args.retry_failed,
        "force_pages": force_pages,
    }
    for path in paths:
        ok, folder, reason = eligible_pdf(path, dump_root, args.include_mixed)
        if ok:
            tasks.append({"pdf_path": str(path), "dump_folder": str(folder), "args": public_args, "tesseract_cmd": str(tesseract)})
        else:
            skipped.append((str(path), reason, str(folder)))

    print(f"Всего путей: {len(paths)} | OCR-кандидатов: {len(tasks)} | пропущено: {len(skipped)}")
    if skipped:
        by_reason = {}
        for _, reason, _ in skipped:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        print("Пропуски PDF:", ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items())))
    if args.dry_run:
        for task in tasks[:30]:
            print(task["pdf_path"])
        if len(tasks) > 30:
            print(f"... ещё {len(tasks)-30}")
        return

    summary_path = dump_root / "ocr_run_summary.csv"
    summary_fields = ["source_pdf", "status", "output_folder", "processed_pages", "ok_pages", "skipped_pages", "error_pages", "reason"]
    counts = {"ok": 0, "error": 0, "missing": 0}
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_pdf_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="OCR PDF", unit="pdf"):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"pdf": "", "status": "error", "folder": "", "processed": 0, "ok": 0, "skipped": 0, "errors": 1, "reason": f"executor: {type(exc).__name__}: {exc}"}
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                writer.writerow({
                    "source_pdf": result.get("pdf", ""), "status": result.get("status", ""), "output_folder": result.get("folder", ""),
                    "processed_pages": result.get("processed", 0), "ok_pages": result.get("ok", 0),
                    "skipped_pages": result.get("skipped", 0), "error_pages": result.get("errors", 0), "reason": result.get("reason", ""),
                })
                f.flush()
    print("Готово:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"Сводка: {summary_path}")


if __name__ == "__main__":
    main()
