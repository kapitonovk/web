#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный OCR-дампер БТИ-PDF.

Вход: CSV с колонкой «путь».

Для каждого PDF создаёт одну папку в --outdir:
  <safe_pdf_name>__<path_hash>/
    manifest.json                 метаданные документа, режим, поворот, итоги
    ocr_index.csv                 постраничный кеш и журнал результата
    page0001_text.txt             сырой OCR-текст Tesseract
    page0001_result.tsv           слова, координаты, confidence
    page0001_meta.json            параметры и метрики конкретной страницы
    ...
    control/
      page0001_render.png         контрольный исходный grayscale-рендер
      page0001_prepared.png       bitmap, переданный Tesseract
      page0042_render.png
      page0042_prepared.png
      ...

Базовые настройки, подтверждённые тестом:
  - raw grayscale render, без внешних фильтров;
  - Tesseract Adaptive Otsu: thresholding_method=1;
  - PSM 11;
  - язык rus;
  - 400 DPI;
  - общий автоповорот 0° / 180° по первой доступной странице PDF;
  - страницы крупнее --max-megapixels пропускаются (обычно A3/A2 чертежи).

Все ключевые настройки вынесены в аргументы с дефолтами, чтобы отдельные
сложные документы можно было перезапустить без правки кода.

Зависимости:
  py -m pip install pymupdf pytesseract pillow tqdm

Tesseract рядом со скриптом:
  .\tesseract\tesseract.exe
  .\tesseract\tessdata\rus.traineddata

Пример:
  py bti_ocr_dump.py --csv ocr_candidates.csv --sep ";" --outdir bti_ocr_dumps --workers 2

Продолжить после остановки:
  та же команда — готовые страницы из ocr_index.csv пропустятся.

Повторить только timeout/error:
  ... --retry-failed

Полностью перераспознать документ/список:
  ... --overwrite
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Один Tesseract-процесс на один поток; число процессов задаётся --workers.
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


INDEX_COLUMNS = [
    "page", "status", "dpi", "psm", "ocr_lang", "rotation", "megapixels",
    "chars", "digits", "commas_dots", "tsv_words", "tsv_digit_words", "seconds",
    "reason", "text_file", "tsv_file", "meta_file", "control_render", "control_prepared",
    "updated_at",
]
DONE_STATUSES = {"ok", "low_text", "skipped_too_large"}
RETRYABLE_STATUSES = {"error", "timeout"}


def safe_name(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", " ", text).strip(" ._")
    return text[:80] or "pdf"


def short_hash(path):
    return hashlib.sha1(str(path).encode("utf-8", errors="replace")).hexdigest()[:10]


def output_folder_for(pdf_path, outdir):
    return outdir / f"{safe_name(pdf_path.stem)}__{short_hash(pdf_path)}"


def norm_text(text):
    text = (text or "").replace("\xa0", " ").replace("\u202f", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
        raise SystemExit("Tesseract не найден. Нужен .\\tesseract\\tesseract.exe или --tesseract-cmd.")
    pytesseract.pytesseract.tesseract_cmd = str(exe)
    tessdata = exe.parent / "tessdata"
    if tessdata.is_dir():
        os.environ.setdefault("TESSDATA_PREFIX", str(tessdata))
    try:
        languages = set(pytesseract.get_languages(config=""))
    except Exception as exc:
        raise SystemExit(f"Не удалось запустить Tesseract: {exc}")
    missing = [part for part in lang.split("+") if part not in languages]
    if missing:
        raise SystemExit(f"В Tesseract отсутствуют языки: {', '.join(missing)}")


def load_index(path):
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


def save_index(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for page_no in sorted(rows):
            writer.writerow({field: rows[page_no].get(field, "") for field in INDEX_COLUMNS})


def render_page(page, dpi, rotation=0):
    scale = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False)
    try:
        image = Image.frombytes("L", [pix.width, pix.height], pix.samples)
    finally:
        pix = None
    if rotation:
        rotated = image.rotate(rotation, expand=True, fillcolor=255)
        image.close()
        image = rotated
    return image


def predicted_megapixels(page, dpi):
    return ((page.rect.width * dpi / 72) * (page.rect.height * dpi / 72)) / 1_000_000


def tesseract_config(psm, thresholding_method, write_images=False):
    pieces = [
        f"--oem 1 --psm {psm}",
        "-c preserve_interword_spaces=1",
        f"-c thresholding_method={thresholding_method}",
    ]
    if write_images:
        pieces.append("-c tessedit_write_images=1")
    return " ".join(pieces)


def orientation_score(text):
    """Оценка 0°/180°: читаемый русский текст, цифры и типичные БТИ-маркеры."""
    text = norm_text(text)
    if not text:
        return 0
    lower = text.casefold()
    cyrillic = len(re.findall(r"[а-яё]", lower))
    digits = len(re.findall(r"\d", text))
    markers = len(re.findall(r"эксплик|площад|помещен|квартир|технич|паспорт|этаж|адрес|инвентар", lower))
    garbage = lower.count("(cid:") * 100 + lower.count("�") * 50
    return min(len(text), 3000) + cyrillic * 2 + digits + markers * 80 - garbage


def detect_rotation(doc, cfg):
    """Определяет общий 0/180°-поворот по первой странице, доступной для рендера."""
    config = tesseract_config(cfg["psm"], cfg["thresholding_method"], write_images=False)
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        if predicted_megapixels(page, cfg["dpi"]) > cfg["max_megapixels"]:
            continue
        raw = render_page(page, cfg["dpi"], rotation=0)
        try:
            scores = {}
            for angle in (0, 180):
                candidate = raw if angle == 0 else raw.rotate(180, expand=True, fillcolor=255)
                try:
                    text = pytesseract.image_to_string(candidate, lang=cfg["ocr_lang"], config=config, timeout=cfg["timeout"])
                    scores[angle] = orientation_score(text)
                except Exception:
                    scores[angle] = -1
                finally:
                    if candidate is not raw:
                        candidate.close()
            return (180 if scores[180] > scores[0] else 0), page_idx + 1, scores
        finally:
            raw.close()
    return 0, None, {0: 0, 180: 0}


def control_pages(page_count, count, end_offset):
    """Начало, две равноудалённые внутренние точки и почти конец документа."""
    if page_count <= 0 or count <= 0:
        return set()
    last = max(1, page_count - max(0, end_offset))
    if count == 1:
        return {1}
    values = {1, last}
    # В стандартном случае count=4: 1, ~1/3, ~2/3, page_count-end_offset.
    for i in range(1, count - 1):
        position = 1 + round((last - 1) * i / (count - 1))
        values.add(min(page_count, max(1, position)))
    return values


def move_tessinput(workdir, target_dir, page_no):
    candidates = list(workdir.glob("tessinput.*")) + list(workdir.glob("*.processed.tif"))
    if not candidates:
        return ""
    source = max(candidates, key=lambda path: path.stat().st_mtime)
    target = target_dir / f"page{page_no:04d}_tessinput{source.suffix or '.tif'}"
    shutil.move(str(source), str(target))
    return target.name


def count_metrics(text, tsv):
    word_count = digit_words = 0
    for row in csv.DictReader(tsv.splitlines(), delimiter="\t"):
        value = (row.get("text") or "").strip()
        if row.get("level") != "5" or not value:
            continue
        word_count += 1
        if re.search(r"\d", value):
            digit_words += 1
    return {
        "chars": len(text),
        "digits": len(re.findall(r"\d", text)),
        "commas_dots": len(re.findall(r"[,.]", text)),
        "tsv_words": word_count,
        "tsv_digit_words": digit_words,
    }


def should_process(existing, retry_failed, overwrite, forced):
    if forced or overwrite or not existing:
        return True
    status = existing.get("status", "")
    if retry_failed and status in RETRYABLE_STATUSES:
        return True
    return status not in DONE_STATUSES and status not in RETRYABLE_STATUSES


def write_page_meta(output_folder, page_no, row):
    path = output_folder / f"page{page_no:04d}_meta.json"
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    row["meta_file"] = path.name


def process_pdf_worker(task):
    pdf_path = Path(task["pdf"])
    output_folder = Path(task["output_folder"])
    cfg = task["cfg"]
    force_pages = set(cfg["force_pages"])

    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    pytesseract.pytesseract.tesseract_cmd = cfg["tesseract"]

    output_folder.mkdir(parents=True, exist_ok=True)
    control_dir = output_folder / "control"
    index_path = output_folder / "ocr_index.csv"
    index = load_index(index_path)
    started = time.time()
    outcome = {
        "pdf": str(pdf_path), "output_folder": str(output_folder), "status": "ok",
        "processed": 0, "ok": 0, "low": 0, "large": 0, "errors": 0,
    }

    if not pdf_path.is_file():
        outcome.update({"status": "missing", "errors": 1, "reason": "PDF не найден"})
        return outcome
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        outcome.update({"status": "error", "errors": 1, "reason": f"open PDF: {type(exc).__name__}: {exc}"})
        return outcome

    try:
        rotation, rotation_page, rotation_scores = detect_rotation(doc, cfg)
        controls = control_pages(doc.page_count, cfg["control_count"], cfg["control_end_offset"])
        ocr_config = tesseract_config(cfg["psm"], cfg["thresholding_method"], write_images=False)

        for page_idx in range(doc.page_count):
            page_no = page_idx + 1
            forced = page_no in force_pages
            if not should_process(index.get(page_no), cfg["retry_failed"], cfg["overwrite"], forced):
                continue

            page = doc.load_page(page_idx)
            expected_mp = predicted_megapixels(page, cfg["dpi"])
            is_control = page_no in controls
            row = {
                "page": page_no,
                "status": "",
                "dpi": cfg["dpi"],
                "psm": cfg["psm"],
                "ocr_lang": cfg["ocr_lang"],
                "rotation": rotation,
                "megapixels": round(expected_mp, 2),
                "chars": "",
                "digits": "",
                "commas_dots": "",
                "tsv_words": "",
                "tsv_digit_words": "",
                "seconds": "",
                "reason": "",
                "text_file": "",
                "tsv_file": "",
                "meta_file": "",
                "control_render": "",
                "control_prepared": "",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            outcome["processed"] += 1

            if expected_mp > cfg["max_megapixels"] and not forced:
                row.update({
                    "status": "skipped_too_large",
                    "reason": f"{expected_mp:.2f} MP > limit {cfg['max_megapixels']:.2f} MP",
                })
                write_page_meta(output_folder, page_no, row)
                index[page_no] = row
                save_index(index_path, index)
                outcome["large"] += 1
                continue

            image = render_page(page, cfg["dpi"], rotation=rotation)
            try:
                if is_control:
                    control_dir.mkdir(parents=True, exist_ok=True)
                    render_name = f"page{page_no:04d}_render.png"
                    prepared_name = f"page{page_no:04d}_prepared.png"
                    image.save(control_dir / render_name)
                    # В текущем выбранном режиме bitmap перед Tesseract не меняется.
                    image.save(control_dir / prepared_name)
                    row["control_render"] = f"control/{render_name}"
                    row["control_prepared"] = f"control/{prepared_name}"

                t0 = time.time()
                try:
                    text = pytesseract.image_to_string(
                        image, lang=cfg["ocr_lang"], config=ocr_config, timeout=cfg["timeout"]
                    )
                    tsv = pytesseract.image_to_data(
                        image,
                        lang=cfg["ocr_lang"],
                        config=ocr_config,
                        timeout=cfg["timeout"],
                        output_type=pytesseract.Output.STRING,
                    )
                    elapsed = time.time() - t0
                    text = norm_text(text)
                    text_name = f"page{page_no:04d}_text.txt"
                    tsv_name = f"page{page_no:04d}_result.tsv"
                    (output_folder / text_name).write_text(text, encoding="utf-8")
                    (output_folder / tsv_name).write_text(tsv, encoding="utf-8")
                    row.update({
                        "chars": len(text),
                        "seconds": round(elapsed, 2),
                        "text_file": text_name,
                        "tsv_file": tsv_name,
                        **count_metrics(text, tsv),
                    })
                    if len(text) < cfg["min_chars"]:
                        row.update({"status": "low_text", "reason": f"распознано только {len(text)} символов"})
                        outcome["low"] += 1
                    else:
                        row["status"] = "ok"
                        outcome["ok"] += 1
                except RuntimeError as exc:
                    message = str(exc)
                    row.update({
                        "status": "timeout" if "timeout" in message.casefold() else "error",
                        "seconds": round(time.time() - t0, 2),
                        "reason": f"{type(exc).__name__}: {message}",
                    })
                    outcome["errors"] += 1
                except Exception as exc:
                    row.update({
                        "status": "error",
                        "seconds": round(time.time() - t0, 2),
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
                    outcome["errors"] += 1
            finally:
                image.close()

            write_page_meta(output_folder, page_no, row)
            index[page_no] = row
            save_index(index_path, index)

        save_index(index_path, index)
        manifest = {
            "source_pdf": str(pdf_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "ok",
            "page_count": doc.page_count,
            "rotation": rotation,
            "rotation_detection_page": rotation_page,
            "rotation_detection_scores": rotation_scores,
            "control_pages": sorted(controls),
            "processed_this_run": outcome["processed"],
            "ok_this_run": outcome["ok"],
            "low_text_this_run": outcome["low"],
            "skipped_too_large_this_run": outcome["large"],
            "errors_this_run": outcome["errors"],
            "settings": {key: value for key, value in cfg.items() if key != "tesseract"},
        }
        (output_folder / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return outcome
    except Exception as exc:
        outcome.update({
            "status": "error",
            "errors": outcome["errors"] + 1,
            "reason": f"worker: {type(exc).__name__}: {exc}",
        })
        return outcome
    finally:
        doc.close()


def main():
    ap = argparse.ArgumentParser(description="Полный OCR-дампер БТИ: raw grayscale + Adaptive Otsu + PSM 11 + TSV.")
    ap.add_argument("--csv", required=True, help="CSV с колонкой «путь»")
    ap.add_argument("--path-column", default="путь")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--outdir", default="bti_ocr_dumps", help="Корень папок OCR-дампов")
    ap.add_argument("--workers", type=int, default=2, help="Одновременные PDF; при 400 DPI начните с 2")
    ap.add_argument("--dpi", type=int, default=400, help="Рендер DPI; найденный рабочий дефолт: 400")
    ap.add_argument("--ocr-lang", default="rus", help="Язык Tesseract; найденный рабочий дефолт: rus")
    ap.add_argument("--ocr-psm", type=int, default=11, help="PSM; найденный рабочий дефолт: 11")
    ap.add_argument("--thresholding-method", type=int, default=1, help="Tesseract thresholding_method; 1 = Adaptive Otsu")
    ap.add_argument("--max-megapixels", type=float, default=20.0, help="Пропустить листы крупнее этого лимита")
    ap.add_argument("--timeout", type=int, default=180, help="Максимум секунд на OCR одной страницы")
    ap.add_argument("--min-chars", type=int, default=15, help="Меньше символов = статус low_text")
    ap.add_argument("--control-count", type=int, default=4, help="Число контрольных страниц с PNG")
    ap.add_argument("--control-end-offset", type=int, default=2, help="Последняя контрольная: page_count - это число")
    ap.add_argument("--retry-failed", action="store_true", help="Повторить только timeout/error")
    ap.add_argument("--overwrite", action="store_true", help="Перераспознать все страницы, игнорируя кеш")
    ap.add_argument("--force-pages", default="", help="Страницы через запятую; OCR даже при превышении лимита MP")
    ap.add_argument("--tesseract-cmd", default=None, help="Путь к tesseract.exe, если он не рядом")
    args = ap.parse_args()

    if len(args.sep) != 1:
        raise SystemExit("--sep должен быть одним символом")
    if args.workers < 1:
        raise SystemExit("--workers должен быть не меньше 1")
    if args.dpi < 150:
        raise SystemExit("--dpi слишком низкий")
    if args.max_megapixels <= 0:
        raise SystemExit("--max-megapixels должен быть больше 0")
    if args.control_count < 0:
        raise SystemExit("--control-count не может быть отрицательным")
    try:
        force_pages = sorted({int(x.strip()) for x in args.force_pages.split(",") if x.strip()})
    except ValueError:
        raise SystemExit("--force-pages: номера страниц через запятую, например 17,18")

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"Нет CSV: {csv_path}")
    paths = read_paths(csv_path, args.path_column, args.sep)
    if not paths:
        raise SystemExit("В CSV нет непустых путей")

    tesseract = find_tesseract(Path(__file__).resolve().parent, args.tesseract_cmd)
    validate_tesseract(tesseract, args.ocr_lang)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "dpi": args.dpi,
        "ocr_lang": args.ocr_lang,
        "psm": args.ocr_psm,
        "thresholding_method": args.thresholding_method,
        "max_megapixels": args.max_megapixels,
        "timeout": args.timeout,
        "min_chars": args.min_chars,
        "control_count": args.control_count,
        "control_end_offset": args.control_end_offset,
        "retry_failed": args.retry_failed,
        "overwrite": args.overwrite,
        "force_pages": force_pages,
        "tesseract": str(tesseract),
    }
    tasks = [
        {"pdf": str(path), "output_folder": str(output_folder_for(path, outdir)), "cfg": cfg}
        for path in paths
    ]

    run_summary = outdir / "run_summary.csv"
    fields = [
        "source_pdf", "status", "output_folder", "processed_pages", "ok_pages",
        "low_text_pages", "skipped_too_large_pages", "error_pages", "reason",
    ]
    counts = {"ok": 0, "error": 0, "missing": 0}
    with run_summary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_pdf_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="BTI OCR", unit="pdf"):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "pdf": "", "output_folder": "", "status": "error", "processed": 0,
                        "ok": 0, "low": 0, "large": 0, "errors": 1,
                        "reason": f"executor: {type(exc).__name__}: {exc}",
                    }
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                writer.writerow({
                    "source_pdf": result.get("pdf", ""),
                    "status": result.get("status", ""),
                    "output_folder": result.get("output_folder", ""),
                    "processed_pages": result.get("processed", 0),
                    "ok_pages": result.get("ok", 0),
                    "low_text_pages": result.get("low", 0),
                    "skipped_too_large_pages": result.get("large", 0),
                    "error_pages": result.get("errors", 0),
                    "reason": result.get("reason", ""),
                })
                f.flush()

    print("Готово:", ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"Сводка запуска: {run_summary}")


if __name__ == "__main__":
    main()
