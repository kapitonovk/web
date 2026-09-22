#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Качественный OCR БТИ-PDF: PSM 11 + TSV + восстановленный порядок чтения.

Вход: CSV с колонкой «путь».

Для каждого PDF создаёт в --dump-root/<safe_stem>__<hash>/:
  ocr_quality/
    page_0001.txt                 сырой plain-text Tesseract (PSM 11)
    page_0001.tsv                 TSV: все слова + координаты + confidence
    page_0001_reading_order.txt   TSV, собранный в визуальные строки
    page_0001.json                метаданные и статус страницы
  ocr_quality_index.csv           постраничный кеш / индекс
  ocr_quality_manifest.json       сводка OCR одного PDF

Логика:
  - PSM 11 по умолчанию: максимум найденного текста и мелких цифр.
  - Рендер 350 DPI, grayscale, autocontrast, умеренный contrast 1.35.
  - Нет бинаризации, удаления линий, поиска таблиц или контентных фильтров.
  - Единственный автоматический skip: слишком большой лист по расчётному
    числу мегапикселей (обычно A3/A2 чертежи).
  - Не вращает страницы.
  - reading_order.txt собирается ИЗ TSV: слова группируются в визуальные
    строки по вертикальной близости, затем сортируются слева направо.
    Он улучшает читаемость PSM 11, но raw TXT и TSV всегда сохраняются.

Кеширование:
  - ok, low_text, skipped_too_large не обрабатываются повторно.
  - --retry-failed повторяет только timeout/error.
  - --overwrite ПЕРЕРАСПОЗНАЁТ все страницы, кроме больших листов;
    --force-pages обрабатывает указанные страницы даже сверх лимита.

Зависимости:
  py -m pip install pymupdf pytesseract pillow tqdm

Tesseract рядом со скриптом:
  .\tesseract\tesseract.exe
  .\tesseract\tessdata\rus.traineddata
  .\tesseract\tessdata\eng.traineddata

Примеры:
  # Основной запуск
  py bti_ocr_selected.py --csv ocr_candidates.csv --sep ";" --dump-root bti_dumps --workers 2

  # Перераспознать страницы после смены настроек
  py bti_ocr_selected.py --csv ocr_candidates.csv --sep ";" --dump-root bti_dumps --workers 2 --overwrite

  # OCR конкретного большого листа
  py bti_ocr_selected.py --csv one_pdf.csv --sep ";" --dump-root bti_dumps --workers 1 --force-pages "37"
"""

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import fitz
import pytesseract
from PIL import Image, ImageEnhance, ImageOps

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items, **kwargs):
        return items


OCR_COLUMNS = [
    "page", "status", "dpi", "psm", "megapixels", "chars", "tsv_words",
    "reading_order_lines", "seconds", "reason", "text_file", "tsv_file",
    "reading_order_file", "meta_file", "updated_at",
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


def dump_folder_for(pdf_path, dump_root):
    return dump_root / f"{safe_name(pdf_path.stem)}__{short_hash(pdf_path)}"


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
            value = (row.get(column) or "").strip().strip('"')
            if value and value not in seen:
                paths.append(Path(value))
                seen.add(value)
    return paths


def find_tesseract(script_dir, supplied):
    if supplied:
        path = Path(supplied)
        return path if path.is_file() else None
    for path in [
        script_dir / "tesseract" / "tesseract.exe",
        script_dir / "Tesseract-OCR" / "tesseract.exe",
        script_dir / "tesseract.exe",
    ]:
        if path.is_file():
            return path
    return None


def validate_tesseract(exe, lang):
    if not exe:
        raise SystemExit("Tesseract не найден. Нужен .\\tesseract\\tesseract.exe или --tesseract-cmd.")
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
        writer = csv.DictWriter(f, fieldnames=OCR_COLUMNS)
        writer.writeheader()
        for page_no in sorted(rows):
            writer.writerow({field: rows[page_no].get(field, "") for field in OCR_COLUMNS})


def predicted_megapixels(page, dpi):
    return ((page.rect.width * dpi / 72) * (page.rect.height * dpi / 72)) / 1_000_000


def render_page(page, dpi):
    scale = dpi / 72
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        colorspace=fitz.csGRAY,
        alpha=False,
    )
    try:
        return Image.frombytes("L", [pix.width, pix.height], pix.samples)
    finally:
        pix = None


def prepare_for_ocr(image):
    """Мягкая обработка: усиливает слабый скан, не уничтожая дробные знаки."""
    image = ImageOps.autocontrast(image, cutoff=0.5)
    return ImageEnhance.Contrast(image).enhance(1.35)


def parse_tsv_words(tsv):
    """Возвращает все непустые word-level (level=5) записи Tesseract TSV."""
    reader = csv.DictReader(tsv.splitlines(), delimiter="\t")
    words = []
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text or row.get("level") != "5":
            continue
        try:
            conf = float(row.get("conf", "-1"))
            left = int(row.get("left", "0"))
            top = int(row.get("top", "0"))
            width = int(row.get("width", "0"))
            height = int(row.get("height", "0"))
        except ValueError:
            continue
        words.append({
            "text": text, "conf": conf, "left": left, "top": top,
            "width": width, "height": height,
            "cx": left + width / 2, "cy": top + height / 2,
        })
    return words


def reading_order_from_tsv(tsv):
    """
    Собирает PSM 11 TSV в визуальные строки.

    Не фильтрует слова по confidence: для БТИ важны мелкие цифры, а итоговое
    решение о качестве лучше принимать позднее по raw TSV и контексту.
    """
    words = parse_tsv_words(tsv)
    if not words:
        return "", 0

    heights = [w["height"] for w in words if w["height"] > 0]
    median_height = statistics.median(heights) if heights else 20
    # Допуск по вертикали: и в пикселях, и пропорционально реальной высоте букв.
    y_tolerance = max(8.0, median_height * 0.65)

    words.sort(key=lambda w: (w["cy"], w["left"]))
    lines = []
    for word in words:
        candidates = []
        for line in lines:
            delta = abs(word["cy"] - line["cy"])
            if delta <= y_tolerance:
                candidates.append((delta, line))
        if candidates:
            _, line = min(candidates, key=lambda item: item[0])
            line["words"].append(word)
            line["cy"] = sum(w["cy"] for w in line["words"]) / len(line["words"])
            line["top"] = min(line["top"], word["top"])
            line["bottom"] = max(line["bottom"], word["top"] + word["height"])
        else:
            lines.append({
                "cy": word["cy"], "top": word["top"], "bottom": word["top"] + word["height"], "words": [word]
            })

    lines.sort(key=lambda line: (line["top"], min(w["left"] for w in line["words"])))
    rendered = []
    previous_bottom = None
    paragraph_gap = max(median_height * 1.8, 28)
    for line in lines:
        line["words"].sort(key=lambda w: w["left"])
        if previous_bottom is not None and line["top"] - previous_bottom > paragraph_gap:
            rendered.append("")
        rendered.append(" ".join(w["text"] for w in line["words"]))
        previous_bottom = max(previous_bottom or 0, line["bottom"])
    return "\n".join(rendered).strip(), len(lines)


def ocr_page(page, dpi, lang, psm, timeout):
    raw = render_page(page, dpi)
    image = prepare_for_ocr(raw)
    if image is not raw:
        raw.close()
    megapixels = (image.width * image.height) / 1_000_000
    config = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
    try:
        # Два формата вызываются отдельно: raw text удобен для контроля,
        # TSV — источник истины для порядка, строк и будущих таблиц.
        text = pytesseract.image_to_string(image, lang=lang, config=config, timeout=timeout)
        tsv = pytesseract.image_to_data(
            image,
            lang=lang,
            config=config,
            timeout=timeout,
            output_type=pytesseract.Output.STRING,
        )
        return norm_text(text), tsv, megapixels
    finally:
        image.close()


def write_page_files(ocr_dir, row, text=None, tsv=None, reading_order=None):
    page_no = int(row["page"])
    if text is not None:
        path = ocr_dir / f"page_{page_no:04d}.txt"
        path.write_text(text, encoding="utf-8")
        row["text_file"] = path.name
    if tsv is not None:
        path = ocr_dir / f"page_{page_no:04d}.tsv"
        path.write_text(tsv, encoding="utf-8")
        row["tsv_file"] = path.name
    if reading_order is not None:
        path = ocr_dir / f"page_{page_no:04d}_reading_order.txt"
        path.write_text(reading_order, encoding="utf-8")
        row["reading_order_file"] = path.name
    path = ocr_dir / f"page_{page_no:04d}.json"
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    row["meta_file"] = path.name


def should_process(existing, retry_failed, overwrite, force):
    if force or overwrite or not existing:
        return True
    status = existing.get("status", "")
    if retry_failed and status in RETRYABLE_STATUSES:
        return True
    return status not in DONE_STATUSES and status not in RETRYABLE_STATUSES


def worker(task):
    pdf_path = Path(task["pdf"])
    out_folder = Path(task["folder"])
    cfg = task["cfg"]
    force_pages = set(cfg["force_pages"])

    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    pytesseract.pytesseract.tesseract_cmd = cfg["tesseract"]

    ocr_dir = out_folder / "ocr_quality"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_folder / "ocr_quality_index.csv"
    index = load_index(index_path)
    outcome = {
        "pdf": str(pdf_path), "folder": str(out_folder), "status": "ok",
        "processed": 0, "ok": 0, "low": 0, "large": 0, "errors": 0,
    }
    started = time.time()

    if not pdf_path.is_file():
        outcome.update({"status": "missing", "errors": 1, "reason": "PDF не найден"})
        return outcome
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        outcome.update({"status": "error", "errors": 1, "reason": f"open PDF: {type(exc).__name__}: {exc}"})
        return outcome

    try:
        for page_idx in range(doc.page_count):
            page_no = page_idx + 1
            forced = page_no in force_pages
            if not should_process(index.get(page_no), cfg["retry_failed"], cfg["overwrite"], forced):
                continue

            page = doc.load_page(page_idx)
            expected_mp = predicted_megapixels(page, cfg["dpi"])
            row = {
                "page": page_no,
                "status": "",
                "dpi": cfg["dpi"],
                "psm": cfg["psm"],
                "megapixels": round(expected_mp, 2),
                "chars": "",
                "tsv_words": "",
                "reading_order_lines": "",
                "seconds": "",
                "reason": "",
                "text_file": "",
                "tsv_file": "",
                "reading_order_file": "",
                "meta_file": "",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            outcome["processed"] += 1

            # Единственный автоматический skip: физически большой лист.
            if expected_mp > cfg["max_megapixels"] and not forced:
                row.update({
                    "status": "skipped_too_large",
                    "reason": f"{expected_mp:.2f} MP > limit {cfg['max_megapixels']:.2f} MP",
                })
                write_page_files(ocr_dir, row)
                index[page_no] = row
                save_index(index_path, index)
                outcome["large"] += 1
                continue

            t0 = time.time()
            try:
                text, tsv, actual_mp = ocr_page(
                    page, cfg["dpi"], cfg["lang"], cfg["psm"], cfg["timeout"]
                )
                reading_order, line_count = reading_order_from_tsv(tsv)
                row.update({
                    "megapixels": round(actual_mp, 2),
                    "chars": len(text),
                    "tsv_words": len(parse_tsv_words(tsv)),
                    "reading_order_lines": line_count,
                    "seconds": round(time.time() - t0, 2),
                })
                if len(text) < cfg["min_chars"]:
                    row.update({"status": "low_text", "reason": f"распознано только {len(text)} символов"})
                    outcome["low"] += 1
                else:
                    row["status"] = "ok"
                    outcome["ok"] += 1
                write_page_files(ocr_dir, row, text, tsv, reading_order)
            except RuntimeError as exc:
                message = str(exc)
                row.update({
                    "status": "timeout" if "timeout" in message.casefold() else "error",
                    "seconds": round(time.time() - t0, 2),
                    "reason": f"{type(exc).__name__}: {message}",
                })
                write_page_files(ocr_dir, row)
                outcome["errors"] += 1
            except Exception as exc:
                row.update({
                    "status": "error",
                    "seconds": round(time.time() - t0, 2),
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                write_page_files(ocr_dir, row)
                outcome["errors"] += 1

            index[page_no] = row
            save_index(index_path, index)

        save_index(index_path, index)
        manifest = {
            "source_pdf": str(pdf_path),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.time() - started, 2),
            "page_count": doc.page_count,
            "processed_this_run": outcome["processed"],
            "ok_this_run": outcome["ok"],
            "low_text_this_run": outcome["low"],
            "skipped_too_large_this_run": outcome["large"],
            "errors_this_run": outcome["errors"],
            "settings": {key: value for key, value in cfg.items() if key != "tesseract"},
        }
        (out_folder / "ocr_quality_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return outcome
    except Exception as exc:
        outcome.update({
            "status": "error", "errors": outcome["errors"] + 1,
            "reason": f"worker: {type(exc).__name__}: {exc}",
        })
        return outcome
    finally:
        doc.close()


def main():
    ap = argparse.ArgumentParser(
        description="Качественный OCR БТИ: PSM 11, TSV, reading order и пропуск больших листов."
    )
    ap.add_argument("--csv", required=True, help="CSV с колонкой «путь»")
    ap.add_argument("--path-column", default="путь")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--dump-root", required=True, help="Папка результатов native-дампа")
    ap.add_argument("--workers", type=int, default=2, help="Одновременные PDF; для 350 DPI начать с 2")
    ap.add_argument("--dpi", type=int, default=350)
    ap.add_argument("--max-megapixels", type=float, default=20.0)
    ap.add_argument("--ocr-lang", default="rus+eng")
    ap.add_argument("--ocr-psm", type=int, default=11, help="PSM 11 по умолчанию: максимальная полнота OCR")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--min-chars", type=int, default=15)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="Перераспознать все страницы, игнорируя кеш")
    ap.add_argument("--force-pages", default="", help="Страницы через запятую; OCR даже сверх лимита MP")
    ap.add_argument("--tesseract-cmd", default=None)
    args = ap.parse_args()

    if len(args.sep) != 1:
        raise SystemExit("--sep должен быть одним символом")
    if args.workers < 1:
        raise SystemExit("--workers должен быть не меньше 1")
    if args.dpi < 300:
        raise SystemExit("Для качественного режима --dpi должен быть не менее 300")
    if args.max_megapixels <= 0:
        raise SystemExit("--max-megapixels должен быть больше 0")
    try:
        force_pages = sorted({int(part.strip()) for part in args.force_pages.split(",") if part.strip()})
    except ValueError:
        raise SystemExit("--force-pages: номера страниц через запятую, например 17,18")

    csv_path = Path(args.csv)
    dump_root = Path(args.dump_root)
    if not csv_path.is_file():
        raise SystemExit(f"Нет CSV: {csv_path}")
    if not dump_root.is_dir():
        raise SystemExit(f"Нет папки дампов: {dump_root}")
    paths = read_paths(csv_path, args.path_column, args.sep)
    if not paths:
        raise SystemExit("В CSV нет непустых путей")

    tesseract = find_tesseract(Path(__file__).resolve().parent, args.tesseract_cmd)
    validate_tesseract(tesseract, args.ocr_lang)

    cfg = {
        "dpi": args.dpi,
        "max_megapixels": args.max_megapixels,
        "lang": args.ocr_lang,
        "psm": args.ocr_psm,
        "timeout": args.timeout,
        "min_chars": args.min_chars,
        "retry_failed": args.retry_failed,
        "overwrite": args.overwrite,
        "force_pages": force_pages,
        "tesseract": str(tesseract),
    }
    tasks = [
        {"pdf": str(path), "folder": str(dump_folder_for(path, dump_root)), "cfg": cfg}
        for path in paths
    ]

    summary_path = dump_root / "ocr_quality_run_summary.csv"
    fields = [
        "source_pdf", "status", "output_folder", "processed_pages", "ok_pages",
        "low_text_pages", "skipped_too_large_pages", "error_pages", "reason",
    ]
    counts = {"ok": 0, "error": 0, "missing": 0}
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="BTI OCR", unit="pdf"):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "pdf": "", "folder": "", "status": "error", "processed": 0,
                        "ok": 0, "low": 0, "large": 0, "errors": 1,
                        "reason": f"executor: {type(exc).__name__}: {exc}",
                    }
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                writer.writerow({
                    "source_pdf": result.get("pdf", ""),
                    "status": result.get("status", ""),
                    "output_folder": result.get("folder", ""),
                    "processed_pages": result.get("processed", 0),
                    "ok_pages": result.get("ok", 0),
                    "low_text_pages": result.get("low", 0),
                    "skipped_too_large_pages": result.get("large", 0),
                    "error_pages": result.get("errors", 0),
                    "reason": result.get("reason", ""),
                })
                f.flush()

    print("Готово:", ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"Сводка: {summary_path}")


if __name__ == "__main__":
    main()
