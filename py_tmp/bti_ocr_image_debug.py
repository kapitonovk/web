#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый OCR-дампер БТИ: raw grayscale + встроенный Adaptive Otsu Tesseract + PSM 11.

Для первых 40 страниц каждого PDF из CSV (колонка «путь») сохраняет:
  render.png       — исходный grayscale-рендер PDF
  prepared.png     — изображение, поданное Tesseract (то же raw grayscale)
  tessinput.tif    — внутренний thresholded bitmap Tesseract, если сборка его отдаёт
  result.tsv       — слова, координаты, confidence
  text.txt         — сырой plain text Tesseract
  page_order.txt   — пространственная текстовая имитация страницы по TSV
  meta.json        — параметры, размеры, время и базовые метрики

Настройки намеренно фиксированы:
  - 400 DPI
  - rus
  - PSM 11
  - raw grayscale без нашей обработки
  - Tesseract Adaptive Otsu: thresholding_method=1
  - без вращения и без контентных эвристик

Зависимости:
  py -m pip install pymupdf pytesseract pillow tqdm

Tesseract рядом со скриптом:
  .\tesseract\tesseract.exe
  .\tesseract\tessdata\rus.traineddata

Пример:
  py bti_ocr_image_debug.py --csv test_two_pdfs.csv --sep ";" --outdir ocr_otsu_test

Повторить поверх существующих результатов:
  py bti_ocr_image_debug.py --csv test_two_pdfs.csv --sep ";" --outdir ocr_otsu_test --overwrite
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import time
import unicodedata
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


DPI = 400
PSM = 11
LANG = "rus"
PAGES_PER_PDF = 40
TESS_CONFIG = "--oem 1 --psm 11 -c preserve_interword_spaces=1 -c tessedit_write_images=1 -c thresholding_method=1"


def safe_name(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", " ", text).strip(" ._")
    return text[:80] or "pdf"


def short_hash(path):
    return hashlib.sha1(str(path).encode("utf-8", errors="replace")).hexdigest()[:10]


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


def validate_tesseract(exe):
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
    if LANG not in languages:
        raise SystemExit("В Tesseract нет rus.traineddata")


def render_page(page):
    scale = DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False)
    try:
        return Image.frombytes("L", [pix.width, pix.height], pix.samples)
    finally:
        pix = None


def move_tessinput(workdir, page_dir):
    candidates = list(workdir.glob("tessinput.*")) + list(workdir.glob("*.processed.tif"))
    if not candidates:
        return ""
    source = max(candidates, key=lambda path: path.stat().st_mtime)
    target = page_dir / f"tessinput{source.suffix or '.tif'}"
    shutil.move(str(source), str(target))
    return target.name


def parse_tsv_words(tsv):
    reader = csv.DictReader(tsv.splitlines(), delimiter="\t")
    words = []
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text or row.get("level") != "5":
            continue
        try:
            left = int(row.get("left", "0"))
            top = int(row.get("top", "0"))
            width = int(row.get("width", "0"))
            height = int(row.get("height", "0"))
        except ValueError:
            continue
        words.append({
            "text": text,
            "left": left,
            "top": top,
            "right": left + width,
            "bottom": top + height,
            "height": height,
        })
    return words


def render_page_order(tsv, page_width, page_height, columns=140, vertical_scale=1.8):
    """Моноширинная пространственная реконструкция: сохраняет X/Y-положение слов."""
    words = parse_tsv_words(tsv)
    if not words:
        return "[В TSV нет распознанных слов]\n"
    columns = max(columns, 40)
    char_px = max(1.0, page_width / columns)
    row_px = max(1.0, char_px * vertical_scale)
    rows = max(1, math.ceil(page_height / row_px) + 1)
    canvas = [[] for _ in range(rows)]
    for word in sorted(words, key=lambda w: (w["top"], w["left"])):
        row = min(rows - 1, max(0, int(word["top"] / row_px)))
        col = min(columns - 1, max(0, int(word["left"] / char_px)))
        canvas[row].append((col, word["text"]))
    output = []
    previous_nonempty = None
    for row_no, items in enumerate(canvas):
        if not items:
            continue
        if previous_nonempty is not None:
            output.extend([""] * min(3, max(0, row_no - previous_nonempty - 1)))
        cursor = 0
        fragments = []
        for col, text in sorted(items, key=lambda item: item[0]):
            col = max(col, cursor + (1 if fragments else 0))
            if col > cursor:
                fragments.append(" " * (col - cursor))
                cursor = col
            fragments.append(text)
            cursor += len(text)
        output.append("".join(fragments).rstrip())
        previous_nonempty = row_no
    return "\n".join(output).rstrip() + "\n"


def count_metrics(text, tsv):
    rows = list(csv.DictReader(tsv.splitlines(), delimiter="\t"))
    words = [row for row in rows if row.get("level") == "5" and (row.get("text") or "").strip()]
    digit_words = [row for row in words if re.search(r"\d", row.get("text") or "")]
    return {
        "chars": len(text),
        "digits": len(re.findall(r"\d", text)),
        "commas_dots": len(re.findall(r"[,.]", text)),
        "tsv_words": len(words),
        "tsv_digit_words": len(digit_words),
    }


def process_page(page, page_dir, args):
    page_dir.mkdir(parents=True, exist_ok=True)
    raw = render_page(page)
    try:
        raw.save(page_dir / "render.png")
        # В этом режиме prepared и render намеренно одинаковы.
        raw.save(page_dir / "prepared.png")
        with tempfile.TemporaryDirectory(prefix="bti_otsu_") as temp:
            workdir = Path(temp)
            previous = Path.cwd()
            try:
                os.chdir(workdir)
                started = time.time()
                text = pytesseract.image_to_string(raw, lang=LANG, config=TESS_CONFIG, timeout=args.timeout)
                tsv = pytesseract.image_to_data(raw, lang=LANG, config=TESS_CONFIG, timeout=args.timeout, output_type=pytesseract.Output.STRING)
                seconds = time.time() - started
                tessinput = move_tessinput(workdir, page_dir)
            finally:
                os.chdir(previous)
        text = norm_text(text)
        (page_dir / "text.txt").write_text(text, encoding="utf-8")
        (page_dir / "result.tsv").write_text(tsv, encoding="utf-8")
        page_order = render_page_order(tsv, raw.width, raw.height)
        (page_dir / "page_order.txt").write_text(page_order, encoding="utf-8")
        metrics = count_metrics(text, tsv)
        meta = {
            "page": page.number + 1,
            "ocr_lang": LANG,
            "ocr_psm": PSM,
            "dpi": DPI,
            "preprocessing": "raw_grayscale",
            "tesseract_thresholding": "adaptive_otsu (thresholding_method=1)",
            "tesseract_config": TESS_CONFIG,
            "page_size_pt": {"width": page.rect.width, "height": page.rect.height},
            "render_size_px": {"width": raw.width, "height": raw.height},
            "megapixels": round((raw.width * raw.height) / 1_000_000, 2),
            "seconds": round(seconds, 2),
            "tessinput_file": tessinput,
            **metrics,
        }
        (page_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "ok", **meta}
    finally:
        raw.close()


def process_pdf(pdf_path, outdir, args):
    pdf_dir = outdir / f"{safe_name(pdf_path.stem)}__{short_hash(pdf_path)}"
    pages_dir = pdf_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    results = []
    manifest = {
        "source_pdf": str(pdf_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "raw_grayscale + adaptive_otsu + psm11",
        "pages_per_pdf": args.pages_per_pdf,
        "status": "error",
    }
    if not pdf_path.is_file():
        manifest["error"] = "PDF not found"
        (pdf_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return [{"source_pdf": str(pdf_path), "status": "missing", "reason": "PDF not found"}]
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        manifest["error"] = f"open PDF: {type(exc).__name__}: {exc}"
        (pdf_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return [{"source_pdf": str(pdf_path), "status": "error", "reason": manifest["error"]}]
    try:
        limit = min(args.pages_per_pdf, doc.page_count)
        for page_idx in range(limit):
            page_dir = pages_dir / f"page_{page_idx+1:04d}"
            meta_path = page_dir / "meta.json"
            if meta_path.is_file() and not args.overwrite:
                try:
                    old = json.loads(meta_path.read_text(encoding="utf-8"))
                    results.append({"source_pdf": str(pdf_path), "page": page_idx + 1, "status": "skipped_existing", **old})
                    continue
                except Exception:
                    pass
            try:
                result = process_page(doc.load_page(page_idx), page_dir, args)
                results.append({"source_pdf": str(pdf_path), "page": page_idx + 1, **result})
            except Exception as exc:
                page_dir.mkdir(parents=True, exist_ok=True)
                error = {"source_pdf": str(pdf_path), "page": page_idx + 1, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
                meta_path.write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
                results.append(error)
        manifest.update({"status": "ok", "page_count": doc.page_count, "processed_pages": limit})
        (pdf_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return results
    finally:
        doc.close()


def main():
    ap = argparse.ArgumentParser(description="Тестовый БТИ OCR: raw grayscale + Otsu + PSM 11, первые 40 страниц PDF.")
    ap.add_argument("--csv", required=True, help="CSV с колонкой «путь»")
    ap.add_argument("--path-column", default="путь")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--outdir", default="ocr_otsu_test")
    ap.add_argument("--pages-per-pdf", type=int, default=PAGES_PER_PDF, help="По умолчанию 40")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tesseract-cmd", default=None)
    args = ap.parse_args()

    if len(args.sep) != 1:
        raise SystemExit("--sep должен быть одним символом")
    if args.pages_per_pdf < 1:
        raise SystemExit("--pages-per-pdf должен быть не меньше 1")
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"Нет CSV: {csv_path}")
    paths = read_paths(csv_path, args.path_column, args.sep)
    if not paths:
        raise SystemExit("В CSV нет непустых путей")

    tesseract = find_tesseract(Path(__file__).resolve().parent, args.tesseract_cmd)
    validate_tesseract(tesseract)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for pdf_path in tqdm(paths, desc="Otsu OCR test", unit="pdf"):
        all_results.extend(process_pdf(pdf_path, outdir, args))

    summary_path = outdir / "summary.csv"
    fields = ["source_pdf", "page", "status", "chars", "digits", "commas_dots", "tsv_words", "tsv_digit_words", "seconds", "megapixels", "tessinput_file", "reason"]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Готово: {summary_path}")
    print("На страницу: render.png, prepared.png, result.tsv, text.txt, page_order.txt, meta.json и (если доступен) tessinput.tif.")


if __name__ == "__main__":
    main()
