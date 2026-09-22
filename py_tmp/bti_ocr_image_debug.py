#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный image-debug Tesseract для БТИ-сканов.

Каждая из первых N страниц КАЖДОГО PDF тестируется во ВСЕХ вариантах
изображения при одинаковых настройках OCR: rus, PSM 11.

Варианты:
  1. base            Исходный grayscale 400 DPI, без нашей обработки.
  2. contrast        Autocontrast + сильный contrast 2.6.
  3. invert          Инверсия -> autocontrast + contrast 2.6.
  4. invert_back     Обработка в инверсной полярности -> инверсия обратно.
                      Это оставляет Tesseract тёмный текст на светлом фоне.
  5. tess_sauvola    Исходный grayscale; встроенная Sauvola binarization
                      Tesseract.
  6. tess_otsu       Исходный grayscale; встроенная Adaptive Otsu
                      Tesseract.

Для каждого варианта сохраняются:
  render.png          исходный grayscale рендер PDF
  prepared.png        изображение, переданное в Tesseract
  text.txt            plain text
  result.tsv          слова + координаты + confidence
  meta.json           конфигурация, время, метрики
  tessinput.tif       внутренний thresholded bitmap Tesseract, если удалось

ВАЖНО: tessinput.tif создаётся Tesseract в рабочей папке процесса. Скрипт
запускает OCR-вызовы последовательно и переносит созданный tif в папку
соответствующего варианта. В некоторых сборках Tesseract файл может не
создаваться; тогда это отмечается в meta.json.

Вход: CSV с колонкой «путь».

Зависимости:
  py -m pip install pymupdf pytesseract pillow tqdm

Tesseract рядом со скриптом:
  .\tesseract\tesseract.exe
  .\tesseract\tessdata\rus.traineddata

Пример:
  py bti_ocr_image_debug.py --csv test_two_pdfs.csv --sep ";" --outdir ocr_image_debug
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


VARIANTS = [
    {
        "id": "01_base",
        "label": "raw grayscale: no preprocessing",
        "prepare": "base",
        "tess_config": "",
    },
    {
        "id": "02_contrast",
        "label": "autocontrast + strong contrast 2.6",
        "prepare": "contrast",
        "tess_config": "",
    },
    {
        "id": "03_invert",
        "label": "invert -> autocontrast + strong contrast 2.6 (white text on black)",
        "prepare": "invert",
        "tess_config": "",
    },
    {
        "id": "04_invert_back",
        "label": "invert -> enhance -> invert back (dark text on light)",
        "prepare": "invert_back",
        "tess_config": "",
    },
    {
        "id": "05_tess_sauvola",
        "label": "raw grayscale + Tesseract Sauvola thresholding",
        "prepare": "base",
        "tess_config": "-c thresholding_method=2 -c thresholding_window_size=0.33 -c thresholding_kfactor=0.34",
    },
    {
        "id": "06_tess_otsu",
        "label": "raw grayscale + Tesseract Adaptive Otsu thresholding",
        "prepare": "base",
        "tess_config": "-c thresholding_method=1",
    },
]


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
    if lang not in languages:
        raise SystemExit(f"В Tesseract нет языка: {lang}")


def render_page(page, dpi):
    scale = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False)
    try:
        return Image.frombytes("L", [pix.width, pix.height], pix.samples)
    finally:
        pix = None


def strong_contrast(image):
    image = ImageOps.autocontrast(image, cutoff=0.5)
    return ImageEnhance.Contrast(image).enhance(2.6)


def prepare_image(raw, method):
    if method == "base":
        return raw.copy()
    if method == "contrast":
        return strong_contrast(raw)
    if method == "invert":
        inverted = ImageOps.invert(raw)
        try:
            return strong_contrast(inverted)
        finally:
            inverted.close()
    if method == "invert_back":
        inverted = ImageOps.invert(raw)
        try:
            enhanced = strong_contrast(inverted)
        finally:
            inverted.close()
        try:
            return ImageOps.invert(enhanced)
        finally:
            enhanced.close()
    raise ValueError(f"Unknown prepare mode: {method}")


def count_tsv(tsv):
    word_count = digit_words = numeric_like = 0
    reader = csv.DictReader(tsv.splitlines(), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        word_count += 1
        if re.search(r"\d", text):
            digit_words += 1
        if re.fullmatch(r"[0-9OОЗЗбБIlІl.,:;\-]+", text):
            numeric_like += 1
    return word_count, digit_words, numeric_like


def move_tessinput(workdir, variant_dir):
    """Переносит thresholded-image, если сборка Tesseract его создала."""
    candidates = list(workdir.glob("tessinput.*")) + list(workdir.glob("*.processed.tif"))
    if not candidates:
        return ""
    src = max(candidates, key=lambda p: p.stat().st_mtime)
    suffix = src.suffix or ".tif"
    dest = variant_dir / f"tessinput{suffix}"
    shutil.move(str(src), str(dest))
    return dest.name


def run_variant(page, raw, variant, args, page_dir):
    page_dir.mkdir(parents=True, exist_ok=True)
    render_path = page_dir / "render.png"
    raw.save(render_path)
    prepared = prepare_image(raw, variant["prepare"])
    try:
        prepared.save(page_dir / "prepared.png")
        config = (
            f"--oem 1 --psm {args.ocr_psm} "
            f"-c preserve_interword_spaces=1 "
            f"-c tessedit_write_images=1 "
            f"{variant['tess_config']}"
        )
        with tempfile.TemporaryDirectory(prefix="bti_tess_") as temp:
            workdir = Path(temp)
            previous = Path.cwd()
            try:
                os.chdir(workdir)
                t0 = time.time()
                text = pytesseract.image_to_string(prepared, lang=args.ocr_lang, config=config, timeout=args.timeout)
                tsv = pytesseract.image_to_data(prepared, lang=args.ocr_lang, config=config, timeout=args.timeout, output_type=pytesseract.Output.STRING)
                elapsed = time.time() - t0
                tessinput_file = move_tessinput(workdir, page_dir)
            finally:
                os.chdir(previous)
        text = norm_text(text)
        (page_dir / "text.txt").write_text(text, encoding="utf-8")
        (page_dir / "result.tsv").write_text(tsv, encoding="utf-8")
        words, digit_words, numeric_like = count_tsv(tsv)
        meta = {
            "page": page.number + 1,
            "variant": variant,
            "ocr_lang": args.ocr_lang,
            "ocr_psm": args.ocr_psm,
            "page_size_pt": {"width": page.rect.width, "height": page.rect.height},
            "render_size_px": {"width": prepared.width, "height": prepared.height},
            "megapixels": round((prepared.width * prepared.height) / 1_000_000, 2),
            "seconds": round(elapsed, 2),
            "chars": len(text),
            "digits": len(re.findall(r"\d", text)),
            "commas_dots": len(re.findall(r"[,.]", text)),
            "tsv_words": words,
            "tsv_digit_words": digit_words,
            "tsv_numeric_like_words": numeric_like,
            "tessinput_file": tessinput_file,
        }
        (page_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "ok", **meta}
    finally:
        prepared.close()


def process_pdf(pdf_path, outdir, args):
    pdf_dir = outdir / f"{safe_name(pdf_path.stem)}__{short_hash(pdf_path)}"
    pages_dir = pdf_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    results = []
    manifest = {
        "source_pdf": str(pdf_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "variants": VARIANTS,
        "pages_per_pdf": args.pages_per_pdf,
        "dpi": args.dpi,
        "psm": args.ocr_psm,
        "language": args.ocr_lang,
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
            page = doc.load_page(page_idx)
            raw = render_page(page, args.dpi)
            try:
                for variant in VARIANTS:
                    variant_dir = pages_dir / f"page_{page_idx+1:04d}" / variant["id"]
                    if (variant_dir / "meta.json").is_file() and not args.overwrite:
                        try:
                            old = json.loads((variant_dir / "meta.json").read_text(encoding="utf-8"))
                            results.append({"source_pdf": str(pdf_path), "page": page_idx + 1, "variant": variant["id"], "status": "skipped_existing", **old})
                            continue
                        except Exception:
                            pass
                    try:
                        result = run_variant(page, raw, variant, args, variant_dir)
                        results.append({"source_pdf": str(pdf_path), "page": page_idx + 1, "variant": variant["id"], **result})
                    except Exception as exc:
                        variant_dir.mkdir(parents=True, exist_ok=True)
                        error = {
                            "source_pdf": str(pdf_path), "page": page_idx + 1,
                            "variant": variant["id"], "status": "error",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                        (variant_dir / "meta.json").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
                        results.append(error)
            finally:
                raw.close()
        manifest.update({"status": "ok", "page_count": doc.page_count, "processed_pages": limit})
        (pdf_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return results
    finally:
        doc.close()


def main():
    ap = argparse.ArgumentParser(description="Тест всех image-preprocessing вариантов на первых 10 страницах БТИ-PDF.")
    ap.add_argument("--csv", required=True, help="CSV с колонкой «путь»")
    ap.add_argument("--path-column", default="путь")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--outdir", default="ocr_image_debug")
    ap.add_argument("--pages-per-pdf", type=int, default=10, help="По умолчанию первые 10 страниц каждого PDF")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--ocr-lang", default="rus")
    ap.add_argument("--ocr-psm", type=int, default=11)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tesseract-cmd", default=None)
    args = ap.parse_args()

    if len(args.sep) != 1:
        raise SystemExit("--sep должен быть одним символом")
    if args.pages_per_pdf < 1:
        raise SystemExit("--pages-per-pdf должен быть не меньше 1")
    if args.dpi < 200:
        raise SystemExit("--dpi слишком низкий для диагностического теста")
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

    all_results = []
    for pdf_path in tqdm(paths, desc="OCR image debug", unit="pdf"):
        all_results.extend(process_pdf(pdf_path, outdir, args))

    summary_path = outdir / "summary.csv"
    fields = [
        "source_pdf", "page", "variant", "status", "chars", "digits", "commas_dots",
        "tsv_words", "tsv_digit_words", "tsv_numeric_like_words", "seconds", "megapixels",
        "tessinput_file", "reason",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Готово: {summary_path}")
    print("Для каждой страницы и варианта сохранены render.png, prepared.png, text.txt, result.tsv, meta.json и (если Tesseract отдал) tessinput.tif.")


if __name__ == "__main__":
    main()
