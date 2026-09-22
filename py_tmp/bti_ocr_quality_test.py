#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестирует OCR-настройки на первых N страницах каждого БТИ-PDF и сохраняет
все bitmap-варианты, TXT и TSV для визуального сравнения.

Для каждой страницы варианты ЧЕРЕДУЮТСЯ циклически:
  A: rus, PSM 11, 350 DPI, мягкий contrast
  B: rus, PSM 11, 400 DPI, мягкий contrast
  C: rus, PSM 4,  400 DPI, мягкий contrast
  D: rus, PSM 11, 400 DPI, adaptive threshold (Sauvola-подобный)

Например при --pages-per-pdf 20 каждый вариант получит примерно 5 страниц
каждого PDF. Это именно диагностический тест: он не меняет основной OCR-кеш
и не использует папки dump-root.

Вход: CSV с колонкой «путь».

Результат:
  <outdir>/<safe_pdf_name>__<hash>/
    manifest.json
    summary.csv
    pages/
      page_0001_A_p11_350/
        render.png       исходный grayscale render
        prepared.png     bitmap, фактически переданный Tesseract
        text.txt         plain text
        result.tsv       слова + координаты + confidence
        meta.json        метрики страницы и настройки

Зависимости:
  py -m pip install pymupdf pytesseract pillow tqdm numpy

Tesseract рядом со скриптом:
  .\tesseract\tesseract.exe
  .\tesseract\tessdata\rus.traineddata

Пример:
  py bti_ocr_quality_test.py --csv test_two_pdfs.csv --sep ";" --outdir ocr_quality_test
"""

import argparse
import csv
import hashlib
import json
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import fitz
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items, **kwargs):
        return items


VARIANTS = [
    {
        "id": "A",
        "dpi": 350,
        "psm": 11,
        "prep": "soft",
        "label": "rus | PSM 11 | 350 DPI | soft contrast",
    },
    {
        "id": "B",
        "dpi": 400,
        "psm": 11,
        "prep": "soft",
        "label": "rus | PSM 11 | 400 DPI | soft contrast",
    },
    {
        "id": "C",
        "dpi": 400,
        "psm": 4,
        "prep": "soft",
        "label": "rus | PSM 4 | 400 DPI | soft contrast",
    },
    {
        "id": "D",
        "dpi": 400,
        "psm": 11,
        "prep": "adaptive",
        "label": "rus | PSM 11 | 400 DPI | adaptive threshold",
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
        available = set(pytesseract.get_languages(config=""))
    except Exception as exc:
        raise SystemExit(f"Не удалось запустить Tesseract: {exc}")
    if lang not in available:
        raise SystemExit(f"В Tesseract нет языка: {lang}")


def render_page(page, dpi):
    scale = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False)
    try:
        return Image.frombytes("L", [pix.width, pix.height], pix.samples)
    finally:
        pix = None


def prepare_soft(image):
    image = ImageOps.autocontrast(image, cutoff=0.5)
    return ImageEnhance.Contrast(image).enhance(1.35)


def adaptive_threshold(image, window=51, bias=12):
    """
    Локальный порог в духе Sauvola/Niblack без OpenCV.
    Используется ТОЛЬКО как тестовый вариант D: может помочь на сером фоне,
    но может повредить тонкие десятичные запятые, поэтому bitmap сохраняется.
    """
    img = image.filter(ImageFilter.MedianFilter(size=3))
    array = np.asarray(img, dtype=np.float32)
    pad = window // 2
    padded = np.pad(array, ((pad, pad), (pad, pad)), mode="reflect")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    h, w = array.shape
    y0 = np.arange(h)
    y1 = y0 + window
    x0 = np.arange(w)
    x1 = x0 + window
    sums = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    local_mean = sums / (window * window)
    binary = np.where(array < (local_mean - bias), 0, 255).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def prepare_image(image, method):
    if method == "soft":
        return prepare_soft(image)
    if method == "adaptive":
        base = prepare_soft(image)
        try:
            return adaptive_threshold(base)
        finally:
            base.close()
    raise ValueError(f"Unknown prep: {method}")


def count_tsv_words(tsv):
    rows = list(csv.DictReader(tsv.splitlines(), delimiter="\t"))
    words = [r for r in rows if r.get("level") == "5" and (r.get("text") or "").strip()]
    digit_words = [r for r in words if re.search(r"\d", r.get("text") or "")]
    numeric_like = [r for r in words if re.fullmatch(r"[0-9OОЗЗбБIlІl.,:-]+", (r.get("text") or "").strip())]
    return len(words), len(digit_words), len(numeric_like)


def text_metrics(text, tsv):
    words, digit_words, numeric_like = count_tsv_words(tsv)
    return {
        "chars": len(text),
        "digits": len(re.findall(r"\d", text)),
        "commas_dots": len(re.findall(r"[,.]", text)),
        "tsv_words": words,
        "tsv_digit_words": digit_words,
        "tsv_numeric_like_words": numeric_like,
    }


def run_variant(page, variant, args, page_dir):
    raw = render_page(page, variant["dpi"])
    try:
        raw.save(page_dir / "render.png")
        prepared = prepare_image(raw, variant["prep"])
        try:
            prepared.save(page_dir / "prepared.png")
            config = f"--oem 1 --psm {variant['psm']} -c preserve_interword_spaces=1"
            t0 = time.time()
            text = pytesseract.image_to_string(prepared, lang=args.ocr_lang, config=config, timeout=args.timeout)
            tsv = pytesseract.image_to_data(prepared, lang=args.ocr_lang, config=config, timeout=args.timeout, output_type=pytesseract.Output.STRING)
            seconds = time.time() - t0
            text = norm_text(text)
            (page_dir / "text.txt").write_text(text, encoding="utf-8")
            (page_dir / "result.tsv").write_text(tsv, encoding="utf-8")
            metrics = text_metrics(text, tsv)
            meta = {
                "variant": variant,
                "page": page.number + 1,
                "page_size_pt": {"width": page.rect.width, "height": page.rect.height},
                "render_size_px": {"width": prepared.width, "height": prepared.height},
                "megapixels": round((prepared.width * prepared.height) / 1_000_000, 2),
                "seconds": round(seconds, 2),
                **metrics,
            }
            (page_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"status": "ok", **meta}
        finally:
            prepared.close()
    finally:
        raw.close()


def process_pdf(pdf_path, outdir, args):
    pdf_out = outdir / f"{safe_name(pdf_path.stem)}__{short_hash(pdf_path)}"
    pages_out = pdf_out / "pages"
    pages_out.mkdir(parents=True, exist_ok=True)
    results = []
    manifest = {
        "source_pdf": str(pdf_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "variants": VARIANTS,
        "pages_per_pdf": args.pages_per_pdf,
        "status": "error",
    }
    if not pdf_path.is_file():
        manifest["error"] = "PDF not found"
        (pdf_out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return [{"source_pdf": str(pdf_path), "status": "missing"}]
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        manifest["error"] = f"open PDF: {type(exc).__name__}: {exc}"
        (pdf_out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return [{"source_pdf": str(pdf_path), "status": "error", "reason": manifest["error"]}]
    try:
        limit = min(args.pages_per_pdf, doc.page_count)
        for page_idx in range(limit):
            page = doc.load_page(page_idx)
            variant = VARIANTS[page_idx % len(VARIANTS)]
            dirname = f"page_{page_idx+1:04d}_{variant['id']}_psm{variant['psm']}_{variant['dpi']}"
            page_dir = pages_out / dirname
            if page_dir.exists() and not args.overwrite:
                meta_path = page_dir / "meta.json"
                if meta_path.is_file():
                    old = json.loads(meta_path.read_text(encoding="utf-8"))
                    results.append({"source_pdf": str(pdf_path), "page": page_idx + 1, "variant": variant["id"], "status": "skipped_existing", **old})
                    continue
            page_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = run_variant(page, variant, args, page_dir)
                results.append({"source_pdf": str(pdf_path), "page": page_idx + 1, "variant": variant["id"], **result})
            except Exception as exc:
                error = {"source_pdf": str(pdf_path), "page": page_idx + 1, "variant": variant["id"], "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
                (page_dir / "meta.json").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
                results.append(error)
        manifest.update({"status": "ok", "page_count": doc.page_count, "processed_pages": limit})
        (pdf_out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return results
    finally:
        doc.close()


def main():
    ap = argparse.ArgumentParser(description="Сравнительный OCR-тест: 4 варианта, чередующиеся по страницам, с bitmap-дампом.")
    ap.add_argument("--csv", required=True, help="CSV с колонкой «путь»")
    ap.add_argument("--path-column", default="путь")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--outdir", default="ocr_quality_test")
    ap.add_argument("--pages-per-pdf", type=int, default=20, help="Обработать только первые N страниц каждого PDF")
    ap.add_argument("--ocr-lang", default="rus")
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
    validate_tesseract(tesseract, args.ocr_lang)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for pdf_path in tqdm(paths, desc="OCR test PDFs", unit="pdf"):
        all_results.extend(process_pdf(pdf_path, outdir, args))

    summary_path = outdir / "summary.csv"
    fields = [
        "source_pdf", "page", "variant", "status", "chars", "digits", "commas_dots",
        "tsv_words", "tsv_digit_words", "tsv_numeric_like_words", "seconds", "megapixels",
        "reason",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Готово: {summary_path}")
    print("В каждой папке pages/ лежат render.png, prepared.png, text.txt, result.tsv и meta.json.")


if __name__ == "__main__":
    main()
