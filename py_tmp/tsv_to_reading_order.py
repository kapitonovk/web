#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Собирает читаемый текст из Tesseract TSV по координатам слов.

Берёт TSV-файлы рекурсивно из папки (например, результат варианта
06_tess_otsu в bti_ocr_image_debug) и рядом с каждым result.tsv пишет:
  reading_order.txt

Слова группируются в визуальные строки по Y-координате, затем внутри строки
сортируются слева направо. Пустая строка добавляется при большом вертикальном
разрыве. Низкий confidence не отбрасывается: для БТИ важнее ничего не потерять.

Пример:
  py tsv_to_reading_order.py --root ".\ocr_image_debug"

Пересобрать уже существующие reading_order.txt:
  py tsv_to_reading_order.py --root ".\ocr_image_debug" --overwrite
"""

import argparse
import csv
import statistics
from pathlib import Path


def parse_words(tsv_path):
    with tsv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
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
                "cy": top + height / 2,
                "height": height,
            })
    return words


def make_reading_order(words):
    if not words:
        return ""
    heights = [w["height"] for w in words if w["height"] > 0]
    median_height = statistics.median(heights) if heights else 20
    y_tolerance = max(8.0, median_height * 0.65)

    words.sort(key=lambda w: (w["cy"], w["left"]))
    lines = []
    for word in words:
        candidates = [
            (abs(word["cy"] - line["cy"]), line)
            for line in lines
            if abs(word["cy"] - line["cy"]) <= y_tolerance
        ]
        if candidates:
            _, line = min(candidates, key=lambda x: x[0])
            line["words"].append(word)
            line["cy"] = sum(w["cy"] for w in line["words"]) / len(line["words"])
            line["top"] = min(line["top"], word["top"])
            line["bottom"] = max(line["bottom"], word["bottom"])
        else:
            lines.append({
                "cy": word["cy"],
                "top": word["top"],
                "bottom": word["bottom"],
                "words": [word],
            })

    lines.sort(key=lambda line: (line["top"], min(w["left"] for w in line["words"])))
    output = []
    previous_bottom = None
    paragraph_gap = max(28, median_height * 1.8)
    for line in lines:
        line["words"].sort(key=lambda w: w["left"])
        if previous_bottom is not None and line["top"] - previous_bottom > paragraph_gap:
            output.append("")
        output.append(" ".join(word["text"] for word in line["words"]))
        previous_bottom = max(previous_bottom or 0, line["bottom"])
    return "\n".join(output).strip() + "\n"


def main():
    ap = argparse.ArgumentParser(description="Восстанавливает читаемый порядок строк из Tesseract TSV.")
    ap.add_argument("--root", required=True, help="Корневая папка с result.tsv")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Нет папки: {root}")

    tsv_paths = sorted(root.rglob("result.tsv"))
    if not tsv_paths:
        raise SystemExit("result.tsv не найдены")

    made = skipped = empty = 0
    for tsv_path in tsv_paths:
        out = tsv_path.with_name("reading_order.txt")
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        text = make_reading_order(parse_words(tsv_path))
        out.write_text(text, encoding="utf-8")
        made += 1
        if not text.strip():
            empty += 1

    print(f"TSV: {len(tsv_paths)} | создано: {made} | уже было: {skipped} | пустых: {empty}")


if __name__ == "__main__":
    main()
