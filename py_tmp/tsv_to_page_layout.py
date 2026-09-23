#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Создаёт моноширинную текстовую имитацию страницы по Tesseract TSV.

В отличие от обычного reading_order, этот скрипт сохраняет горизонтальное
положение распознанных слов: слово ставится в текстовую «сетку» по left/top.
Результат нужен для визуальной оценки: похоже ли TSV на исходный скан,
где были колонки, номера и площади.

Для каждого result.tsv создаёт рядом:
  page_layout.txt

Пример:
  py tsv_to_page_layout.py --root ".\ocr_image_debug"

Более широкая сетка (лучше различает колонки, но строки длиннее):
  py tsv_to_page_layout.py --root ".\ocr_image_debug" --cols 180

Пересобрать имеющиеся файлы:
  py tsv_to_page_layout.py --root ".\ocr_image_debug" --overwrite
"""

import argparse
import csv
import math
from pathlib import Path


def parse_words(tsv_path):
    with tsv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        words = []
        max_right = max_bottom = 0
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
            right = left + width
            bottom = top + height
            words.append({"text": text, "left": left, "top": top, "right": right, "bottom": bottom})
            max_right = max(max_right, right)
            max_bottom = max(max_bottom, bottom)
    return words, max_right, max_bottom


def fit_text(text, max_width):
    # В txt один символ не равен пикселю: длинные слова не должны затирать соседей.
    if max_width <= 0:
        return ""
    if len(text) <= max_width:
        return text
    if max_width == 1:
        return text[:1]
    return text[: max_width - 1] + "…"


def render_layout(words, page_width_px, page_height_px, columns, vertical_scale):
    if not words:
        return "[В TSV нет распознанных слов]\n"

    columns = max(40, columns)
    char_px = max(1.0, page_width_px / columns)
    # Уменьшаем число текстовых строк: иначе 400 DPI A4 даст тысячи пустых строк.
    row_px = max(1.0, char_px * vertical_scale)
    rows = max(1, math.ceil(page_height_px / row_px) + 1)
    canvas = [[] for _ in range(rows)]

    # Сначала верхние слова, при равной высоте — левые.
    for word in sorted(words, key=lambda w: (w["top"], w["left"])):
        row = min(rows - 1, max(0, int(word["top"] / row_px)))
        col = min(columns - 1, max(0, int(word["left"] / char_px)))
        width_chars = max(1, int((word["right"] - word["left"]) / char_px))
        text = fit_text(word["text"], max(width_chars, len(word["text"])))
        canvas[row].append((col, text))

    rendered = []
    last_nonempty = -1
    for idx, items in enumerate(canvas):
        if not items:
            continue
        line = []
        cursor = 0
        # Если несколько слов попали в одну текстовую строку, они сохраняются
        # по X-координате. При коллизии второе слово ставится после первого.
        for col, text in sorted(items, key=lambda x: x[0]):
            col = max(col, cursor + (1 if line else 0))
            if col > cursor:
                line.append(" " * (col - cursor))
                cursor = col
            line.append(text)
            cursor += len(text)
        rendered.append((idx, "".join(line).rstrip()))
        last_nonempty = idx

    if not rendered:
        return "[В TSV нет распознанных слов]\n"

    output = []
    previous_row = rendered[0][0]
    for row_idx, line in rendered:
        # Не выводим сотни пустых строк: максимум 3 между визуальными блоками.
        gap = min(3, max(0, row_idx - previous_row))
        output.extend([""] * gap)
        output.append(line)
        previous_row = row_idx
    return "\n".join(output).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description="Строит моноширинную пространственную имитацию страницы из Tesseract TSV.")
    ap.add_argument("--root", required=True, help="Корневая папка с result.tsv")
    ap.add_argument("--cols", type=int, default=140, help="Ширина текстовой сетки; по умолчанию 140")
    ap.add_argument("--vertical-scale", type=float, default=1.8, help="Вертикальное сжатие; больше = компактнее")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Нет папки: {root}")
    if args.cols < 40:
        raise SystemExit("--cols должен быть не меньше 40")
    if args.vertical_scale <= 0:
        raise SystemExit("--vertical-scale должен быть больше 0")

    tsv_paths = sorted(root.rglob("result.tsv"))
    if not tsv_paths:
        raise SystemExit("result.tsv не найдены")

    made = skipped = empty = 0
    for tsv_path in tsv_paths:
        out_path = tsv_path.with_name("page_layout.txt")
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        words, width, height = parse_words(tsv_path)
        text = render_layout(words, width, height, args.cols, args.vertical_scale)
        out_path.write_text(text, encoding="utf-8")
        made += 1
        if not words:
            empty += 1

    print(f"TSV: {len(tsv_paths)} | создано: {made} | уже было: {skipped} | пустых: {empty}")


if __name__ == "__main__":
    main()
