#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_pdf_target_list.py

Строит список "целевых" PDF (выписки ЕГРН по квартирам) из сложного дерева
папок, без физического копирования файлов. Результат — CSV со списком путей,
которые дальше можно передать в egrn_*_extract.py (например, через симлинки
или временную сборку дерева root/*/*.pdf).

Логика отбора:
  1. Файл обязательно .pdf
  2. Путь не должен проходить через папки-исключения:
     "raw" (историческая копия — не трогаем),
     "БТИ", "оценка" (там не выписки ЕГРН, а совсем другие документы)
  3. В имени файла не должно быть маркеров машиноместа/нежилого/помещения:
     "м-м", "м_м", "машиноместо", "помещ", "нежил", "кладов"
  4. В имени файла не должно быть маркеров "не выписка" (другой тип документа):
     "отчет", "соглашение", "рв", "печать", "акт", "план", "паспорт"
  5. В имени файла должен быть хотя бы один "квартирный" маркер:
     "кв" или адресный паттерн (ул./пр./дом/номер дома с запятой).
     ВНИМАНИЕ: "ЕГРН" в имени файла НЕ считается надёжным маркером —
     "ЕГРН" может быть частью названия и у м/м, и у нежилых помещений
     (например "Выписка ЕГРН м-м 12.pdf"). Наличие "ЕГРН" в имени
     используется только как дополнительный (не решающий) сигнал в review.
  6. Файл не должен совпадать (по нормализованному имени) со списком
     нежилых помещений (--nonres-list), который уже есть по факту предыдущей
     обработки bkfn.

Всё, что не подошло ни под явное включение, ни под явное исключение —
уходит в отдельный "review" CSV для ручной проверки.

Запуск:
  python3 build_pdf_target_list.py --root "N:\Sale\!General\Адреса" \
      --nonres-list nonres_files.txt \
      --out target_pdfs.csv --review-out review_pdfs.csv
"""

import argparse
import csv
import re
import unicodedata
from pathlib import Path

# ── Маркеры исключения: объект не квартира (машиноместо/нежилое) ───────────
OBJECT_NEG_RE = re.compile(
    r"м[-_\s]?м\b|машино[-\s]?мест|помещ|кладов|нежил",
    re.IGNORECASE,
)

# ── Маркеры исключения: документ не является выпиской ЕГРН ─────────────────
# "рв" оставлен как отдельная подстрока по требованию; т.к. это короткая
# подстрока, применяется с границами слова, чтобы не резать случайные
# совпадения внутри других слов.
DOC_TYPE_NEG_RE = re.compile(
    r"отчет|отчёт|соглашени|\bрв\b|печат|\bакт\b|план|паспорт",
    re.IGNORECASE,
)

# ── Маркеры включения (признак квартиры/адреса) ─────────────────────────────
KV_RE = re.compile(r"кв\b|квартир", re.IGNORECASE)
ADDR_RE = re.compile(
    r"\bул\.?|\bпр(-?кт|\.)?|\bб[-\s]?р\b|\bдом\b|\bд\.\s?\d|,\s?\d+[а-яa-z]?\b",
    re.IGNORECASE,
)
# "ЕГРН" — не решающий маркер, только вспомогательный сигнал (см. docstring)
EGRN_RE = re.compile(r"егрн", re.IGNORECASE)

# ── Папки, которые целиком исключаем из обхода ──────────────────────────────
EXCLUDED_DIR_NAMES_RE = re.compile(r"^raw$|бти|оценк", re.IGNORECASE)


def normalize_name(name: str) -> str:
    """Нормализация имени файла для сравнения со списком нежилых."""
    name = unicodedata.normalize("NFKC", name).casefold().strip()
    name = re.sub(r"__\d+(?=\.pdf$)", "", name)  # снять суффиксы дублей __2, __3
    name = re.sub(r"\s+", " ", name)
    return name


def load_nonres_set(path: Path | None) -> set[str]:
    if not path or not path.is_file():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        names.add(normalize_name(Path(line).name))
    return names


def excluded_dir_in_path(path: Path, root: Path) -> str | None:
    """Возвращает имя папки-исключения, если путь проходит через неё."""
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    for part in rel_parts[:-1]:  # без учета самого файла
        if EXCLUDED_DIR_NAMES_RE.search(part):
            return part
    return None


def classify(pdf_path: Path, root: Path, nonres_set: set[str]) -> tuple[str, str]:
    """Возвращает (decision, reason). decision из {'include','exclude','review'}."""
    name = pdf_path.name

    bad_dir = excluded_dir_in_path(pdf_path, root)
    if bad_dir:
        return "exclude", f"путь проходит через исключённую папку: «{bad_dir}»"

    if OBJECT_NEG_RE.search(name):
        return "exclude", "маркер машиноместа/нежилого/помещения в имени"

    if DOC_TYPE_NEG_RE.search(name):
        return "exclude", "маркер документа-не-выписки (отчет/соглашение/рв/печать/акт/план/паспорт)"

    if normalize_name(name) in nonres_set:
        return "exclude", "совпадает со списком нежилых помещений"

    has_kv = bool(KV_RE.search(name))
    has_addr = bool(ADDR_RE.search(name))
    has_egrn = bool(EGRN_RE.search(name))  # вспомогательный, не решающий

    if has_kv or has_addr:
        signals = []
        if has_kv:
            signals.append("кв")
        if has_addr:
            signals.append("адрес-паттерн")
        if has_egrn:
            signals.append("егрн (вспомогательно)")
        return "include", "маркеры: " + ", ".join(signals)

    if has_egrn:
        # "ЕГРН" сам по себе ненадёжен (встречается и у м-м/нежилых),
        # поэтому без кв/адреса уходит в review, а не в include
        return "review", "только «ЕГРН» в имени без кв/адреса — ЕГРН не решающий маркер, нужна проверка"

    return "review", "нет явных маркеров кв/адреса/ЕГРН — нужна ручная проверка"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="корень с деревом папок-адресов")
    ap.add_argument("--nonres-list", default=None,
                     help="txt со списком файлов нежилых помещений (по одному на строку)")
    ap.add_argument("--out", default="target_pdfs.csv")
    ap.add_argument("--review-out", default="review_pdfs.csv")
    ap.add_argument("--excluded-out", default="excluded_pdfs.csv")
    args = ap.parse_args()

    root = Path(args.root)
    nonres_set = load_nonres_set(Path(args.nonres_list) if args.nonres_list else None)

    include_rows, review_rows, exclude_rows = [], [], []

    for pdf_path in root.rglob("*.pdf"):
        if not pdf_path.is_file():
            continue
        decision, reason = classify(pdf_path, root, nonres_set)
        row = [str(pdf_path), pdf_path.name, decision, reason]
        if decision == "include":
            include_rows.append(row)
        elif decision == "review":
            review_rows.append(row)
        else:
            exclude_rows.append(row)

    header = ["путь", "имя_файла", "решение", "причина"]

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(include_rows)

    with open(args.review_out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(review_rows)

    with open(args.excluded_out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(exclude_rows)

    print(f"include (целевые): {len(include_rows)} -> {args.out}")
    print(f"review (на ручную проверку): {len(review_rows)} -> {args.review_out}")
    print(f"excluded (отброшены): {len(exclude_rows)} -> {args.excluded_out}")


if __name__ == "__main__":
    main()
