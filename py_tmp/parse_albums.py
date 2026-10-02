#!/usr/bin/env python3
"""Сборка единой таблицы из PDF-альбомов оценки квартир.

Структура: <root>/<Адрес>/<Альбом>.pdf
Запуск:    python parse_albums.py <root> [out.csv]
Зависимости: pip install pdfplumber tqdm pandas
Выход: out.csv (по умолчанию albums.csv), errors.csv, parse_albums.log
"""
import re
import sys
import logging
from pathlib import Path

import pdfplumber
import pandas as pd
from tqdm import tqdm

COLS = ["address", "filename", "album_number", "album_date",
        "area_m2", "floor", "rooms", "price_m2", "price_total"]
MONTHS = {m: i + 1 for i, m in enumerate(
    "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split())}

logging.basicConfig(filename="parse_albums.log", filemode="w", level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
log = logging.getLogger()


# ---------- нормализация ----------
def low_nl(s):
    """нижний регистр, ё->е, без мягких переносов; переводы строк сохраняются"""
    return (s or "").lower().replace("ё", "е").replace("\u00ad", "").replace("\xa0", " ")

def low(s):
    """то же, но пробелы и переносы схлопнуты"""
    return re.sub(r"\s+", " ", low_nl(s)).strip()

def squash(s):
    """вообще без пробелов: устойчиво к переносам внутри слов"""
    return re.sub(r"\s+", "", low_nl(s))

def alnum(s):
    return re.sub(r"[^0-9a-zа-я]", "", low(s))

def addr_in(folder, text):
    """адрес папки входит в текст, и номер дома не продолжается цифрой (5 != 55)"""
    f = alnum(folder)
    return bool(f) and re.search(re.escape(f) + r"(?!\d)", alnum(text)) is not None

def num(s):
    s = re.sub(r"[^\d,.]", "", (s or "").replace("\xa0", ""))
    s = s.replace(",", ".")
    if s.count(".") > 1:
        s = s.replace(".", "", s.count(".") - 1)
    try:
        return float(s)
    except ValueError:
        return None


# ---------- титульный лист ----------
def parse_title(text):
    """-> (number, date, title_low, error)"""
    if "квартир" not in squash(text):
        return None, None, "", "на титульном листе нет слова 'квартиры'"
    t = low(text)

    number = None
    m = re.search(r"отчет\s*№\s*([^\n]+)", low_nl(text))
    if m:
        number = re.split(r"\s+(?:от|об|дата)\s", m.group(1).strip() + " ")[0].strip() or None

    date = None
    m = re.search(r"дата\s*составления\s*отчета[^0-9]{0,20}(.{0,60})", t)
    if m:
        d = m.group(1)
        m1 = re.search(r"(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{4})", d)
        m2 = re.search(r"(\d{1,2})\s*[»\"”]?\s*([а-я]+)\s*(\d{4})", d)
        if m1:
            date = f"{m1[3]}-{int(m1[2]):02d}-{int(m1[1]):02d}"
        elif m2 and m2[2] in MONTHS:
            date = f"{m2[3]}-{MONTHS[m2[2]]:02d}-{int(m2[1]):02d}"
    return number, date, t, None


# ---------- таблицы ----------
def classify(h):
    """h = заголовок колонки без пробелов"""
    if "стоимость" in h and "руб" in h:
        return "price_m2" if re.search(r"кв\.?м", h) else "price_total"
    if "общаяплощадь" in h: return "area_m2"
    if "комнат" in h:       return "rooms"
    if "этаж" in h:         return "floor"
    if "адрес" in h:        return "address"
    return None

def build_map(rows):
    """Ищет шапку (может быть многострочной) в первых 5 строках.
    -> ({индекс_колонки: поле}, индекс первой строки данных) или (None, 0)"""
    heads = {}
    for i, row in enumerate(rows[:5]):
        for j, c in enumerate(row):
            heads[j] = heads.get(j, "") + squash(c)
        cmap = {}
        for j, h in heads.items():
            k = classify(h)
            if k and k not in cmap.values():
                cmap[j] = k
        if {"area_m2", "rooms", "price_m2", "price_total"} <= set(cmap.values()):
            return cmap, i + 1
    return None, 0

def is_contents(text):
    """Страница с оглавлением: заголовок в первых строках"""
    head = squash("\n".join((text or "").splitlines()[:5]))
    return "содержание" in head or "оглавление" in head


# ---------- альбом ----------
def parse_album(pdf_path, folder):
    """-> (список строк, None) или (None, текст ошибки)"""
    rows_out = []
    with pdfplumber.open(pdf_path) as pdf:
        number, date, title, err = parse_title(pdf.pages[0].extract_text() or "")
        if err:
            return None, err
        if not number: log.warning("%s: не найден номер отчета", pdf_path)
        if not date:   log.warning("%s: не найдена дата отчета", pdf_path)

        title_has_addr = addr_in(folder, title)
        prev_map, prev_n = None, 0
        any_addr_col = found_data = False
        stopped_at = None

        for pno, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            # оглавление после содержательных таблиц -> материалы обоснования, выходим
            if pno > 0 and found_data and is_contents(text):
                stopped_at = pno + 1
                break

            # пока таблиц с оценками не было, страницы без «стоимость» пропускаем
            if prev_map is None and "стоимость" not in squash(text):
                page.flush_cache()
                continue

            for tbl in page.extract_tables():
                if not tbl:
                    continue
                cmap, start = build_map(tbl)
                if cmap is None and prev_map and len(tbl[0]) == prev_n:
                    cmap, start = prev_map, 0      # продолжение таблицы без шапки
                if cmap is None:
                    continue                       # нерелевантная таблица
                prev_map, prev_n = cmap, len(tbl[0])
                has_addr = "address" in cmap.values()
                any_addr_col |= has_addr
                found_data = True
                if not has_addr and not title_has_addr:
                    continue

                for row in tbl[start:]:
                    r = {k: row[j] for j, k in cmap.items() if j < len(row)}
                    if has_addr and not addr_in(folder, r.get("address", "")):
                        continue
                    area = num(r.get("area_m2"))
                    pm2, ptot = num(r.get("price_m2")), num(r.get("price_total"))
                    if area is None or (pm2 is None and ptot is None):
                        continue                   # не строка данных (нумерация, итоги и т.п.)
                    rows_out.append(dict(
                        address=folder, filename=pdf_path.name,
                        album_number=number, album_date=date,
                        area_m2=area,
                        floor=low(r.get("floor", "")) or None,
                        rooms=low(r.get("rooms", "")) or None,
                        price_m2=pm2, price_total=ptot))
            page.flush_cache()

        log.info("%s: остановка на стр. %s из %d", pdf_path.name,
                 stopped_at or "—", len(pdf.pages))

    if not any_addr_col and not title_has_addr:
        return None, "адрес не найден ни в колонке таблицы, ни на титульном листе"
    if not found_data:
        log.warning("%s: не найдено таблиц с оценками", pdf_path)
    elif not rows_out:
        log.warning("%s: 0 строк для адреса '%s'", pdf_path, folder)
    return rows_out, None


# ---------- main ----------
def main():
    if len(sys.argv) < 2:
        sys.exit("Использование: python parse_albums.py <root> [out.csv]")
    root = Path(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "albums.csv"

    pdfs = [(d.name, p) for d in sorted(root.iterdir()) if d.is_dir()
            for p in sorted(d.glob("*.pdf"))]
    data, errors = [], []

    for folder, p in tqdm(pdfs, desc="Альбомы", unit="pdf"):
        try:
            rows, err = parse_album(p, folder)
        except Exception as e:
            rows, err = None, f"exception: {e!r}"
            log.exception("%s", p)
        if err:
            log.error("%s | %s", p, err)
            errors.append(dict(address=folder, filename=p.name, error=err))
        else:
            log.info("%s: %d строк", p, len(rows))
            data += rows

    pd.DataFrame(data, columns=COLS).to_csv(out, index=False, encoding="utf-8-sig")
    if errors:
        pd.DataFrame(errors).to_csv("errors.csv", index=False, encoding="utf-8-sig")
    print(f"Строк: {len(data)}, альбомов: {len(pdfs)}, "
          f"пропущено/ошибок: {len(errors)} (см. errors.csv, parse_albums.log)")


if __name__ == "__main__":
    main()
