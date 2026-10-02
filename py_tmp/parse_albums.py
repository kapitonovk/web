#!/usr/bin/env python3
"""Сборка единой таблицы из PDF-альбомов оценки квартир.
Использование: python parse_albums.py <корневая_папка> [out.csv]
"""
import re, sys, logging
from pathlib import Path
import pdfplumber, pandas as pd
from tqdm import tqdm

MONTHS = {m: i + 1 for i, m in enumerate(
    "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split())}
COLS = ["address", "filename", "album_number", "album_date",
        "area_m2", "floor", "rooms", "price_m2", "price_total"]

logging.basicConfig(filename="parse_albums.log", filemode="w", level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
log = logging.getLogger()


def low(s):            # нижний регистр, ё->е, схлопнуть пробелы (в т.ч. переносы)
    s = (s or "").lower().replace("ё", "е").replace("\u00ad", "")
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()

def squash(s):         # то же, но вообще без пробелов (устойчиво к переносам внутри слов)
    return re.sub(r"\s+", "", low(s))

def alnum(s):          # для сравнения адресов
    return re.sub(r"[^0-9a-zа-я]", "", low(s))

def addr_in(folder, text):   # адрес папки входит в текст, и дом не продолжается цифрой (5 != 55)
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


def parse_title(text):
    t = low(text)
    ts = squash(text)
    if "квартир" not in ts:
        return None, None, t, "no 'квартиры' on title"
    m = re.search(r"отчет\s*№\s*([^\n]*?)(?:\s+(?:от|дата|об\s+оценке)\b|$)", low_keep_nl(text))
    number = m.group(1).strip() if m and m.group(1).strip() else None
    if not number:
        m = re.search(r"отчет\s*№\s*(\S+)", t)
        number = m.group(1) if m else None
    date = None
    m = re.search(r"дата\s*составления\s*отчета[^0-9«\"]{0,20}(.{0,60})", t)
    if m:
        d = m.group(1)
        m1 = re.search(r"(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{4})", d)
        m2 = re.search(r"(\d{1,2})\s*[»\"”]?\s*([а-я]+)\s*(\d{4})", d)
        if m1:
            date = f"{m1[3]}-{int(m1[2]):02d}-{int(m1[1]):02d}"
        elif m2 and m2[2] in MONTHS:
            date = f"{m2[3]}-{MONTHS[m2[2]]:02d}-{int(m2[1]):02d}"
    return number, date, t, None

def low_keep_nl(s):    # lower + ё->е, но с сохранением переводов строк
    return (s or "").lower().replace("ё", "е").replace("\xa0", " ")


def classify(h):       # h = заголовок колонки без пробелов
    if "стоимость" in h and "руб" in h:
        return "price_m2" if re.search(r"кв\.?м", h) else "price_total"
    if "общаяплощадь" in h: return "area_m2"
    if "комнат" in h:       return "rooms"
    if "этаж" in h:         return "floor"
    if "адрес" in h:        return "address"
    return None

def build_map(rows):
    """Ищем шапку в первых строках (может быть многострочной). -> (map, индекс_первой_строки_данных)"""
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


def parse_album(pdf_path, folder):
    rows_out = []
    with pdfplumber.open(pdf_path) as pdf:
        number, date, title, err = parse_title(pdf.pages[0].extract_text() or "")
        if err:
            return None, err
        if not number: log.warning("%s: не найден номер отчета", pdf_path)
        if not date:   log.warning("%s: не найдена дата отчета", pdf_path)
        title_has_addr = addr_in(folder, title)
        prev_map, prev_n = None, 0
        any_addr_col = False
        for pno, page in enumerate(pdf.pages):
            for tbl in page.extract_tables():
                if not tbl:
                    continue
                cmap, start = build_map(tbl)
                if cmap is None and prev_map and len(tbl[0]) == prev_n:
                    cmap, start = prev_map, 0          # продолжение таблицы без шапки
                if cmap is None:
                    continue
                prev_map, prev_n = cmap, len(tbl[0])
                has_addr = "address" in cmap.values()
                any_addr_col |= has_addr
                if not has_addr and not title_has_addr:
                    continue
                for row in tbl[start:]:
                    r = {k: row[j] for j, k in cmap.items() if j < len(row)}
                    if has_addr and not addr_in(folder, r.get("address", "")):
                        continue
                    area, pm2, ptot = num(r.get("area_m2")), num(r.get("price_m2")), num(r.get("price_total"))
                    if area is None or (pm2 is None and ptot is None):
                        continue                       # не строка данных
                    rows_out.append(dict(
                        address=folder, filename=pdf_path.name, album_number=number, album_date=date,
                        area_m2=area, floor=low(r.get("floor", "")) or None,
                        rooms=low(r.get("rooms", "")) or None, price_m2=pm2, price_total=ptot))
        if not any_addr_col and not title_has_addr:
            return None, "адрес не найден ни в колонке таблицы, ни на титульном листе"
    if not rows_out:
        log.warning("%s: 0 строк для адреса '%s'", pdf_path, folder)
    return rows_out, None


def main():
    root = Path(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "albums.csv"
    pdfs = [(d.name, p) for d in sorted(root.iterdir()) if d.is_dir() for p in sorted(d.glob("*.pdf"))]
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
    print(f"Строк: {len(data)}, альбомов: {len(pdfs)}, пропущено/ошибок: {len(errors)} (см. errors.csv, parse_albums.log)")

if __name__ == "__main__":
    main()