#!/usr/bin/env python3
"""Сборка единой таблицы из PDF-альбомов оценки квартир.

Структура: <root>/<Адрес>/<Альбом>.pdf
Запуск:    python parse_albums.py <root> [out.csv]
Зависимости: pip install pymupdf pdfplumber tqdm pandas
Выход: out.csv (дозапись после каждого альбома), errors.csv, done.txt (для продолжения), parse_albums.log
"""
import re
import sys
import time
import logging
from pathlib import Path

import fitz  # PyMuPDF: быстрый текст
import pdfplumber  # таблицы, только на нужных страницах
import pandas as pd
from tqdm import tqdm

COLS = ["address", "filename", "album_number", "album_date",
        "area_m2", "floor", "rooms", "price_m2", "price_total"]
MONTHS = {m: i + 1 for i, m in enumerate(
    "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split())}
MAX_MISS = 2  # сколько страниц подряд без таблиц терпим внутри серии

logging.basicConfig(filename="parse_albums.log", filemode="a", level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
log = logging.getLogger()


# ---------- нормализация ----------
def low_nl(s):
    return (s or "").lower().replace("ё", "е").replace("\u00ad", "").replace("\xa0", " ")

def low(s):
    return re.sub(r"\s+", " ", low_nl(s)).strip()

def squash(s):
    return re.sub(r"\s+", "", low_nl(s))

def alnum(s):
    return re.sub(r"[^0-9a-zа-я]", "", low(s))

def addr_in(folder, text):
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
    if "стоимость" in h and "руб" in h:
        return "price_m2" if re.search(r"кв\.?м", h) else "price_total"
    if "общаяплощадь" in h: return "area_m2"
    if "комнат" in h:       return "rooms"
    if "этаж" in h:         return "floor"
    if "адрес" in h:        return "address"
    return None

def build_map(rows):
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
    head = squash("\n".join((text or "").splitlines()[:5]))
    return "содержание" in head or "оглавление" in head

def has_header(sq):
    return "стоимост" in sq and "площад" in sq and "комнат" in sq


# ---------- альбом ----------
def parse_album(pdf_path, folder):
    """-> (список строк, None) или (None, текст ошибки)"""
    rows_out = []
    doc = fitz.open(pdf_path)
    try:
        number, date, title, err = parse_title(doc[0].get_text())
        if err:
            return None, err
        if not number: log.warning("%s: не найден номер отчета", pdf_path)
        if not date:   log.warning("%s: не найдена дата отчета", pdf_path)

        title_has_addr = addr_in(folder, title)
        prev_map, prev_n = None, 0
        any_addr_col = found_data = active = False
        miss, stopped_at, plumbed = 0, None, 0

        with pdfplumber.open(pdf_path) as pdf:
            for pno in range(len(doc)):
                text = doc[pno].get_text()

                if pno > 0 and found_data and is_contents(text):
                    stopped_at = pno + 1
                    break

                if not (active or has_header(squash(text))):
                    continue                      # страница не про таблицу оценок

                page = pdf.pages[pno]
                plumbed += 1
                got = False
                for tbl in page.extract_tables():
                    if not tbl:
                        continue
                    cmap, start = build_map(tbl)
                    if cmap is None and prev_map and len(tbl[0]) == prev_n:
                        cmap, start = prev_map, 0
                    if cmap is None:
                        continue
                    prev_map, prev_n = cmap, len(tbl[0])
                    has_addr = "address" in cmap.values()
                    any_addr_col |= has_addr
                    got = found_data = True
                    if not has_addr and not title_has_addr:
                        continue
                    for row in tbl[start:]:
                        r = {k: row[j] for j, k in cmap.items() if j < len(row)}
                        if has_addr and not addr_in(folder, r.get("address", "")):
                            continue
                        area = num(r.get("area_m2"))
                        pm2, ptot = num(r.get("price_m2")), num(r.get("price_total"))
                        if area is None or (pm2 is None and ptot is None):
                            continue
                        rows_out.append(dict(
                            address=folder, filename=pdf_path.name,
                            album_number=number, album_date=date,
                            area_m2=area,
                            floor=low(r.get("floor", "")) or None,
                            rooms=low(r.get("rooms", "")) or None,
                            price_m2=pm2, price_total=ptot))
                page.flush_cache()

                if got:
                    active, miss = True, 0
                else:
                    miss += 1
                    if miss > MAX_MISS:
                        active = False

        log.info("%s: стр. всего %d, через pdfplumber %d, остановка на %s",
                 pdf_path.name, len(doc), plumbed, stopped_at or "—")
    finally:
        doc.close()

    if not any_addr_col and not title_has_addr:
        return None, "адрес не найден ни в колонке таблицы, ни на титульном листе"
    if not found_data:
        log.warning("%s: не найдено таблиц с оценками", pdf_path)
    elif not rows_out:
        log.warning("%s: 0 строк для адреса '%s'", pdf_path, folder)
    return rows_out, None


# ---------- запись ----------
def append_csv(path, records, cols=None):
    if not records:
        return
    df = pd.DataFrame(records, columns=cols)
    df.to_csv(path, mode="a", header=not Path(path).exists(), index=False, encoding="utf-8-sig")


def main():
    if len(sys.argv) < 2:
        sys.exit("Использование: python parse_albums.py <root> [out.csv]")
    root = Path(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "albums.csv"
    done_f = Path("done.txt")
    done = set(done_f.read_text(encoding="utf-8").splitlines()) if done_f.exists() else set()

    pdfs = [(d.name, p) for d in sorted(root.iterdir()) if d.is_dir()
            for p in sorted(d.glob("*.pdf"))]
    todo = [(f, p) for f, p in pdfs if f"{f}/{p.name}" not in done]
    print(f"Всего альбомов: {len(pdfs)}, уже готово: {len(pdfs) - len(todo)}")

    n_rows = n_err = 0
    for folder, p in tqdm(todo, desc="Альбомы", unit="pdf"):
        t0 = time.time()
        try:
            rows, err = parse_album(p, folder)
        except Exception as e:
            rows, err = None, f"exception: {e!r}"
            log.exception("%s", p)
        if err:
            log.error("%s | %s", p, err)
            append_csv("errors.csv", [dict(address=folder, filename=p.name, error=err)])
            n_err += 1
        else:
            append_csv(out, rows, COLS)
            n_rows += len(rows)
            log.info("%s: %d строк за %.1f с", p.name, len(rows), time.time() - t0)
        with done_f.open("a", encoding="utf-8") as f:   # помечаем только после записи
            f.write(f"{folder}/{p.name}\n")

    print(f"Новых строк: {n_rows}, ошибок/пропусков: {n_err} (см. errors.csv, parse_albums.log)")


if __name__ == "__main__":
    main()
