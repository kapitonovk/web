#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
egrn_kv_extract_cached.py

Читает список PDF из CSV, отбирает выписки ЕГРН по квартирам и работает
с возобновляемым CSV-кэшем. Скрипт можно прервать Ctrl+C: каждый завершённый
PDF уже записан в один из cache-файлов и не будет обработан повторно.

Квартира:
- вид объекта недвижимости: «Помещение»;
- Назначение: «Жилое»;
- следующая строка: «Наименование: ...».

Результаты в рабочей папке:
- egrn_kv_success_cache.csv — успешно распознанные квартиры;
- egrn_kv_skipped_cache.csv — нормальные пропуски: файл + причина;
- egrn_kv_errors_cache.csv  — ошибки чтения PDF: файл + ошибка;
- EGRN_kvartiry.xlsx         — пересобирается из success + errors после
                                штатного завершения или Ctrl+C;
- EGRN_kvartiry_skipped.xlsx — пересобирается из skipped после штатного
                                завершения или Ctrl+C.

Запуск:
    python egrn_kv_extract_cached.py --csv target_paths-2.csv
    python egrn_kv_extract_cached.py --csv target_paths-2.csv --sep ";"
    python egrn_kv_extract_cached.py --csv target_paths-2.csv --sep ","

Зависимости:
    pip install pdfplumber openpyxl tqdm
"""

import argparse
import csv
import datetime as dt
import re
from pathlib import Path

import openpyxl
import pdfplumber
from tqdm import tqdm

MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5,
    "июня": 6, "июля": 7, "августа": 8, "сентября": 9,
    "октября": 10, "ноября": 11, "декабря": 12,
}

SUCCESS_HEADER = [
    "путь", "дата загрузки", "кадастровый номер", "адрес", "номер объекта",
    "площадь", "этаж", "дата собственности", "номер регистрации права",
    "вид права", "вид объекта", "назначение", "наименование",
    "дата получения ЕГРН", "кадастровый номер дома",
]
SKIPPED_HEADER = ["файл", "причина"]
ERRORS_HEADER = ["файл", "ошибка"]


def parse_ru_date(s: str) -> str:
    m = re.search(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", s, re.I)
    if not m:
        return ""
    day, month_name, year = m.groups()
    month = MONTHS.get(month_name.lower())
    return f"{int(day):02d}.{month:02d}.{year}" if month else ""


def load_pdf_paths(csv_path: Path, sep: str) -> list[Path]:
    """Читает список путей из колонки «путь» исходного CSV."""
    paths = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        if not reader.fieldnames or "путь" not in reader.fieldnames:
            raise SystemExit(
                f"ОШИБКА: в CSV нет колонки «путь». Найдены: {reader.fieldnames}"
            )
        for row in reader:
            raw_path = (row.get("путь") or "").strip()
            if raw_path:
                paths.append(Path(raw_path))
    return paths


def read_done_paths(*cache_paths: Path) -> set[str]:
    """Собирает пути из всех кэшей: их не надо открывать повторно."""
    done = set()
    for cache_path in cache_paths:
        if not cache_path.is_file():
            continue
        with cache_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                path_text = row.get("путь") or row.get("файл")
                if path_text:
                    done.add(path_text)
    return done


def open_cache(path: Path, header: list[str]):
    """Открывает CSV для дописывания и при первом запуске создаёт шапку."""
    is_new = not path.exists() or path.stat().st_size == 0
    f = path.open("a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(f, fieldnames=header)
    if is_new:
        writer.writeheader()
        f.flush()
    return f, writer


def extract_fields(pdf_path: Path):
    """Возвращает (словарь полей, '') либо (None, причина пропуска)."""
    with pdfplumber.open(pdf_path) as pdf:
        full = "\n".join((page.extract_text() or "") for page in pdf.pages)

    if not full.strip():
        return None, "PDF без извлекаемого текста (вероятно скан)"

    vid_m = re.search(r"\n(\S[^\n]*)\nвид объекта недвижимости", full, re.I)
    vid = vid_m.group(1).strip() if vid_m else ""
    is_parking = vid.casefold() in {"машино-место", "машиноместо"}
    if "помещение" not in vid.casefold() and not is_parking:
        return None, "вид объекта недвижимости не «Помещение» и не «Машино-место»"

    naznach_m = re.search(
        r"Назначение:\s*(\S[^\n]*)\nНаименование:\s*(\S[^\n]*)",
        full,
        re.I,
    )
    if not naznach_m and not is_parking:
        return None, "не найден блок «Назначение / Наименование»"

    if naznach_m:
        naznachenie = naznach_m.group(1).strip()
        naimenovanie = naznach_m.group(2).strip()
    else:
        naznach_only = re.search(r"Назначение:[ \t]*([^\n]*)", full, re.I)
        naimen_only = re.search(r"Наименование:[ \t]*([^\n]*)", full, re.I)
        naznachenie = naznach_only.group(1).strip() if naznach_only else ""
        naimenovanie = naimen_only.group(1).strip() if naimen_only else ""
    if not is_parking and naznachenie.casefold() not in {"жилое", "нежилое"}:
        return None, f"назначение не «Жилое» и не «Нежилое»: «{naznachenie}»"

    kad_m = re.search(r"Кадастровый номер:\s*(\S+)", full, re.I)
    kad = kad_m.group(1) if kad_m else ""

    date_top_m = re.search(
        r"\n(\d{1,2}\s+[а-яё]+\s+\d{4}\s*г?\.)", full, re.I
    )
    date_top = parse_ru_date(date_top_m.group(1)) if date_top_m else ""

    addr_m = re.search(r"Адрес:\s*(.*?)\nПлощадь", full, re.I | re.S)
    addr_raw = re.sub(r"\s+", " ", addr_m.group(1)).strip() if addr_m else ""

    obj_num = ""
    addr_clean = addr_raw
    kv_m = re.search(
        r",?\s*(?:кв|помещение|помещ|м/м|машиноместо)\.?\s*([0-9A-Za-zА-Яа-яЁё/_-]+)\s*$", addr_raw, re.I
    )
    if kv_m:
        obj_num = kv_m.group(1).strip()
        addr_clean = addr_raw[:kv_m.start()].strip(" ,")

    area_m = re.search(r"Площадь.*?:\s*([\d.,]+)", full, re.I)
    area = area_m.group(1) if area_m else ""

    floor_m = re.search(r"[Ээ]таж №\s*([-\d]+)", full)
    floor = floor_m.group(1) if floor_m else ""

    kad_house_m = re.search(
        r"Кадастровые номера иных объектов недвижимости\D+(\d\d:\d\d:\d\d\d\d\d\d\d:\d+)",
        full,
        re.I,
    )
    kad_house = kad_house_m.group(1) if kad_house_m else ""

    reg_m = re.search(
        r"регистрации\s+\d\.\d\s*(\S[^\n]*)\nправа:\s*(\S+)\n(\d{2}\.\d{2}\.\d{4})",
        full,
        re.I,
    )
    right_type = reg_m.group(1).strip() if reg_m else ""
    registration_number = reg_m.group(2) if reg_m else ""
    ownership_date = reg_m.group(3) if reg_m else ""

    if not kad or not addr_clean:
        missing = []
        if not kad:
            missing.append("кадастровый номер")
        if not addr_clean:
            missing.append("адрес")
        return None, "не найдены ключевые поля: " + ", ".join(missing)

    return {
        "кадастровый номер": kad,
        "адрес": addr_clean,
        "номер объекта": obj_num,
        "площадь": area,
        "этаж": floor,
        "дата собственности": ownership_date,
        "номер регистрации права": registration_number,
        "вид права": right_type,
        "вид объекта": vid,
        "назначение": naznachenie,
        "наименование": naimenovanie,
        "дата получения ЕГРН": date_top,
        "кадастровый номер дома": kad_house,
    }, ""


def write_final_xlsx(success_cache: Path, skipped_cache: Path, errors_cache: Path,
                     out_path: Path, skipped_out_path: Path):
    """Пересобирает итоговые xlsx только из уже сохранённых CSV-кэшей."""
    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet("Квартиры")
    ws.append([
        "№ п/п", "дата загрузки", "кадастровый номер",
        "Адрес помещения (ул, б-р, дом)", "номер объекта", "площадь", "этаж",
        "дата собственности", "номер регистрации права", "вид права",
        "вид объекта", "назначение", "наименование", "дата получения ЕГРН",
        "кадастровый номер дома", "файл", "ошибка",
    ])

    n = 0
    if success_cache.is_file():
        with success_cache.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                n += 1
                ws.append([
                    n, row["дата загрузки"], row["кадастровый номер"], row["адрес"],
                    row["номер объекта"], row["площадь"], row["этаж"],
                    row["дата собственности"], row["номер регистрации права"],
                    row["вид права"], row["вид объекта"], row["назначение"],
                    row["наименование"], row["дата получения ЕГРН"],
                    row["кадастровый номер дома"], row["путь"], "",
                ])

    if errors_cache.is_file():
        with errors_cache.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                ws.append([
                    None, "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                    row["файл"], row["ошибка"],
                ])
    wb.save(out_path)

    skip_wb = openpyxl.Workbook(write_only=True)
    skip_ws = skip_wb.create_sheet("Пропуски")
    skip_ws.append(SKIPPED_HEADER)
    if skipped_cache.is_file():
        with skipped_cache.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                skip_ws.append([row["файл"], row["причина"]])
    skip_wb.save(skipped_out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="исходный CSV; обязательна колонка «путь»")
    ap.add_argument("--sep", default=";", help="разделитель исходного CSV; по умолчанию ';'")
    ap.add_argument("--out", default="EGRN_kvartiry.xlsx", help="итоговый Excel с квартирами")
    ap.add_argument(
        "--skipped-out", default="EGRN_kvartiry_skipped.xlsx",
        help="итоговый Excel-лог нормальных пропусков",
    )
    ap.add_argument(
        "--cache-dir", default=".",
        help="папка для CSV-кэшей; по умолчанию текущая папка",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"ОШИБКА: CSV не найден: {csv_path}")
    if len(args.sep) != 1:
        raise SystemExit("ОШИБКА: --sep должен состоять из одного символа, например ';' или ','")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    success_cache = cache_dir / "egrn_kv_success_cache.csv"
    skipped_cache = cache_dir / "egrn_kv_skipped_cache.csv"
    errors_cache = cache_dir / "egrn_kv_errors_cache.csv"

    pdf_paths = load_pdf_paths(csv_path, args.sep)
    done_paths = read_done_paths(success_cache, skipped_cache, errors_cache)
    remaining = [p for p in pdf_paths if str(p) not in done_paths]

    print(f"путей прочитано из CSV: {len(pdf_paths)}")
    print(f"уже есть в кэше: {len(done_paths)}")
    print(f"осталось обработать: {len(remaining)}")

    success_file, success_writer = open_cache(success_cache, SUCCESS_HEADER)
    skipped_file, skipped_writer = open_cache(skipped_cache, SKIPPED_HEADER)
    errors_file, errors_writer = open_cache(errors_cache, ERRORS_HEADER)

    today = dt.date.today().strftime("%d.%m.%Y")
    session_success = session_skipped = session_errors = 0

    try:
        for pdf_path in tqdm(remaining, desc="Обработка PDF", unit="файл"):
            path_text = str(pdf_path)

            if not pdf_path.is_file():
                skipped_writer.writerow({"файл": path_text, "причина": "файл не найден по пути из CSV"})
                skipped_file.flush()
                session_skipped += 1
                continue

            try:
                fields, skip_reason = extract_fields(pdf_path)
            except Exception as e:
                # Ошибка чтения остаётся в основной итоговой таблице.
                errors_writer.writerow({"файл": path_text, "ошибка": f"ошибка чтения: {e}"})
                errors_file.flush()
                session_errors += 1
                continue

            if fields is None:
                skipped_writer.writerow({"файл": path_text, "причина": skip_reason})
                skipped_file.flush()
                session_skipped += 1
                continue

            success_writer.writerow({
                "путь": path_text,
                "дата загрузки": today,
                **fields,
            })
            success_file.flush()
            session_success += 1

    except KeyboardInterrupt:
        print("\nОстановка по Ctrl+C. Все завершённые PDF уже записаны в кэш.")

    finally:
        success_file.close()
        skipped_file.close()
        errors_file.close()

        print("Пересборка итоговых Excel из кэшей...")
        write_final_xlsx(
            success_cache, skipped_cache, errors_cache,
            Path(args.out), Path(args.skipped_out),
        )

    print(f"за этот запуск: квартир={session_success}, пропусков={session_skipped}, ошибок={session_errors}")
    print(f"основная выгрузка: {args.out}")
    print(f"лог пропусков: {args.skipped_out}")
    print(f"кэш: {cache_dir}")


if __name__ == "__main__":
    main()
