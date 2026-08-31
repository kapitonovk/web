#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_symlink_tree.py

Собирает "виртуальное" дерево root/<адрес>/*.pdf из CSV со списком целевых
PDF (результат build_pdf_target_list.py), используя символические ссылки —
без физического копирования файлов. Дерево совместимо с текущими
egrn_*_extract.py, которые ожидают структуру root/*/*.pdf.

Имя папки-адреса берётся:
  - либо из колонки "адрес" CSV, если она есть,
  - либо (по умолчанию) как имя непосредственной родительской папки исходного файла.

Запуск:
  python3 build_symlink_tree.py --csv target_pdfs.csv --dest ./work_tree
  python3 egrn_bkfn_extract-3.py --root ./work_tree --out result.xlsx

На Windows символические ссылки требуют прав администратора либо
включённого Developer Mode. Если симлинки создать не удаётся, используйте
--copy для обычного копирования (медленнее, занимает место на диске).
"""

import argparse
import csv
import shutil
from pathlib import Path


def unique_target(dest_dir: Path, name: str) -> Path:
    target = dest_dir / name
    if not target.exists():
        return target
    stem, suffix = Path(name).stem, Path(name).suffix
    counter = 2
    while True:
        target = dest_dir / f"{stem}__{counter}{suffix}"
        if not target.exists():
            return target
        counter += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV из build_pdf_target_list.py (include-список)")
    ap.add_argument("--dest", required=True, help="куда собрать дерево root/адрес/*.pdf")
    ap.add_argument("--copy", action="store_true", help="копировать вместо симлинков")
    args = ap.parse_args()

    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    created, skipped_missing, errors = 0, 0, 0

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        has_addr_col = "адрес" in (reader.fieldnames or [])
        for row in reader:
            src = Path(row["путь"])
            if not src.is_file():
                skipped_missing += 1
                continue

            addr_name = row["адрес"].strip() if has_addr_col and row.get("адрес") else src.parent.name
            addr_dir = dest_root / addr_name
            addr_dir.mkdir(parents=True, exist_ok=True)

            target = unique_target(addr_dir, src.name)
            try:
                if args.copy:
                    shutil.copy2(src, target)
                else:
                    target.symlink_to(src.resolve())
                created += 1
            except OSError as e:
                print(f"ОШИБКА: {src} -> {target} | {e}")
                errors += 1

    print(f"создано ссылок/копий: {created}")
    print(f"пропущено (файл не найден): {skipped_missing}")
    print(f"ошибок: {errors}")
    print(f"дерево готово: {dest_root}")


if __name__ == "__main__":
    main()
