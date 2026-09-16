#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path


DEFAULT_EXCLUDE = {
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "$RECYCLE.BIN",
    "System Volume Information",
}


def natural_key(name: str):
    """Сортировка без учёта регистра; кириллица сохраняется как есть."""
    return name.casefold()


def format_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024

    return f"{n} B"


def iter_entries(path: Path):
    """Безопасно получает и сортирует содержимое папки."""
    try:
        with os.scandir(path) as scan:
            return sorted(scan, key=lambda x: natural_key(x.name))
    except PermissionError:
        return None
    except OSError as exc:
        return exc


def write_tree(
    root: Path,
    out,
    max_depth: int | None,
    max_items_per_dir: int | None,
    excluded: set[str],
    show_size: bool,
):
    stats = Counter()
    extensions = Counter()

    def walk(folder: Path, prefix: str = "", depth: int = 0):
        entries = iter_entries(folder)

        if entries is None:
            out.write(prefix + "[нет доступа]\n")
            stats["permission_errors"] += 1
            return

        if isinstance(entries, OSError):
            out.write(prefix + f"[ошибка чтения: {entries}]\n")
            stats["errors"] += 1
            return

        visible = []
        skipped_excluded = 0

        for entry in entries:
            if entry.name in excluded and entry.is_dir(follow_symlinks=False):
                skipped_excluded += 1
                continue
            visible.append(entry)

        if skipped_excluded:
            out.write(prefix + f"[исключено папок: {skipped_excluded}]\n")

        omitted = 0
        if max_items_per_dir is not None and len(visible) > max_items_per_dir:
            omitted = len(visible) - max_items_per_dir
            visible = visible[:max_items_per_dir]

        for index, entry in enumerate(visible):
            is_last = index == len(visible) - 1 and omitted == 0
            branch = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
            entry_path = folder / entry.name

            try:
                if entry.is_symlink():
                    out.write(prefix + branch + entry.name + " -> [symlink]\n")
                    stats["symlinks"] += 1
                    continue

                if entry.is_dir(follow_symlinks=False):
                    out.write(prefix + branch + entry.name + "/\n")
                    stats["dirs"] += 1

                    if max_depth is not None and depth >= max_depth:
                        out.write(child_prefix + "[глубина ограничена]\n")
                        stats["depth_limited_dirs"] += 1
                    else:
                        walk(entry_path, child_prefix, depth + 1)

                elif entry.is_file(follow_symlinks=False):
                    suffix = ""
                    size = 0

                    if show_size:
                        try:
                            size = entry.stat(follow_symlinks=False).st_size
                            suffix = f" ({format_size(size)})"
                        except OSError:
                            suffix = " (размер недоступен)"

                    out.write(prefix + branch + entry.name + suffix + "\n")
                    stats["files"] += 1
                    stats["bytes"] += size

                    ext = Path(entry.name).suffix.lower() or "[без расширения]"
                    extensions[ext] += 1

                else:
                    out.write(prefix + branch + entry.name + " [прочее]\n")
                    stats["other"] += 1

            except PermissionError:
                out.write(prefix + branch + entry.name + " [нет доступа]\n")
                stats["permission_errors"] += 1
            except OSError as exc:
                out.write(prefix + branch + entry.name + f" [ошибка: {exc}]\n")
                stats["errors"] += 1

        if omitted:
            out.write(prefix + f"└── … ещё {omitted} элементов не показано\n")
            stats["omitted_items"] += omitted

    out.write(f"ROOT: {root}\n\n")
    out.write(root.name + "/\n")
    stats["dirs"] += 1
    walk(root)

    return stats, extensions


def main():
    parser = argparse.ArgumentParser(
        description="Экспорт структуры папки в UTF-8 tree.txt и inventory.csv"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Путь к анализируемой папке. Если не указан, программа спросит его."
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Максимальная глубина вложенности. Например: --depth 6"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Максимум элементов, показываемых в одной папке. Например: --max-items 300"
    )
    parser.add_argument(
        "--sizes",
        action="store_true",
        help="Показывать размеры файлов. Медленнее на очень больших папках."
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Не исключать стандартные тяжёлые служебные папки."
    )
    parser.add_argument(
        "--output",
        default="tree.txt",
        help="Имя итогового tree-файла. По умолчанию tree.txt"
    )
    parser.add_argument(
        "--inventory",
        default="inventory.csv",
        help="Имя CSV-сводки по расширениям. По умолчанию inventory.csv"
    )

    args = parser.parse_args()

    raw_path = args.folder
    if not raw_path:
        raw_path = input("Вставь путь к папке и нажми Enter:\n> ").strip().strip('"')

    root = Path(raw_path).expanduser()

    if not root.is_dir():
        print(f"Ошибка: папка не найдена или это не папка:\n{root}")
        sys.exit(1)

    output = Path(args.output).expanduser()
    inventory = Path(args.inventory).expanduser()

    excluded = set() if args.include_hidden else DEFAULT_EXCLUDE

    print(f"Сканирую: {root}")
    print(f"Tree:    {output.resolve()}")
    print(f"CSV:     {inventory.resolve()}")

    # newline="" важен для корректного CSV в Windows;
    # encoding="utf-8-sig" добавляет BOM, поэтому Excel обычно открывает кириллицу правильно.
    with output.open("w", encoding="utf-8", newline="\n") as out:
        stats, extensions = write_tree(
            root=root,
            out=out,
            max_depth=args.depth,
            max_items_per_dir=args.max_items,
            excluded=excluded,
            show_size=args.sizes,
        )

        out.write("\n\n")
        out.write("===== SUMMARY =====\n")
        out.write(f"Папок: {stats['dirs']}\n")
        out.write(f"Файлов: {stats['files']}\n")

        if args.sizes:
            out.write(f"Размер учтённых файлов: {format_size(stats['bytes'])}\n")

        if stats["symlinks"]:
            out.write(f"Символических ссылок: {stats['symlinks']}\n")

        if stats["permission_errors"]:
            out.write(f"Недоступных объектов: {stats['permission_errors']}\n")

        if stats["errors"]:
            out.write(f"Ошибок чтения: {stats['errors']}\n")

        if stats["depth_limited_dirs"]:
            out.write(
                f"Папок, где сработало ограничение глубины: "
                f"{stats['depth_limited_dirs']}\n"
            )

        if stats["omitted_items"]:
            out.write(
                f"Не показано элементов из-за --max-items: "
                f"{stats['omitted_items']}\n"
            )

    with inventory.open("w", encoding="utf-8-sig", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["extension", "file_count"])

        for ext, count in extensions.most_common():
            writer.writerow([ext, count])

    print("\nГотово.")
    print("Пришли сюда tree.txt.")
    print("Если он всё ещё большой — пришли также inventory.csv и первые/важные фрагменты tree.txt.")


if __name__ == "__main__":
    main()