from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import pandas as pd

SHEET = "Ценообразование"
HEADER_ROWS = 2


def run():
    root = tk.Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(
        title="Выберите Excel-файлы",
        filetypes=[("Excel", "*.xlsx *.xlsm")],
    )
    if not files:
        return

    output = filedialog.asksaveasfilename(
        title="Куда сохранить результат",
        defaultextension=".xlsx",
        initialfile="объединено.xlsx",
        filetypes=[("Excel", "*.xlsx")],
    )
    if not output:
        return

    tables = [pd.read_excel(path, sheet_name=SHEET, header=None) for path in files]
    result = pd.concat([tables[0]] + [table.iloc[HEADER_ROWS:] for table in tables[1:]], ignore_index=True)
    result.to_excel(output, sheet_name=SHEET, index=False, header=False)


if __name__ == "__main__":
    run()
