"""Download and clean official UCI Online Retail transactions."""
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import urllib.request
from zipfile import ZipFile

import numpy as np
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice"]


def get_dataset(path: Path) -> Path:
    """An interrupted download/extraction must never become cached source data."""
    if path.is_file():
        return path
    if path.suffix.lower() != ".xlsx":
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the official UCI Online Retail dataset (~23 MB)...", flush=True)
    try:
        with TemporaryDirectory(dir=path.parent) as temp_dir:
            archive_path = Path(temp_dir) / "download.zip"
            with urllib.request.urlopen(DATA_URL, timeout=90) as response, archive_path.open("wb") as dest:
                shutil.copyfileobj(response, dest)
            with ZipFile(archive_path) as archive:
                members = [n for n in archive.namelist() if n.lower().endswith("online retail.xlsx")]
                if not members:
                    raise ValueError("Expected spreadsheet missing from UCI download")
                staged = Path(temp_dir) / "retail.xlsx"
                with archive.open(members[0]) as source, staged.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                staged.replace(path)
    except Exception as exc:
        raise RuntimeError(
            "Automatic download failed. Download Online Retail.xlsx from "
            "https://archive.ics.uci.edu/dataset/352/online+retail, "
            f"save it at {path}, and rerun the build."
        ) from exc
    return path


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    """Keep valid product purchases; basket training deduplicates product IDs."""
    reader = pd.read_csv if path.suffix.lower() == ".csv" else pd.read_excel
    kwargs = {} if path.suffix.lower() == ".csv" else {"engine": "openpyxl"}
    raw = reader(path, usecols=COLUMNS, dtype={"InvoiceNo": str, "StockCode": str}, **kwargs)
    original_rows = len(raw)
    df = raw.dropna(subset=COLUMNS).copy()
    for column in ("InvoiceNo", "StockCode", "Description"):
        df[column] = df[column].astype(str).str.strip().str.upper()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", format="mixed")
    for column in ("Quantity", "UnitPrice"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    valid = (
        df["InvoiceNo"].str.len().gt(0)
        & ~df["InvoiceNo"].str.startswith("C")
        & df["StockCode"].str.fullmatch(r"\d{5}[A-Z]?").fillna(False)
        & df["Description"].str.len().gt(1)
        & df["Quantity"].gt(0) & np.isfinite(df["Quantity"])
        & df["UnitPrice"].gt(0) & np.isfinite(df["UnitPrice"])
        & df["InvoiceDate"].notna()
    )
    df = df.loc[valid].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df = df.loc[np.isfinite(df["Revenue"])].sort_values(
        ["InvoiceDate", "InvoiceNo", "StockCode"], kind="stable",
    )
    return df, {"original_rows": original_rows, "clean_rows": len(df),
                "removed_rows": original_rows - len(df)}
