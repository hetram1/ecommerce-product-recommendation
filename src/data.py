"""Download and clean the UCI Online Retail transactions."""
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import urllib.request
from zipfile import ZipFile

import pandas as pd

DATA_URL = 'https://archive.ics.uci.edu/static/public/352/online+retail.zip'
COLUMNS = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice']


def get_dataset(path: Path) -> Path:
    """Use a local spreadsheet when available; otherwise fetch the official UCI archive."""
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print('Downloading the official UCI Online Retail dataset (~23 MB)...')
    try:
        with TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / 'download.zip'
            with urllib.request.urlopen(DATA_URL, timeout=90) as response, open(archive_path, 'wb') as dest:
                shutil.copyfileobj(response, dest)
            with ZipFile(archive_path) as archive:
                members = [n for n in archive.namelist() if n.lower().endswith('online retail.xlsx')]
                if not members:
                    raise ValueError('Expected spreadsheet missing from UCI download')
                with archive.open(members[0]) as source, open(path, 'wb') as dest:
                    shutil.copyfileobj(source, dest)
    except Exception as exc:
        raise RuntimeError(
            'Automatic download failed. Get "Online Retail.xlsx" from '
            'https://archive.ics.uci.edu/dataset/352/online+retail, '
            f'put it at {path}, and rerun this command.'
        ) from exc
    return path


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    """Filter cancellations, invalid rows, and non-product stock codes."""
    if path.suffix.lower() == '.csv':
        raw = pd.read_csv(path, usecols=COLUMNS)
    else:
        raw = pd.read_excel(path, usecols=COLUMNS, engine='openpyxl')
    original_rows = len(raw)
    df = raw.dropna(subset=COLUMNS).copy()
    df['InvoiceNo'] = df['InvoiceNo'].astype(str).str.strip()
    df['StockCode'] = df['StockCode'].astype(str).str.strip().str.upper()
    df['Description'] = df['Description'].astype(str).str.strip().str.upper()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    valid = (
        ~df['InvoiceNo'].str.upper().str.startswith('C')
        & df['StockCode'].str.fullmatch(r'\d{5}[A-Z]?').fillna(False)
        & df['Description'].str.len().gt(1)
        & df['Quantity'].gt(0)
        & df['UnitPrice'].gt(0)
        & df['InvoiceDate'].notna()
    )
    df = df.loc[valid].copy()
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    df = df.sort_values('InvoiceDate')
    return df, {'original_rows': original_rows, 'clean_rows': len(df)}
