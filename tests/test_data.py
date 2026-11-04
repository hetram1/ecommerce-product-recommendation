from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest

from src.data import COLUMNS, get_dataset, load_and_clean


@pytest.mark.parametrize('column,value', [
    ('InvoiceNo', '  '), ('InvoiceNo', ' c123 '), ('StockCode', 'POST'),
    ('Description', ' '), ('InvoiceDate', 'not-a-date'), ('Quantity', 0),
    ('Quantity', float('inf')), ('UnitPrice', -1), ('UnitPrice', float('inf')),
])
def test_invalid_transactions_are_removed(tmp_path, column, value):
    valid = dict(zip(COLUMNS, ['1', '00123', 'BLUE MUG', 1, '2011-01-01', 2]))
    invalid = {**valid, column: value}
    path = tmp_path / 'data.csv'
    pd.DataFrame([valid, invalid]).to_csv(path, index=False)
    cleaned, stats = load_and_clean(path)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['StockCode'] == '00123'
    assert stats['removed_rows'] == 1


def test_missing_csv_does_not_download_an_excel_file_under_wrong_extension(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_dataset(tmp_path / 'missing.csv')


def test_failed_download_does_not_leave_partial_source(monkeypatch, tmp_path):
    monkeypatch.setattr('src.data.urllib.request.urlopen', lambda *a, **k: BytesIO(b'broken archive'))
    target = tmp_path / 'Online Retail.xlsx'
    with pytest.raises(RuntimeError, match='Automatic download failed'):
        get_dataset(target)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_successful_download_extracts_only_expected_file(monkeypatch, tmp_path):
    content = BytesIO()
    with ZipFile(content, 'w') as archive:
        archive.writestr('nested/Online Retail.xlsx', b'spreadsheet fixture')
        archive.writestr('../unrelated.txt', b'never extracted')
    monkeypatch.setattr('src.data.urllib.request.urlopen', lambda *a, **k: BytesIO(content.getvalue()))
    target = tmp_path / 'Online Retail.xlsx'
    assert get_dataset(target).read_bytes() == b'spreadsheet fixture'
    assert list(tmp_path.iterdir()) == [target]
