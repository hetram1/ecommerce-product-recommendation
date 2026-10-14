import pandas as pd
from src.recommender import invoice_baskets, train_recommender, recommend
from src.data import load_and_clean


def test_cleaning_removes_cancellations_and_non_products(tmp_path):
    path = tmp_path / 'orders.csv'
    pd.DataFrame([
        ['1', '12345', 'BLUE MUG', 1, '2011-01-01', 2.0],
        ['C2', '12345', 'BLUE MUG', -1, '2011-01-02', 2.0],
        ['3', 'POST', 'POSTAGE', 1, '2011-01-02', 2.0],
        ['4', '12346', 'RED MUG', 0, '2011-01-02', 2.0],
        ['5', '12346', 'RED MUG', 2, '2011-01-02', 4.0],
    ], columns=['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice']).to_csv(path, index=False)
    cleaned, stats = load_and_clean(path)
    assert len(cleaned) == 2
    assert stats['original_rows'] == 5
    assert set(cleaned['StockCode']) == {'12345', '12346'}


def test_recommendation_excludes_source_item_and_uses_copurchases():
    baskets = [['A', 'B'], ['A', 'B'], ['A', 'C'], ['B', 'C']]
    names = {key: key for key in ('A', 'B', 'C')}
    model = train_recommender(baskets, names, max_products=3, min_orders=1)
    result = recommend(model, 'A', k=2)
    assert result[0]['id'] == 'B'
    assert 'A' not in [r['id'] for r in result]
    assert recommend(model, 'unknown', k=2, method='popularity')


def test_invoice_baskets_deduplicate_products():
    df = pd.DataFrame({'InvoiceNo': ['1', '1', '1', '2', '2'],
                       'StockCode': ['A', 'A', 'B', 'A', 'C']})
    assert invoice_baskets(df) == [['A', 'B'], ['A', 'C']]
