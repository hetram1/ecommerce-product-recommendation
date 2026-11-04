import pandas as pd
import pytest

from src.build import main
from src.evaluation import chronological_split, evaluate, evaluate_queries
from src.recommender import METHODS


def test_split_keeps_timestamp_ties_and_spanning_invoices_out_of_training():
    df = pd.DataFrame({
        'InvoiceNo': ['A', 'A', 'B', 'C', 'D', 'E', 'F'],
        'InvoiceDate': pd.to_datetime(['2011-01-01', '2011-01-09', '2011-01-02',
                                       '2011-01-03', '2011-01-04', '2011-01-04', '2011-01-06']),
    })
    train, test, split = chronological_split(df, .5)
    assert set(train['InvoiceNo']) == {'B', 'C'}
    assert set(test['InvoiceNo']) == {'D', 'E', 'F'}
    assert train['InvoiceDate'].max() < test['InvoiceDate'].min()
    assert set(train['InvoiceNo']).isdisjoint(test['InvoiceNo'])
    assert split['spanning_invoices_dropped'] == 1


def test_metrics_match_hand_worked_hits_recall_and_catalog_coverage():
    codes = [f'P{i:02}' for i in range(12)]
    model = {
        'catalog': codes, 'order_counts': dict.fromkeys(codes, 1), 'train_baskets': 20,
        'popular': [{'id': code, 'orders': 1} for code in codes],
        'recommendations': {}, 'bought_together': {},
    }
    result = evaluate_queries(model, [(['P11'], {'P00', 'P06'}), (['P00'], {'P11'})])
    for method in METHODS:
        assert result[method] == {
            'hit_rate_at_5': .5, 'recall_at_5': .25, 'coverage_at_5': .5,
            'hit_rate_at_10': .5, 'recall_at_10': .5, 'coverage_at_10': .916667,
        }


def test_no_queries_yields_unmeasured_metrics_not_invented_zeros(model):
    result = evaluate_queries(model, [])
    assert all(value is None for metrics in result.values() for value in metrics.values())


def test_revealed_truth_is_rejected(model):
    with pytest.raises(ValueError):
        evaluate_queries(model, [(['10001'], {'10001'})])


def test_evaluation_is_reproducible_and_counts_only_eligible_tasks(model):
    test = pd.DataFrame({
        'InvoiceNo': ['1'] * 4 + ['2'] * 2 + ['3'] * 2,
        'StockCode': ['10001', '10002', '10003', 'unseen', '10001', '10002', 'unseen', 'other'],
    })
    first = evaluate(model, test)
    assert first == evaluate(model, test)
    assert first['size_filtered_test_baskets'] == 3
    assert first['evaluated_test_invoices'] == 2
    assert first['evaluated_basket_invoices'] == 1
    assert first['test_catalog_item_fraction'] == .625


def test_build_never_learns_future_products_names_prices_or_popularity(tmp_path):
    rows = []
    for invoice in range(12):
        future = invoice >= 9
        for code in ('10001', '10003' if future else '10002'):
            rows.append([str(invoice), code, 'FUTURE NAME' if future else f'EARLY {code}',
                         999 if future else 1, f'2011-01-{invoice + 1:02}', 999 if future else 2])
    path = tmp_path / 'transactions.csv'
    pd.DataFrame(rows, columns=['InvoiceNo', 'StockCode', 'Description', 'Quantity',
                                'InvoiceDate', 'UnitPrice']).to_csv(path, index=False)
    model, report = main(path, output_dir=tmp_path)
    assert '10003' not in model['catalog']
    assert model['names']['10001'] == 'EARLY 10001'
    assert model['prices_gbp']['10001'] == 2
    assert model['order_counts']['10001'] == 9
    assert report['build_id'] == model['build_id']
    assert report['evaluated_test_invoices'] == 0


@pytest.mark.parametrize('fraction', [0, 1, -1])
def test_invalid_split_configuration(fraction):
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame(), fraction)
