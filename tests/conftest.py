import pytest

from src.evaluation import evaluate_queries
from src.recommender import train_recommender
from src.storage import write_json_atomic


@pytest.fixture
def model():
    baskets = ([['10001', '10002', '10003']] * 8
               + [['10001', '10002']] * 4 + [['10002', '10004']] * 3)
    result = train_recommender(baskets, {
        '10001': 'BLUE MUG', '10002': 'RED MUG',
        '10003': 'BLUE PLATE', '10004': 'TEA TOWEL',
    }, min_orders=1)
    result.update({
        'build_id': 'test-build',
        'prices_gbp': {code: 2.5 for code in result['catalog']},
        'stats': {'train_orders': 15, 'training_line_items': 38},
    })
    return result


@pytest.fixture
def artifacts(tmp_path, model):
    model_path, report_path = tmp_path / 'model.json', tmp_path / 'evaluation.json'
    results = evaluate_queries(model, [(['10001'], {'10002', '10003'})])
    report = {
        'build_id': model['build_id'], 'results': results, 'basket_results': results,
        'evaluated_test_invoices': 1, 'evaluated_basket_invoices': 1,
        'split': {'cutoff': '2011-10-26T13:27:00'},
        'single_product_task': 'Reveal one item.', 'basket_task': 'Reveal two items.',
        'notes': 'Synthetic fixture, not reported project results.',
        'test_catalog_item_fraction': 1.0,
    }
    write_json_atomic(model_path, model)
    write_json_atomic(report_path, report)
    return model_path, report_path
