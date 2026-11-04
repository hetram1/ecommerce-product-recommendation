import math

import pytest

from src.recommender import recommend, recommend_basket, train_recommender


def fit(baskets, **kwargs):
    return train_recommender(baskets, {}, min_orders=1, **kwargs)


def ids(items):
    return [item['id'] for item in items]


def test_cosine_matches_hand_calculation_and_binary_membership():
    model = fit([['A', 'A', 'B'], ['A', 'B'], ['A', 'C'], ['B', 'D']], min_pair_orders=1)
    assert model['order_counts']['A'] == 3
    scores = {item['id']: item for item in model['recommendations']['A']}
    assert scores['B']['score'] == pytest.approx(2 / 3)
    assert scores['C']['score'] == pytest.approx(1 / math.sqrt(3))
    assert scores['B']['joint_orders'] == 2


def test_cosine_and_copurchase_have_distinct_useful_rankings():
    model = fit([['A', 'B']] * 10 + [['A', 'C']] * 8 + [['B', 'D']] * 90)
    assert recommend(model, 'A', 1, 'similarity')[0]['id'] == 'C'
    assert recommend(model, 'A', 1, 'frequently_bought')[0]['id'] == 'B'


def test_one_invoice_associations_are_filtered():
    model = fit([['A', 'B'], ['A', 'B'], ['A', 'C']])
    assert ids(model['recommendations']['A']) == ['B']
    assert 'C' not in ids(model['bought_together']['A'])


def test_fallback_fills_large_request_after_exclusions():
    products = [f'P{i:02}' for i in range(40)]
    model = fit([products] * 2, top_k=2)
    result = recommend_basket(model, products[:10] + ['unknown'], k=20, method='popularity')
    assert len(result) == len(set(ids(result))) == 20
    assert set(ids(result)).isdisjoint(products[:10])
    assert ids(result) == products[10:30]


def test_basket_order_and_duplicate_inputs_do_not_change_ranking(model):
    left = recommend_basket(model, ['10001', '10002', '10001'], k=10)
    right = recommend_basket(model, ['10002', '10001'], k=10)
    assert left == right
    assert set(ids(left)) == {'10003', '10004'}
    assert left[0]['matched_products'] == ['10001', '10002']
    assert left[0]['joint_orders'] is None  # Joint invoices across seeds are not double-counted as unique.


def test_copurchase_basket_normalization_prevents_popular_seed_dominance():
    model = fit([['A', 'C']] * 10 + [['A', 'X']] * 90
                + [['B', 'D']] * 9 + [['B', 'Y']], min_pair_orders=1)
    ranked = recommend_basket(model, ['A', 'B'], k=4, method='frequently_bought')
    assert ids(ranked).index('D') < ids(ranked).index('C')
    assert next(item['score'] for item in ranked if item['id'] == 'D') == pytest.approx(.45)


def test_empty_and_unknown_baskets_fall_back_without_duplicates(model):
    assert ids(recommend_basket(model, [], 4)) == model['catalog']
    assert ids(recommend(model, 'unknown', 4)) == model['catalog']
    assert len(recommend_basket(model, model['catalog'], 5)) == 0


def test_extra_exclusions_apply_to_neighbors_and_fallback(model):
    assert ids(recommend(model, '10001', 10, exclude=['10002', '10003'])) == ['10004']


def test_ranking_and_catalog_ties_are_reproducible():
    baskets = [['C', 'A'], ['A', 'B'], ['B', 'C']]
    left = fit(baskets, min_pair_orders=1)
    right = fit([list(reversed(b)) for b in reversed(baskets)], min_pair_orders=1)
    assert left == right
    assert ids(recommend(left, 'A', 2)) == ['B', 'C']


@pytest.mark.parametrize('k', [0, -1, 51, True, 2.5])
def test_invalid_result_count_is_rejected(model, k):
    with pytest.raises(ValueError):
        recommend(model, '10001', k)


def test_invalid_method_is_rejected(model):
    with pytest.raises(ValueError, match='Unknown method'):
        recommend(model, '10001', method='typo')


@pytest.mark.parametrize('parameter', ['max_products', 'min_orders', 'top_k', 'min_pair_orders'])
def test_invalid_training_configuration_is_rejected(parameter):
    with pytest.raises(ValueError):
        train_recommender([['A', 'B']], {}, **{parameter: 0})
