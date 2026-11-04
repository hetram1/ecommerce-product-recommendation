import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.service import CatalogService
from src.storage import ArtifactError, read_json, write_json_atomic


@pytest.fixture
def client(artifacts):
    with TestClient(create_app(*artifacts)) as test_client:
        yield test_client


def test_health_search_pagination_and_exact_code_priority(client):
    assert client.get('/health').json()['status'] == 'ok'
    result = client.get('/api/v1/products', params={'q': ' mug ', 'limit': 1}).json()
    assert result['total'] == 2
    assert len(result['items']) == 1
    next_page = client.get('/api/v1/products', params={'q': 'MUG', 'limit': 1, 'offset': 1}).json()
    assert result['items'][0]['id'] != next_page['items'][0]['id']
    assert client.get('/api/v1/products?q=10001').json()['items'][0]['id'] == '10001'
    assert client.get('/api/v1/products?q=no-match').json()['items'] == []


@pytest.mark.parametrize('method', ['similarity', 'frequently_bought', 'popularity'])
def test_single_recommendations_have_explanations_and_exclude_source(client, method):
    response = client.get(f'/api/v1/recommendations/10001?method={method}&k=10')
    assert response.status_code == 200
    data = response.json()
    assert data['query_products'] == ['10001']
    items = data['recommendations']
    assert len(items) == len({item['id'] for item in items}) == 3
    assert all(item['id'] != '10001' and item['reason'] for item in items)
    assert all(item['price_gbp'] == 2.5 for item in items)


def test_basket_deduplicates_normalizes_and_excludes_every_seed(client):
    response = client.post('/api/v1/recommendations/basket', json={
        'product_ids': [' 10001 ', '10002', '10001'], 'k': 10,
    })
    assert response.status_code == 200
    data = response.json()
    assert data['query_products'] == ['10001', '10002']
    assert {item['id'] for item in data['recommendations']} == {'10003', '10004'}


def test_empty_basket_returns_popular_starting_points(client):
    result = client.post('/api/v1/recommendations/basket', json={}).json()
    assert len(result['recommendations']) == 4
    assert all(item['source'] == 'popularity' for item in result['recommendations'])


def test_unknown_product_errors_are_actionable_and_never_partial(client):
    assert client.get('/api/v1/products/unknown').status_code == 404
    assert client.get('/api/v1/recommendations/unknown').status_code == 404
    response = client.post('/api/v1/recommendations/basket', json={'product_ids': ['10001', 'unknown']})
    assert response.status_code == 404
    assert response.json()['detail']['product_ids'] == ['UNKNOWN']


@pytest.mark.parametrize('path', [
    '/api/v1/products?limit=0', '/api/v1/products?limit=101', '/api/v1/products?offset=-1',
    '/api/v1/products?q=' + 'x' * 101, '/api/v1/recommendations/10001?k=0',
    '/api/v1/recommendations/10001?k=51', '/api/v1/recommendations/10001?method=typo',
])
def test_invalid_query_parameters_return_422(client, path):
    assert client.get(path).status_code == 422


@pytest.mark.parametrize('payload', [
    {'product_ids': ['']}, {'product_ids': ['   ']}, {'product_ids': [123]},
    {'product_ids': ['10001'] * 51}, {'product_ids': '10001'}, {'k': True},
    {'k': 1.5}, {'k': 0}, {'k': 51}, {'method': 'typo'}, {'unexpected': 'field'},
])
def test_invalid_basket_payload_returns_422(client, payload):
    assert client.post('/api/v1/recommendations/basket', json=payload).status_code == 422


def test_unavailable_model_returns_503_but_docs_stay_available(tmp_path):
    with TestClient(create_app(tmp_path / 'missing.json')) as client:
        assert client.get('/health').status_code == 503
        assert client.get('/api/v1/products').status_code == 503
        assert client.get('/docs').status_code == 200


def test_stale_evaluation_does_not_break_recommendations(artifacts):
    model_path, report_path = artifacts
    report = read_json(report_path)
    report['build_id'] = 'another-build'
    write_json_atomic(report_path, report)
    with TestClient(create_app(model_path, report_path)) as client:
        assert client.get('/api/v1/evaluation').status_code == 503
        assert client.get('/api/v1/recommendations/10001').status_code == 200


def test_search_cache_does_not_leak_mutable_responses(model):
    service = CatalogService(model)
    first = service.search('mug')
    first['items'][0]['name'] = 'changed by client'
    assert service.search('MUG')['items'][0]['name'] != 'changed by client'
    assert service._find.cache_info().hits >= 1


def test_invalid_model_is_rejected_before_serving(model):
    model['recommendations']['10001'][0]['score'] = float('nan')
    with pytest.raises(ArtifactError):
        CatalogService(model)


@pytest.mark.parametrize('field,value', [('build_id', ''), ('prices_gbp', [])])
def test_malformed_artifact_fields_fail_with_actionable_error(model, field, value):
    model[field] = value
    with pytest.raises(ArtifactError, match='src.build'):
        CatalogService(model)


@pytest.mark.parametrize('problem', ['missing_metrics', 'invalid_metric', 'missing_notes'])
def test_incomplete_evaluation_returns_503_without_breaking_search(artifacts, problem):
    model_path, report_path = artifacts
    report = read_json(report_path)
    if problem == 'missing_metrics':
        report['results'] = {}
    elif problem == 'invalid_metric':
        report['results']['similarity']['recall_at_5'] = 3.5
    else:
        del report['notes']
    write_json_atomic(report_path, report)
    with TestClient(create_app(model_path, report_path)) as client:
        assert client.get('/api/v1/evaluation').status_code == 503
        assert client.get('/api/v1/products').status_code == 200
