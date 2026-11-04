from pathlib import Path

from streamlit.testing.v1 import AppTest

import src.storage as storage

APP = Path(__file__).resolve().parents[1] / 'app.py'


def configured_app(monkeypatch, artifacts):
    monkeypatch.setattr(storage, 'MODEL_PATH', artifacts[0])
    monkeypatch.setattr(storage, 'EVALUATION_PATH', artifacts[1])
    return AppTest.from_file(str(APP), default_timeout=120).run()


def test_search_basket_add_remove_clear_and_analytics(monkeypatch, artifacts):
    app = configured_app(monkeypatch, artifacts)
    assert not app.exception
    app.text_input(key='search').set_value('mug').run()
    selected = app.selectbox(key='selected_product').value
    app.button(key='add_selected').click().run()
    assert app.session_state.basket == [selected]
    assert app.button(key='add_selected').disabled
    app.radio(key='workspace').set_value('Basket').run()
    add = next(button for button in app.button if str(button.key).startswith('basket_add_'))
    add.click().run()
    assert len(app.session_state.basket) == 2
    app.button(key=f'remove_{selected}').click().run()
    assert selected not in app.session_state.basket
    next(button for button in app.button if button.label == 'Clear basket').click().run()
    assert app.session_state.basket == []
    app.radio(key='workspace').set_value('Analytics').run(timeout=180)
    assert not app.exception
    assert len(app.dataframe[0].value) == 3
    next(radio for radio in app.radio if radio.label == 'Evaluation task').set_value('Basket').run()
    next(select for select in app.selectbox if select.label == 'Recommendation cutoff').set_value(10).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 3


def test_search_empty_state(monkeypatch, artifacts):
    app = configured_app(monkeypatch, artifacts)
    app.text_input(key='search').set_value('no-such-product').run()
    assert not app.exception
    assert any('No matching products' in message.value for message in app.info)


def test_missing_model_has_rebuild_instructions(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, 'MODEL_PATH', tmp_path / 'missing.json')
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert app.error and 'src.build' in app.error[0].value


def test_unmeasured_evaluation_displays_without_fabricating_results(monkeypatch, artifacts):
    report = storage.read_json(artifacts[1])
    for task in ('results', 'basket_results'):
        for metrics in report[task].values():
            for metric in metrics:
                metrics[metric] = None
    report.update({'evaluated_test_invoices': 0, 'evaluated_basket_invoices': 0,
                   'test_catalog_item_fraction': None})
    storage.write_json_atomic(artifacts[1], report)
    app = configured_app(monkeypatch, artifacts)
    app.radio(key='workspace').set_value('Analytics').run(timeout=180)
    assert not app.exception
    assert app.dataframe[0].value['Recall'].isna().all()
    assert any('unmeasured' in text.value for text in app.markdown)
