"""Simple Streamlit interface for item-based retail recommendations."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.recommender import recommend

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Product Recommendations', page_icon='🛍️', layout='wide')


@st.cache_data
def load_artifacts():
    model = json.loads((ROOT / 'artifacts' / 'recommender.json').read_text(encoding='utf-8'))
    metrics = json.loads((ROOT / 'reports' / 'evaluation.json').read_text(encoding='utf-8'))
    return model, metrics


st.title('🛍️ E-Commerce Product Recommendation System')
st.caption('Real UCI Online Retail transactions · purchase-based recommendations · no user ratings')

if not (ROOT / 'artifacts' / 'recommender.json').is_file():
    st.warning('Model not built yet. Run `python -m src.build`, then refresh this page.')
    st.stop()

model, evaluation = load_artifacts()
names = model['names']
orders = model['order_counts']
reverse_labels = {code: f"{names[code]}  [{code}]" for code in model['catalog']}

st.info('Educational demo based on historical UK retail transactions. Results are associations, not predictions of a particular shopper’s preferences.')
tab1, tab2, tab3 = st.tabs(['Recommendations', 'Retail insights', 'Evaluation'])

with tab1:
    st.subheader('Find similar products')
    st.write('Choose an item. The system finds products historically bought in the same orders.')
    codes = sorted(model['catalog'], key=lambda code: (-orders[code], names[code]))
    selected = st.selectbox('Search by product name or stock code', codes,
                            format_func=lambda code: reverse_labels[code])
    method_label = st.radio('Recommendation method',
                            ['Co-purchase similarity', 'Frequently bought together', 'Most popular'],
                            horizontal=True)
    methods = {
        'Co-purchase similarity': 'similarity',
        'Frequently bought together': 'frequently_bought',
        'Most popular': 'popularity',
    }
    chosen_method = methods[method_label]
    items = recommend(model, selected, 8, chosen_method)
    st.caption(f"Selected product appeared in {orders[selected]:,} training orders")
    if items:
        table = pd.DataFrame([
            {'Product': names[item['id']], 'Stock code': item['id'],
             'Orders': orders[item['id']],
             'Joint orders': item.get('joint_orders', '—'),
             'Similarity': item.get('score', '—')}
            for item in items
        ])
        st.dataframe(table, width="stretch", hide_index=True)
    else:
        st.warning('No recommendations available for this item.')
    st.caption('Similarity uses cosine similarity of item occurrence across past invoices. '
               'Frequently bought together ranks by co-occurrence count. '
               'Popular items are used as fallback.')

with tab2:
    st.subheader('Training data overview')
    stats = model['stats']
    a, b, c = st.columns(3)
    a.metric('Clean transaction lines', f"{stats['training_line_items']:,}")
    b.metric('Training orders', f"{stats['train_orders']:,}")
    c.metric('Product catalog', f"{stats['catalog_products']:,}")
    top = model['popular'][:12]
    df_plot = pd.DataFrame({
        'Product': [names[item['id']][:38] for item in top],
        'Orders': [item['orders'] for item in top],
    })
    fig = px.bar(df_plot.iloc[::-1], x='Orders', y='Product', orientation='h',
                 title='Most frequently purchased products (training orders)')
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=55, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption('Wholesale transactions and historical retail conditions influence these rankings.')

with tab3:
    st.subheader('Offline recommendation evaluation')
    st.write('Train on the earliest 80% of invoices; evaluate on later invoices. '
             'One product in a test invoice is the query, and the remaining products are relevant recommendations.')
    rows = []
    for name, result in evaluation['results'].items():
        rows.append({'Method': name.replace('_', ' ').title(),
                     'HitRate@5': result['hit_rate_at_5'],
                     'Recall@5': result['recall_at_5']})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Evaluated test invoices: {evaluation['evaluated_test_invoices']:,}. " + evaluation['notes'])
    st.write('**HitRate@5:** share of evaluated invoices with at least one relevant recommendation in the top 5.')
    st.write('**Recall@5:** average fraction of other eligible items recovered from the invoice.')
    st.warning('Offline co-purchase tests do not demonstrate uplift in sales or user satisfaction.')
