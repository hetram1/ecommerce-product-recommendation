"""Retail Compass: a focused Streamlit client of the shared recommendation service."""
from html import escape
import json

import streamlit as st

from src.recommender import METHODS
from src.service import CatalogService
from src.storage import ArtifactError, EVALUATION_PATH, MODEL_PATH, read_evaluation

st.set_page_config(page_title="Retail Compass | Product recommendations",
                   page_icon="🛍️", layout="wide")

LABELS = {
    "similarity": "Cosine similarity",
    "frequently_bought": "Frequently bought together",
    "popularity": "Most popular",
}
st.html("""
<style>
.block-container {max-width:1200px;padding-top:2.5rem;padding-bottom:3rem}
h1 {font-size:2.4rem!important;letter-spacing:-.065rem;padding-bottom:.3rem!important}
h2,h3 {letter-spacing:-.025rem}
[data-testid="stMetricValue"] {font-size:1.65rem}
.eyebrow {font-size:.73rem;font-weight:700;letter-spacing:.12em;color:#0f766e;margin:0 0 .6rem}
.product-title {font-size:1.02rem;line-height:1.45;font-weight:600;min-height:3rem;margin:.45rem 0 .6rem}
.card-reason {font-size:.83rem;color:#50616c;line-height:1.5;min-height:3.9rem;margin:.7rem 0}
.card-code {font-size:.72rem;color:#61717c;letter-spacing:.04em}
.card-price {font-size:1.12rem;font-weight:600;margin:0}
.card-orders {font-size:.76rem;color:#61717c}
hr {margin:.85rem 0!important}
@media(max-width:700px) {
 .block-container {padding-top:1.5rem}
 h1 {font-size:1.9rem!important}
 .product-title,.card-reason {min-height:0}
}
</style>
""")


def file_stamp(path):
    try:
        info = path.stat()
        return info.st_mtime_ns, info.st_size
    except OSError:
        return None


@st.cache_resource(show_spinner="Loading product catalog…")
def load_catalog(stamp):
    return CatalogService.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_report(stamp, build_id):
    return read_evaluation({"build_id": build_id}, EVALUATION_PATH)


try:
    service = load_catalog(file_stamp(MODEL_PATH))
except ArtifactError as exc:
    st.title("Product recommendations")
    st.error(str(exc))
    st.code("python -m src.build", language="bash")
    st.stop()

model = service.model
catalog = model["catalog"]
if st.session_state.get("catalog_build_id") != model["build_id"]:
    st.session_state.basket = [
        code for code in st.session_state.get("basket", []) if code in model["order_counts"]
    ]
    st.session_state.catalog_build_id = model["build_id"]
st.session_state.setdefault("basket", [])


def add_item(code):
    if code not in st.session_state.basket and len(st.session_state.basket) < 50:
        st.session_state.basket = [*st.session_state.basket, code]


def remove_item(code):
    st.session_state.basket = [item for item in st.session_state.basket if item != code]


def clear_basket():
    st.session_state.basket = []


def product_label(code):
    return f"{model['names'][code].title()} · {code}"


def render_cards(items, context):
    if not items:
        st.info("Every available product is already in your basket.")
        return
    for index, item in enumerate(items):
        # Create a new row for each pair so mobile stacking preserves ranking order.
        if index % 2 == 0:
            columns = st.columns(2, gap="medium")
        with columns[index % 2]:
            with st.container(border=True):
                price = (f"£{item['price_gbp']:.2f}" if item["price_gbp"] is not None
                         else "Price unavailable")
                st.html(
                    f'<div class="card-code">{escape(item["id"])}'
                    f'{" · POPULAR FALLBACK" if item["is_fallback"] else ""}</div>'
                    f'<div class="product-title">{escape(item["name"].title())}</div>'
                    f'<div class="card-price">{price}</div>'
                    f'<div class="card-orders">{item["orders"]:,} training baskets</div>'
                    f'<div class="card-reason">{escape(item["reason"])}</div>'
                )
                added = item["id"] in st.session_state.basket
                st.button(
                    "In your basket" if added else "Add to basket",
                    key=f"{context}_add_{item['id']}", width="stretch",
                    disabled=added or len(st.session_state.basket) >= 50,
                    on_click=add_item, args=(item["id"],),
                )
    st.caption("Prices are median historical unit prices in GBP, calculated from training data.")


with st.sidebar:
    st.markdown("### Retail Compass")
    st.caption("Purchase patterns. Practical suggestions.")
    st.divider()
    method = st.selectbox("Recommendation method", list(METHODS),
                          format_func=lambda name: LABELS[name])
    k = st.select_slider("Suggestions", options=[4, 6, 8, 10, 12], value=6)
    st.divider()
    st.metric("Products in your basket", len(st.session_state.basket))
    st.caption("Up to 50 distinct products. Quantities do not change co-purchase recommendations.")
    st.button("Clear basket", on_click=clear_basket,
              disabled=not st.session_state.basket, width="stretch")
    st.divider()
    st.link_button("Explore the REST API ↗", "http://localhost:8000/docs", width="stretch")
    st.caption("Historical UCI Online Retail data · UK · 2010–2011")

st.html('<p class="eyebrow">RETAIL COMPASS / PRODUCT INTELLIGENCE</p>')
st.title("Discover what goes together.")
st.caption("Explore real purchase patterns and build a basket with explainable recommendations.")
page = st.radio("Workspace", ["Discover", "Basket", "Analytics"],
                horizontal=True, label_visibility="collapsed", key="workspace")
st.divider()

if page == "Discover":
    st.subheader("Find your starting product")
    query = st.text_input("Search products", placeholder="Search a product name or stock code",
                          max_chars=100, key="search")
    matches = service.search(query, limit=100)
    if not matches["items"]:
        st.info("No matching products. Try a shorter name or a stock code.")
    else:
        selected = st.selectbox(
            f"Choose a product · {matches['total']:,} matches",
            [item["id"] for item in matches["items"]], format_func=product_label,
            key="selected_product",
        )
        product = service.product(selected)
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.markdown("**" + escape(product["name"].title()) + "**")
                st.caption(f"Stock code {selected} · {product['orders']:,} training baskets")
            with right:
                st.button(
                    "Add selected product", key="add_selected", type="primary", width="stretch",
                    disabled=selected in st.session_state.basket or len(st.session_state.basket) >= 50,
                    on_click=add_item, args=(selected,),
                )
        st.subheader("Recommended for this product")
        st.caption(f"{LABELS[method]} · products already in your basket are excluded")
        response = service.recommendations([selected], k, method, st.session_state.basket)
        render_cards(response["recommendations"], "discover")
elif page == "Basket":
    count = len(st.session_state.basket)
    st.subheader(f"Your basket · {count} product{'s' if count != 1 else ''}")
    if not count:
        st.info("Your basket is empty. Add a product below or find one in Discover.")
    for code in st.session_state.basket:
        product = service.product(code)
        with st.container(border=True):
            left, right = st.columns([5, 1])
            with left:
                st.text(product["name"].title())
                st.caption(f"{code} · {product['orders']:,} training baskets")
            with right:
                st.button("Remove", key=f"remove_{code}", on_click=remove_item,
                          args=(code,), width="stretch")
    st.subheader("Complete your basket" if count else "Popular starting points")
    st.caption("Scores combine matches across your entire basket. Existing products are excluded."
               if count else "Popular products from the training catalog.")
    response = service.recommendations(st.session_state.basket, k, method)
    render_cards(response["recommendations"], "basket")
else:
    st.subheader("Retail insights")
    stats = model.get("stats", {})
    a, b, c = st.columns(3)
    a.metric("Catalog products", f"{len(catalog):,}")
    b.metric("Modeled training baskets", f"{model['train_baskets']:,}")
    c.metric("Clean training lines", f"{stats.get('training_line_items', 0):,}")
    st.caption(f"{stats.get('train_orders', 0):,} earlier invoices in the training partition; "
               "only baskets with 2–40 distinct products are modeled.")
    # Import chart/data libraries only on the analytics page.
    import pandas as pd
    import plotly.graph_objects as go

    top = model["popular"][:10][::-1]
    fig = go.Figure(go.Bar(
        x=[item["orders"] for item in top],
        y=[model["names"][item["id"]].title() for item in top],
        orientation="h", marker_color="#0f766e",
        hovertemplate="%{y}<br>%{x:,} training baskets<extra></extra>",
    ))
    fig.update_layout(
        title="Most purchased products", height=420,
        margin=dict(l=0, r=15, t=55, b=0), xaxis_title="Training baskets",
        yaxis_title=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#172b3a"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.subheader("Verified offline evaluation")
    try:
        report = load_report(file_stamp(EVALUATION_PATH), model["build_id"])
    except ArtifactError as exc:
        st.warning(str(exc))
    else:
        left, right = st.columns([2, 1])
        with left:
            task = st.radio("Evaluation task", ["Single product", "Basket"], horizontal=True)
        with right:
            cutoff = st.selectbox("Recommendation cutoff", [5, 10], format_func=lambda v: f"Top {v}")
        key = "results" if task == "Single product" else "basket_results"
        rows = []
        for method_name, values in report[key].items():
            rows.append({
                "Method": LABELS[method_name],
                "HitRate": values[f"hit_rate_at_{cutoff}"],
                "Recall": values[f"recall_at_{cutoff}"],
                "Catalog coverage": values[f"coverage_at_{cutoff}"],
            })
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={column: st.column_config.NumberColumn(format="percent")
                           for column in ("HitRate", "Recall", "Catalog coverage")},
        )
        n = report["evaluated_test_invoices" if task == "Single product" else "evaluated_basket_invoices"]
        st.caption(f"{n:,} held-out invoices · cutoff {report['split']['cutoff']} · "
                   f"build {report['build_id']}")
        with st.expander("How to read these results"):
            st.write("HitRate: invoices with at least one relevant result. "
                     "Recall: average fraction of hidden eligible products retrieved. "
                     "Catalog coverage: share of training products recommended across these queries.")
            st.write(report["single_product_task"] if task == "Single product" else report["basket_task"])
            st.write(report["notes"])
            fraction = report["test_catalog_item_fraction"]
            if fraction is None:
                st.write("No eligible test baskets were available; evaluation values are unmeasured.")
            else:
                st.write(f"Training-catalog products represent {fraction:.1%} "
                         "of item occurrences in size-filtered test baskets.")
        st.download_button("Download evaluation report", json.dumps(report, indent=2),
                           file_name="evaluation.json", mime="application/json")

st.divider()
st.caption("Historical associations, not individual purchase predictions. Offline accuracy does not establish sales uplift.")
