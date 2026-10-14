"""Item-to-item recommendations from historical co-purchase baskets."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def invoice_baskets(df: pd.DataFrame, min_size: int = 2, max_size: int = 40) -> list[list[str]]:
    baskets = df.groupby('InvoiceNo', sort=False)['StockCode'].agg(lambda s: sorted(set(s)))
    return [basket for basket in baskets if min_size <= len(basket) <= max_size]


def train_recommender(baskets: list[list[str]], names: dict, max_products: int = 1200,
                      min_orders: int = 5, top_k: int = 15) -> dict:
    """Compute cosine similarity on binary invoice-product occurrences, not ratings."""
    from collections import Counter
    counts = Counter(item for basket in baskets for item in basket)
    catalog = [item for item, count in counts.most_common(max_products) if count >= min_orders]
    if not catalog:
        raise ValueError('No products meet the minimum purchase frequency')
    index = {item: i for i, item in enumerate(catalog)}
    rows, cols = [], []
    for invoice_idx, basket in enumerate(baskets):
        for item in basket:
            if item in index:
                rows.append(invoice_idx)
                cols.append(index[item])
    matrix = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                        shape=(len(baskets), len(catalog)), dtype=np.float32)
    freq = np.asarray(matrix.sum(axis=0)).ravel()
    co = (matrix.T @ matrix).toarray()
    np.fill_diagonal(co, 0)
    similarity = cosine_similarity(matrix.T, dense_output=True)
    np.fill_diagonal(similarity, 0)
    recommendations = {}
    bought_together = {}
    for i, item in enumerate(catalog):
        ranked = sorted((j for j in range(len(catalog)) if j != i and co[i, j] > 0),
                        key=lambda j: (-similarity[i, j], -co[i, j], catalog[j]))[:top_k]
        recommendations[item] = [
            {'id': catalog[j], 'score': round(float(similarity[i, j]), 5),
             'joint_orders': int(co[i, j])} for j in ranked
        ]
        frequent = sorted((j for j in range(len(catalog)) if j != i and co[i, j] > 0),
                          key=lambda j: (-co[i, j], catalog[j]))[:top_k]
        bought_together[item] = [
            {'id': catalog[j], 'joint_orders': int(co[i, j])} for j in frequent
        ]
    return {
        'catalog': catalog,
        'names': {item: names.get(item, item) for item in catalog},
        'popular': [{'id': item, 'orders': counts[item]} for item in catalog[:top_k]],
        'order_counts': {item: counts[item] for item in catalog},
        'recommendations': recommendations,
        'bought_together': bought_together,
        'train_baskets': len(baskets),
    }


def recommend(model: dict, product: str, k: int = 5, method: str = 'similarity') -> list[dict]:
    """Use a popular-products fallback if an item has no useful neighbors."""
    if method == 'frequently_bought':
        suggestions = model['bought_together'].get(product, [])
    elif method == 'popularity':
        suggestions = []
    else:
        suggestions = model['recommendations'].get(product, [])
    items = list(suggestions[:k])
    existing = {product, *(item['id'] for item in items)}
    for item in model['popular']:
        if len(items) >= k:
            break
        if item['id'] not in existing:
            items.append({'id': item['id'], 'orders': item['orders']})
            existing.add(item['id'])
    return items[:k]

