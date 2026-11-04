"""Sparse training and deterministic item-to-item inference.

Serving uses only the standard library; training dependencies are imported lazily.
"""
from collections import Counter, defaultdict
from heapq import nsmallest
from math import sqrt

METHODS = ("similarity", "frequently_bought", "popularity")
MAX_RESULTS = 50


def invoice_baskets(df, min_size: int = 2, max_size: int = 40) -> list[list[str]]:
    """One binary basket per invoice; duplicate lines and quantity add no weight."""
    if not 1 <= min_size <= max_size:
        raise ValueError("Basket bounds must satisfy 1 <= min_size <= max_size")
    baskets = df.groupby("InvoiceNo", sort=False)["StockCode"].agg(lambda s: sorted(set(s)))
    return [basket for basket in baskets if min_size <= len(basket) <= max_size]


def train_recommender(
    baskets: list[list[str]], names: dict, max_products: int = 1200,
    min_orders: int = 5, top_k: int = MAX_RESULTS, min_pair_orders: int = 2,
) -> dict:
    """Compute co-occurrences once and cosine-normalize only nonzero entries."""
    import numpy as np
    from scipy.sparse import csr_matrix

    if min(max_products, min_orders, top_k, min_pair_orders) < 1:
        raise ValueError("Training parameters must be positive integers")
    baskets = [sorted(set(basket)) for basket in baskets]
    counts = Counter(item for basket in baskets for item in basket)
    catalog = sorted(counts, key=lambda item: (-counts[item], item))[:max_products]
    catalog = [item for item in catalog if counts[item] >= min_orders]
    if not catalog:
        raise ValueError("No products meet the minimum purchase frequency")
    index = {item: i for i, item in enumerate(catalog)}
    rows, cols = [], []
    for row, basket in enumerate(baskets):
        for item in basket:
            if item in index:
                rows.append(row)
                cols.append(index[item])
    matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(len(baskets), len(catalog)),
    )
    co = (matrix.T @ matrix).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    recommendations, bought_together = {}, {}
    for i, item in enumerate(catalog):
        start, end = co.indptr[i:i + 2]
        candidates = []
        for j, joint in zip(co.indices[start:end], co.data[start:end]):
            if joint >= min_pair_orders:
                other = catalog[j]
                candidates.append({
                    "id": other,
                    "score": float(joint / sqrt(counts[item] * counts[other])),
                    "joint_orders": int(joint),
                })
        recommendations[item] = nsmallest(
            top_k, candidates,
            key=lambda c: (-c["score"], -c["joint_orders"], c["id"]),
        )
        frequent = nsmallest(
            top_k, candidates, key=lambda c: (-c["joint_orders"], c["id"]),
        )
        bought_together[item] = [
            {"id": c["id"], "joint_orders": c["joint_orders"]} for c in frequent
        ]
    return {
        "schema_version": 2,
        "catalog": catalog,
        "names": {item: names.get(item, item) for item in catalog},
        # Full catalog fills fallback requests even after many products are excluded.
        "popular": [{"id": item, "orders": counts[item]} for item in catalog],
        "order_counts": {item: counts[item] for item in catalog},
        "recommendations": recommendations,
        "bought_together": bought_together,
        "train_baskets": len(baskets),
        "config": {
            "max_products": max_products, "min_orders": min_orders,
            "neighbors_per_product": top_k, "min_pair_orders": min_pair_orders,
        },
    }


def recommend_basket(
    model: dict, products: list[str], k: int = 5, method: str = "similarity",
    exclude: list[str] | tuple[str, ...] = (),
) -> list[dict]:
    """Average neighbor scores across distinct known seeds, then fill shortages.

    Cosine uses average similarity. Co-purchase uses average P(candidate | seed),
    so a popular seed cannot dominate just because it has more orders.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method. Choose one of: {', '.join(METHODS)}")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_RESULTS:
        raise ValueError(f"k must be an integer between 1 and {MAX_RESULTS}")
    seeds = sorted(set(products))
    excluded = set(seeds).union(exclude)
    known = [seed for seed in seeds if seed in model["order_counts"]]
    scores, support, matched = defaultdict(float), defaultdict(int), defaultdict(list)
    if method != "popularity":
        neighbors = model["recommendations" if method == "similarity" else "bought_together"]
        for seed in known:
            for candidate in neighbors.get(seed, []):
                item = candidate["id"]
                if item in excluded:
                    continue
                score = (candidate["score"] if method == "similarity" else
                         candidate["joint_orders"] / model["order_counts"][seed])
                scores[item] += score / len(known)
                support[item] += candidate["joint_orders"]
                matched[item].append(seed)
    ranked = nsmallest(
        k, scores,
        key=lambda item: (-scores[item], -len(matched[item]), -support[item], item),
    )
    results = [{
        "id": item, "score": round(scores[item], 8), "source": method,
        "matched_products": matched[item],
        "joint_orders": support[item] if len(known) == 1 else None,
    } for item in ranked]
    existing = excluded.union(ranked)
    for item in model["popular"]:
        if len(results) == k:
            break
        if item["id"] not in existing:
            results.append({
                "id": item["id"],
                "score": round(item["orders"] / model["train_baskets"], 8),
                "source": "popularity", "matched_products": [], "joint_orders": None,
            })
            existing.add(item["id"])
    return results


def recommend(
    model: dict, product: str, k: int = 5, method: str = "similarity",
    exclude: list[str] | tuple[str, ...] = (),
) -> list[dict]:
    """Single-product wrapper, retaining the original public function."""
    return recommend_basket(model, [product], k, method, exclude)
