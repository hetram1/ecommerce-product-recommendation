"""Leakage-resistant chronological evaluation with shared deterministic queries."""
import numpy as np

from src.recommender import METHODS, invoice_baskets, recommend_basket


def chronological_split(df, train_fraction: float = 0.8):
    """Keep timestamp ties together; drop invoices spanning the time boundary."""
    if not 0 < train_fraction < 1 or df.empty:
        raise ValueError("A nonempty dataset and 0 < train_fraction < 1 are required")
    times = df.groupby("InvoiceNo")["InvoiceDate"].agg(["min", "max"])
    ordered = times.sort_values(["min"], kind="stable")
    split_idx = int(len(ordered) * train_fraction)
    if not 1 <= split_idx < len(ordered):
        raise ValueError("Not enough invoices for a chronological split")
    cutoff = ordered.iloc[split_idx]["min"]
    train_ids = times.index[times["max"] < cutoff]
    test_ids = times.index[times["min"] >= cutoff]
    train = df[df["InvoiceNo"].isin(train_ids)].copy()
    test = df[df["InvoiceNo"].isin(test_ids)].copy()
    if train.empty or test.empty:
        raise ValueError("Time boundary leaves an empty train or test partition")
    assert train["InvoiceDate"].max() < test["InvoiceDate"].min()
    return train, test, {
        "strategy": "Chronological invoice split; timestamp ties stay together",
        "target_train_fraction": train_fraction, "cutoff": cutoff.isoformat(),
        "train_start": train["InvoiceDate"].min().isoformat(),
        "train_end": train["InvoiceDate"].max().isoformat(),
        "test_start": test["InvoiceDate"].min().isoformat(),
        "test_end": test["InvoiceDate"].max().isoformat(),
        "train_invoices": len(train_ids), "test_invoices": len(test_ids),
        "spanning_invoices_dropped": len(times) - len(train_ids) - len(test_ids),
    }


def evaluate_queries(model: dict, queries: list[tuple[list[str], set[str]]]) -> dict:
    """Every algorithm receives exactly the same seeds and relevant items."""
    results = {}
    for method in METHODS:
        totals = {k: {"hits": 0, "recall": 0.0, "recommended": set()} for k in (5, 10)}
        for seeds, truth in queries:
            if not truth or set(seeds).intersection(truth):
                raise ValueError("Evaluation requires nonempty, hidden relevant products")
            ranked = recommend_basket(model, seeds, 10, method)
            for k, total in totals.items():
                retrieved = {item["id"] for item in ranked[:k]}
                hits = len(retrieved.intersection(truth))
                total["hits"] += bool(hits)
                total["recall"] += hits / len(truth)
                total["recommended"].update(retrieved)
        metrics = {}
        for k, total in totals.items():
            metrics.update({
                f"hit_rate_at_{k}": round(total["hits"] / len(queries), 6) if queries else None,
                f"recall_at_{k}": round(total["recall"] / len(queries), 6) if queries else None,
                f"coverage_at_{k}": round(len(total["recommended"]) / len(model["catalog"]), 6)
                if queries else None,
            })
        results[method] = metrics
    return results


def evaluate(model: dict, test, seed: int = 42) -> dict:
    """All eligible later invoices; a separate task reveals 2–3 basket products."""
    baskets = invoice_baskets(test)
    catalog = set(model["catalog"])
    rng = np.random.default_rng(seed)
    single_queries, basket_queries = [], []
    eligible_occurrences = 0
    for basket in baskets:
        eligible = sorted(catalog.intersection(basket))
        eligible_occurrences += len(eligible)
        if len(eligible) < 2:
            continue
        permuted = list(rng.permutation(eligible))
        single_queries.append(([permuted[0]], set(eligible) - {permuted[0]}))
        if len(eligible) >= 3:
            revealed = min(3, max(2, len(eligible) // 2))
            seeds = permuted[:revealed]
            basket_queries.append((seeds, set(eligible) - set(seeds)))
    all_occurrences = sum(map(len, baskets))
    return {
        "random_seed": seed, "cutoffs": [5, 10],
        "size_filtered_test_baskets": len(baskets),
        "evaluated_test_invoices": len(single_queries),
        "evaluated_basket_invoices": len(basket_queries),
        "test_catalog_item_fraction": round(eligible_occurrences / all_occurrences, 6)
        if all_occurrences else None,
        "single_product_task": "Reveal one product; retrieve other eligible items from the invoice",
        "basket_task": "Reveal 2–3 products; retrieve remaining eligible items (at least one hidden)",
        "results": evaluate_queries(model, single_queries),
        "basket_results": evaluate_queries(model, basket_queries),
        "notes": (
            "All eligible later invoices are evaluated; all methods use identical queries. "
            "Only training-catalog products count as relevant. Baskets contain 2–40 distinct products. "
            "Coverage is the fraction of the training catalog recommended across evaluated queries. "
            "The source includes wholesale orders and repeat customers. This offline co-purchase "
            "task does not measure personalized preferences, sales uplift, or online satisfaction."
        ),
    }
