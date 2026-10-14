"""Clean real transactions, fit recommendations, and evaluate on later invoices."""
import argparse
import json
from pathlib import Path

from src.data import get_dataset, load_and_clean
from src.recommender import invoice_baskets, train_recommender, recommend

ROOT = Path(__file__).resolve().parents[1]


def main(data_path: Path, max_products: int = 1200):
    path = get_dataset(data_path)
    print('Reading and cleaning data (the Excel import can take a minute)...')
    df, data_stats = load_and_clean(path)
    if df.empty:
        raise ValueError('No usable transactions remain after cleaning')

    # Split on invoice timestamp, never individual line items. No future orders enter fitting.
    invoices = df.groupby('InvoiceNo', sort=False)['InvoiceDate'].min().sort_values()
    split_idx = int(len(invoices) * 0.8)
    if split_idx < 5 or split_idx >= len(invoices):
        raise ValueError('Dataset needs enough invoices for an 80/20 chronological split')
    train_ids = set(invoices.index[:split_idx])
    test_ids = set(invoices.index[split_idx:])
    train = df[df['InvoiceNo'].isin(train_ids)]
    test = df[df['InvoiceNo'].isin(test_ids)]
    names = (train.groupby('StockCode')['Description'].agg(lambda s: s.value_counts().index[0]).to_dict())
    train_baskets = invoice_baskets(train)
    test_baskets = invoice_baskets(test)
    model = train_recommender(train_baskets, names, max_products=max_products)

    # Evaluate both methods on identical held-out invoices and identical seed products.
    import numpy as np
    rng = np.random.default_rng(42)
    catalog = set(model['catalog'])
    eligible = [sorted(catalog.intersection(b)) for b in test_baskets]
    eligible = [b for b in eligible if len(b) >= 2]
    if len(eligible) > 2500:
        idx = rng.choice(len(eligible), 2500, replace=False)
        eligible = [eligible[i] for i in idx]
    seeds = [(basket[int(rng.integers(len(basket)))], set(basket)) for basket in eligible]
    metrics = {}
    for method in ('popularity', 'similarity'):
        hit_total, recall_total = 0, 0.0
        for pivot, basket in seeds:
            truth = basket - {pivot}
            recommended = {r['id'] for r in recommend(model, pivot, 5, method)}
            count = len(truth & recommended)
            hit_total += bool(count)
            recall_total += count / len(truth)
        metrics[method] = {
            'hit_rate_at_5': round(hit_total / len(seeds), 4) if seeds else None,
            'recall_at_5': round(recall_total / len(seeds), 4) if seeds else None,
        }

    model['stats'] = {
        **data_stats,
        'total_orders': len(invoices),
        'train_orders': len(train_ids),
        'test_orders': len(test_ids),
        'catalog_products': len(model['catalog']),
        'training_revenue_gbp': round(float(train['Revenue'].sum()), 2),
        'training_line_items': len(train),
    }
    evaluation = {
        'split': 'Chronological 80/20 invoice-level split',
        'task': 'Given one item from a later invoice, retrieve other items from that invoice',
        'evaluated_test_invoices': len(seeds),
        'top_k': 5,
        'notes': 'Only products seen in the training catalog are evaluated. '
                 'The source includes wholesale orders. This is an offline proxy, not an online A/B test.',
        'results': metrics,
    }
    artifacts = ROOT / 'artifacts'
    reports = ROOT / 'reports'
    artifacts.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    (artifacts / 'recommender.json').write_text(json.dumps(model, indent=1), encoding='utf-8')
    (reports / 'evaluation.json').write_text(json.dumps(evaluation, indent=2), encoding='utf-8')
    print(json.dumps(evaluation, indent=2))
    print('\nSuccess! Run: python -m streamlit run app.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'Online Retail.xlsx')
    parser.add_argument('--max-products', type=int, default=1200)
    args = parser.parse_args()
    main(args.data, args.max_products)
