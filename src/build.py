"""Reproducible offline training and evaluation. Run: python -m src.build."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

from src.data import get_dataset, load_and_clean
from src.evaluation import chronological_split, evaluate
from src.recommender import invoice_baskets, train_recommender
from src.storage import ROOT, write_json_atomic


def main(data_path: Path, max_products: int = 1200, output_dir: Path = ROOT):
    started = time.perf_counter()
    path = get_dataset(data_path)
    print("Reading and cleaning transactions...", flush=True)
    df, data_stats = load_and_clean(path)
    if df.empty:
        raise ValueError("No usable transactions remain after cleaning")
    train, test, split = chronological_split(df)
    names = train.groupby("StockCode")["Description"].agg(
        lambda s: s.value_counts().sort_index().idxmax(),
    ).to_dict()
    baskets = invoice_baskets(train)
    print(f"Fitting on {len(baskets):,} training baskets...", flush=True)
    model = train_recommender(baskets, names, max_products=max_products)
    prices = train.groupby("StockCode")["UnitPrice"].median()
    model["prices_gbp"] = {item: round(float(prices[item]), 2) for item in model["catalog"]}
    model["stats"] = {
        **data_stats, "total_orders": df["InvoiceNo"].nunique(),
        "train_orders": split["train_invoices"], "test_orders": split["test_invoices"],
        "catalog_products": len(model["catalog"]),
        "training_revenue_gbp": round(float(train["Revenue"].sum()), 2),
        "training_line_items": len(train), "modeled_training_baskets": len(baskets),
    }
    model["config"].update({"min_basket_size": 2, "max_basket_size": 40})
    fingerprint = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    provenance = {
        "source_file": path.name, "source_sha256": fingerprint.hexdigest(),
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ("pandas", "numpy", "scipy", "openpyxl")},
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    build_id = sha256(json.dumps({
        "source": provenance["source_sha256"], "config": model["config"],
        "split": split, "schema_version": 2,
    }, sort_keys=True).encode()).hexdigest()[:16]
    model.update({"build_id": build_id, "provenance": provenance})
    print("Evaluating all eligible held-out invoices at k=5 and k=10...", flush=True)
    evaluation = evaluate(model, test)
    evaluation.update({
        "build_id": build_id, "schema_version": 2, "split": split,
        "provenance": provenance, "training_config": model["config"],
        "build_seconds": round(time.perf_counter() - started, 2),
    })
    # Files are individually atomic; build_id detects mixed pairs after interruption.
    write_json_atomic(output_dir / "artifacts" / "recommender.json", model)
    write_json_atomic(output_dir / "reports" / "evaluation.json", evaluation)
    print(json.dumps(evaluation, indent=2), flush=True)
    print("\nReady. UI: python -m streamlit run app.py\nAPI: python -m uvicorn src.api:app --port 8000")
    return model, evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "Online Retail.xlsx")
    parser.add_argument("--max-products", type=int, default=1200)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    main(args.data, args.max_products, args.output_dir)
