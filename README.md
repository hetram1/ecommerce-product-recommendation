# E-Commerce Product Recommendation System

A lightweight item-based recommendation system built with real historical purchases from a UK online retailer. Select a product to see related products, frequently bought-together items, and popular alternatives.

## Features

- Product search and recommendations using item co-purchase cosine similarity.
- Frequently-bought-together and popularity-based recommendations.
- Streamlit dashboard with product-frequency insights.
- A chronological offline evaluation against a popularity baseline (HitRate@5 and Recall@5).

## Dataset

[UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), Chen (2015), [DOI 10.24432/C5BW33](https://doi.org/10.24432/C5BW33), licensed under CC BY 4.0. The source contains approximately 541,909 historical transaction lines. It includes many wholesale orders and is **not** an explicit product-rating dataset.

Cancellations, invalid quantity/price rows, missing descriptions, and non-product stock codes are excluded. The model uses basket-level product presence rather than ratings. Only training transactions are used to compute popularity and similarity.

## Installation

Requires Python 3.11+ (Python 3.12 recommended). **Windows quick start:** run `setup.bat` from Command Prompt or PowerShell, then `run.bat` to open the dashboard. The setup script installs dependencies, downloads/prepares the dataset and runs tests.

For manual setup:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.build
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The build command downloads and extracts the official UCI dataset automatically if `data/Online Retail.xlsx` is absent. If auto-download fails, download it yourself from the dataset link and place it under `data/Online Retail.xlsx`, then rerun the build. Initial Excel processing can take a few minutes.

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux or macOS, replace `py` with `python3` and use `.venv/bin/python`.

## Method

Transactions are split chronologically by invoice (80% earlier invoices for training, 20% later invoices for testing). The recommendation catalog uses up to 1,200 commonly purchased products from training data, appearing in at least five selected training baskets; only invoices with 2–40 distinct products are modeled. Invoice-product membership is represented as a sparse matrix. Cosine similarity of product vectors ranks neighbors. Co-purchase count provides a simpler alternative, and popularity provides a cold-start fallback.

The offline task reveals one product from a later invoice and tests whether the model retrieves other products from that same invoice. Both methods use identical seed products. Only products in the training catalog are eligible for evaluation. The split is reproducible, but invoices may include repeat customers, and offline co-purchase accuracy does not establish increased sales.

After running `python -m src.build`, evaluation metrics appear in `reports/evaluation.json` and the dashboard. No unmeasured model scores or business impact are claimed.

## Files

- `app.py` — Streamlit app
- `src/data.py` — source download and cleaning
- `src/recommender.py` — recommendation algorithm
- `src/build.py` — chronological splitting, fitting, and evaluation
- `tests/` — basic checks
- `artifacts/recommender.json` — generated recommender; created by build
- `reports/evaluation.json` — generated holdout results; created by build

The raw Excel file is intentionally not committed. Generated JSON artifacts may be committed after building if you want the hosted app to run without downloading data.
