# Interview walkthrough

## A short introduction

“I built a product recommendation service using real retail invoices. It cleans
transactions, trains an item-to-item model, exposes validated FastAPI endpoints,
and provides a Streamlit interface for product discovery and basket suggestions.
I compared cosine similarity, co-purchase ranking, and popularity on later invoices
using HitRate, recall, and catalog coverage. The results show a tradeoff between
retrieval accuracy and product exposure.”

Use the measured values in `reports/evaluation.json` when practicing this walkthrough.

## A five-minute demo

1. Open Discover, search for “heart,” and select a product.
2. Explain one recommendation using its joint-basket count and cosine score.
3. Add the selected product and a suggestion. Open Basket and show how results
   change with both products. Remove a product and clear the basket.
4. Switch between cosine, co-purchase, and popularity. Describe the different
   objectives; there is no claim that one algorithm always wins.
5. Open Analytics. Compare top 5 and top 10 for single-product and basket queries.
6. Open the API docs at `http://localhost:8000/docs`. Submit a basket, then an
   unknown code to demonstrate a clear 404. Show a 422 for an invalid `k`.
7. Run `python -m pytest -q` and show the time-split and hand-calculated metric tests.

## Decisions worth explaining

**Why item-to-item recommendations?** The source records purchases rather than
ratings. Invoice membership supplies an explainable signal and works without a
customer profile. It does not identify a particular shopper’s preferences.

**How does cosine work?** Represent each product by a binary vector over training
invoices. For products A and B, cosine is `joint(A,B) / sqrt(orders(A) * orders(B))`.
Normalization reduces the automatic advantage of very popular products. A pair
needs at least two joint training invoices; that threshold was fixed before the
reported evaluation.

**Why a sparse matrix?** Most product pairs never share an invoice. One sparse
`X.T @ X` gives co-purchase counts. Normalizing only nonzero pairs avoids a second
matrix multiplication and avoids materializing dense product matrices. Only 50
neighbors per product per method are stored for serving.

**How do basket recommendations work?** Form the union of the basket products’
stored neighbors and average each candidate’s score across all distinct seeds.
For co-purchase ranking, divide joint counts by each seed’s order count before
averaging. A seed with many orders therefore cannot dominate just through raw
counts. Exclude all basket products, deduplicate candidates, apply deterministic
tie breaks, and use popularity only to fill missing slots.

**How is leakage prevented?** Choose the boundary by invoice start time. Keep
timestamp ties on the test side and exclude invoices that span the boundary.
Every training line precedes every test line. Product selection, names, prices,
popularity, and similarities all use training data. Each comparison uses the same
revealed and hidden items. Tests inject future names, prices, and products to
check that none enters the fitted model.

**Why three metrics?** HitRate measures whether there was any useful result.
Recall measures how many hidden eligible products were recovered. Coverage
measures how much of the catalog was exposed across evaluation queries. A global
popular list can get hits while exposing a very small part of the catalog.

**Why share a service between Streamlit and FastAPI?** Both use identical search,
ranking, validation, and explanation code. Streamlit can run independently for a
simple local demo; REST clients use FastAPI. HTTP transport is not required for
every local UI click. The API loads the artifact once and caches repeated queries.

## Results and limits to discuss honestly

- The README contains the exact measured comparison for this build. Co-purchase
  had the higher single-product HitRate; cosine had higher basket HitRate and
  broader coverage. Do not claim universal superiority.
- Evaluation uses only eligible products already in the training catalog.
  Report the retained item fraction alongside the scores.
- Invoices include wholesale orders and repeat customers. This is a chronological
  co-purchase task, not a customer-disjoint personalization study.
- Offline retrieval scores do not prove higher sales, conversion, or satisfaction.
- New products have no collaborative signal. Popularity is a transparent fallback;
  content features would be a reasonable next step if real metadata becomes available.
- Candidate pooling uses stored top-50 neighbors, so basket ranking is an efficient
  approximation to searching every product pair. Larger catalogs would need measured
  memory/latency profiling and a revised candidate strategy.
- No cart checkout, payment system, authentication, or database is needed for this
  project’s recommendation demonstration. Browser basket state lasts for the session.
