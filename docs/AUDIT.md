# Audit and changes

The original project already had a useful purchase-based model, a Streamlit UI,
an invoice-level chronological split, and three basic regression tests. These
features were retained and extended.

| Finding | Change | Verification |
| --- | --- | --- |
| Co-occurrences and cosine were calculated separately and converted to dense arrays | One sparse product co-occurrence matrix; cosine computed from nonzero counts | Exact hand-calculated cosine and binary-membership tests |
| Popularity retained only 15 products | Full frequency-sorted catalog used for fallback | Large requests after exclusions fill correctly |
| Duplicate products could inflate training counts when the trainer was called directly | Deduplicate every training basket | Duplicate-membership regression test |
| Unknown methods silently used similarity; invalid result counts behaved inconsistently | Explicit method and result-count validation | Engine and HTTP validation tests |
| There was no multi-product basket ranking | Normalize, aggregate, deduplicate, exclude every seed, then fill shortages | Order invariance, seed dominance, exclusion, and UI interaction tests |
| A single co-occurrence could produce an unreliable association | Require at least two joint training invoices | Support-threshold test; all algorithms reevaluated with this fixed setting |
| Time ties could be divided between train and test; a multi-time invoice could span the cutoff | Keep equal timestamps together and drop boundary-spanning invoices | Adversarial time-boundary test |
| Evaluation covered only two algorithms at k=5 on a sample | Compare all three at k=5/10 on all eligible invoices; add basket task and coverage | Hand-worked metrics and reproducibility tests; rebuilt real-data report |
| Corrupt downloads could leave a partially cached spreadsheet | Extract into a temporary directory and atomically rename | Failed-download and selective-extraction tests |
| Training dependencies were imported while serving | Lazy scientific imports; shared lightweight service with bounded caches | Live API/UI checks; no model fitting on requests |
| Missing/mismatched artifacts could crash or show stale metrics | Validated model, atomic writes, matching build IDs, graceful UI/HTTP errors | Missing-model, invalid-model, and stale-report tests |
| No backend | Versioned FastAPI search, product, single, basket, and evaluation routes | TestClient tests and live requests |
| UI required table scanning and had no basket | Search, explainable cards, add/remove/clear basket, concise analytics | Streamlit interaction tests and browser verification |

The new evaluation uses all eligible test invoices and a deterministic revised
query-generation protocol. Its numbers should not be treated as a controlled
before/after comparison with the old sampled report. Current results are in
`reports/evaluation.json`; the README describes their denominators and limitations.
