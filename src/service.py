"""Shared catalog/search/recommendation service for HTTP and Streamlit clients."""
from functools import lru_cache
from math import isfinite
from pathlib import Path

from src.recommender import recommend_basket
from src.storage import ArtifactError, MODEL_PATH, read_json


class UnknownProducts(ValueError):
    def __init__(self, products):
        self.products = sorted(set(products))
        super().__init__(f"Unknown product codes: {', '.join(self.products)}")


class CatalogService:
    def __init__(self, model: dict):
        self._validate(model)
        self.model = model
        self._products = {
            code: {"id": code, "name": model["names"][code],
                   "orders": model["order_counts"][code],
                   "price_gbp": model.get("prices_gbp", {}).get(code)}
            for code in model["catalog"]
        }
        self._search_text = {
            code: f"{code} {product['name']}".casefold()
            for code, product in self._products.items()
        }
        self._find = lru_cache(maxsize=128)(self._find_ids)
        self._rank = lru_cache(maxsize=512)(self._rank_uncached)

    @staticmethod
    def _validate(model: dict) -> None:
        try:
            if (model["schema_version"] != 2 or not isinstance(model["build_id"], str)
                    or not model["build_id"]):
                raise ValueError("Unsupported model version")
            catalog = model["catalog"]
            if not isinstance(catalog, list) or not catalog or len(set(catalog)) != len(catalog):
                raise ValueError("Invalid catalog")
            if not isinstance(model["train_baskets"], int) or model["train_baskets"] < 1:
                raise ValueError("Invalid training basket count")
            codes = set(catalog)
            if [p["id"] for p in model["popular"]] != catalog:
                raise ValueError("Incomplete popularity fallback")
            for code in catalog:
                if not isinstance(code, str) or not code or not isinstance(model["names"][code], str):
                    raise ValueError("Invalid product")
                count = model["order_counts"][code]
                if not isinstance(count, int) or not 1 <= count <= model["train_baskets"]:
                    raise ValueError("Invalid purchase frequency")
                price = model.get("prices_gbp", {}).get(code)
                if price is not None and (not isfinite(price) or price <= 0):
                    raise ValueError("Invalid price")
                for key in ("recommendations", "bought_together"):
                    seen = set()
                    for neighbor in model[key][code]:
                        other = neighbor["id"]
                        if other not in codes or other == code or other in seen:
                            raise ValueError("Invalid neighbor")
                        seen.add(other)
                        if not 1 <= neighbor["joint_orders"] <= min(count, model["order_counts"][other]):
                            raise ValueError("Invalid co-purchase count")
                        if key == "recommendations" and not 0 < neighbor["score"] <= 1:
                            raise ValueError("Invalid similarity")
            for item in model["popular"]:
                if item["orders"] != model["order_counts"][item["id"]]:
                    raise ValueError("Inconsistent popularity counts")
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
            raise ArtifactError("Model is invalid or outdated. Run python -m src.build.") from exc

    @classmethod
    def load(cls, path: Path = MODEL_PATH):
        return cls(read_json(path))

    def product(self, code: str) -> dict:
        code = code.strip().upper()
        if code not in self._products:
            raise UnknownProducts([code])
        return dict(self._products[code])

    def _find_ids(self, query: str) -> tuple[str, ...]:
        terms = query.split()
        matches = [code for code, text in self._search_text.items()
                   if all(term in text for term in terms)]
        return tuple(sorted(matches, key=lambda code: (
            code.casefold() != query,
            not self._products[code]["name"].casefold().startswith(query),
            -self._products[code]["orders"], self._products[code]["name"], code,
        )))

    def search(self, query: str = "", limit: int = 20, offset: int = 0) -> dict:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("limit must be 1–100 and offset must be nonnegative")
        matches = self._find(" ".join(query.casefold().split()))
        return {"items": [self.product(code) for code in matches[offset:offset + limit]],
                "total": len(matches), "limit": limit, "offset": offset}

    def _rank_uncached(self, seeds: tuple, k: int, method: str, exclude: tuple):
        return recommend_basket(self.model, list(seeds), k, method, exclude)

    def recommendations(self, products, k: int = 8, method: str = "similarity", exclude=()) -> dict:
        seeds = tuple(sorted({code.strip().upper() for code in products}))
        unknown = set(seeds) - self._products.keys()
        if unknown:
            raise UnknownProducts(unknown)
        omitted = tuple(sorted(set(exclude)))
        ranked = self._rank(seeds, k, method, omitted)
        items = []
        for item in ranked:
            source = item["source"]
            if source == "popularity":
                reason = f"Purchased in {self._products[item['id']]['orders']:,} training baskets."
            elif len(seeds) == 1:
                reason = f"Bought together in {item['joint_orders']:,} training baskets. "
                reason += (f"Cosine similarity {item['score']:.3f}." if source == "similarity"
                           else f"Present in {item['score']:.1%} of this product's baskets.")
            else:
                reason = f"Matches {len(item['matched_products'])} of {len(seeds)} basket products. "
                reason += (f"Mean cosine similarity {item['score']:.3f}." if source == "similarity"
                           else f"Mean co-purchase confidence {item['score']:.1%}.")
            items.append({**self.product(item["id"]), **item,
                          "matched_products": list(item["matched_products"]),
                          "is_fallback": source == "popularity" and method != "popularity",
                          "reason": reason})
        return {"query_products": list(seeds), "method": method,
                "recommendations": items,
                "fallback_count": sum(item["is_fallback"] for item in items),
                "build_id": self.model["build_id"]}
