"""FastAPI entry point: python -m uvicorn src.api:app --port 8000."""
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from src.service import CatalogService, UnknownProducts
from src.storage import ArtifactError, EVALUATION_PATH, MODEL_PATH, read_evaluation

Method = Literal["similarity", "frequently_bought", "popularity"]
ProductCode = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)]
logger = logging.getLogger(__name__)


class Product(BaseModel):
    id: str
    name: str
    orders: int
    price_gbp: float | None


class SearchResponse(BaseModel):
    items: list[Product]
    total: int
    limit: int
    offset: int


class Recommendation(Product):
    score: float
    source: Method
    matched_products: list[str]
    joint_orders: int | None
    is_fallback: bool
    reason: str


class RecommendationResponse(BaseModel):
    query_products: list[str]
    method: Method
    recommendations: list[Recommendation]
    fallback_count: int
    build_id: str


class BasketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    product_ids: list[ProductCode] = Field(default_factory=list, max_length=50)
    method: Method = "similarity"
    k: int = Field(default=8, ge=1, le=50)

    @field_validator("product_ids")
    @classmethod
    def normalize_ids(cls, values):
        return [value.upper() for value in values]


def get_service(request: Request) -> CatalogService:
    service = request.app.state.service
    if service is None:
        raise HTTPException(503, "Recommendation model unavailable. Run python -m src.build and restart.")
    return service


def create_app(model_path: Path = MODEL_PATH, evaluation_path: Path = EVALUATION_PATH) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        try:
            application.state.service = CatalogService.load(model_path)
        except ArtifactError as exc:
            logger.error("Model startup failed: %s", exc)
            application.state.service = None
        yield

    application = FastAPI(
        title="Retail Compass API", version="2.0.0", lifespan=lifespan,
        description="Historical co-purchase recommendations. Scores are associations, not purchase probabilities. "
                    "Prices are median historical training prices in GBP. Empty baskets use popularity.",
    )

    @application.exception_handler(UnknownProducts)
    async def unknown_product_handler(request, exc):
        return JSONResponse(status_code=404, content={
            "detail": {"message": "Unknown product codes.", "product_ids": exc.products},
        })

    @application.get("/health", tags=["Operations"])
    def health(service: CatalogService = Depends(get_service)):
        return {"status": "ok", "build_id": service.model["build_id"],
                "catalog_products": len(service.model["catalog"])}

    @application.get("/api/v1/products", response_model=SearchResponse, tags=["Products"])
    def products(q: str = Query(default="", max_length=100),
                 limit: int = Query(default=20, ge=1, le=100),
                 offset: int = Query(default=0, ge=0),
                 service: CatalogService = Depends(get_service)):
        return service.search(q, limit, offset)

    @application.get("/api/v1/products/{product_id}", response_model=Product, tags=["Products"])
    def product(product_id: str, service: CatalogService = Depends(get_service)):
        return service.product(product_id)

    @application.post("/api/v1/recommendations/basket", response_model=RecommendationResponse,
                      tags=["Recommendations"])
    def basket(body: BasketRequest, service: CatalogService = Depends(get_service)):
        return service.recommendations(body.product_ids, body.k, body.method)

    @application.get("/api/v1/recommendations/{product_id}", response_model=RecommendationResponse,
                     tags=["Recommendations"])
    def single(product_id: str, method: Method = "similarity",
               k: int = Query(default=8, ge=1, le=50),
               service: CatalogService = Depends(get_service)):
        return service.recommendations([product_id], k, method)

    @application.get("/api/v1/evaluation", tags=["Evaluation"])
    def evaluation(service: CatalogService = Depends(get_service)):
        try:
            return read_evaluation(service.model, evaluation_path)
        except ArtifactError as exc:
            raise HTTPException(503, "Verified evaluation unavailable. Rebuild the model and report together.") from exc

    return application


app = create_app()
