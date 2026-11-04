"""Small JSON artifact utilities; no database or training imports at serving time."""
import json
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.recommender import METHODS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "recommender.json"
EVALUATION_PATH = ROOT / "reports" / "evaluation.json"


class ArtifactError(RuntimeError):
    """An artifact is missing, unreadable, incompatible, or from another build."""


def write_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                dir=path.parent, delete=False) as handle:
            staged = Path(handle.name)
            json.dump(document, handle, indent=1, allow_nan=False)
            handle.write("\n")
        staged.replace(path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(result, dict):
            raise ValueError("Expected an object")
        return result
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Cannot read {path.name}. Run python -m src.build to rebuild artifacts.") from exc


def read_evaluation(model: dict, path: Path = EVALUATION_PATH) -> dict:
    report = read_json(path)
    if not report.get("build_id") or report["build_id"] != model.get("build_id"):
        raise ArtifactError("Evaluation and model are from different builds. Run python -m src.build.")
    try:
        for task in ("results", "basket_results"):
            for method in METHODS:
                for metric in ("hit_rate", "recall", "coverage"):
                    for k in (5, 10):
                        value = report[task][method][f"{metric}_at_{k}"]
                        if value is not None and (
                            isinstance(value, bool) or not isinstance(value, (int, float))
                            or not isfinite(value) or not 0 <= value <= 1
                        ):
                            raise ValueError("Invalid evaluation metric")
        for key in ("evaluated_test_invoices", "evaluated_basket_invoices"):
            if type(report[key]) is not int or report[key] < 0:
                raise ValueError("Invalid evaluation count")
        for key in ("single_product_task", "basket_task", "notes"):
            if not isinstance(report[key], str):
                raise ValueError("Invalid evaluation description")
        if not isinstance(report["split"]["cutoff"], str):
            raise ValueError("Invalid split metadata")
        fraction = report["test_catalog_item_fraction"]
        if fraction is not None and (
            isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not isfinite(fraction) or not 0 <= fraction <= 1
        ):
            raise ValueError("Invalid retained item fraction")
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("Evaluation report is incomplete or invalid. Run python -m src.build.") from exc
    return report
