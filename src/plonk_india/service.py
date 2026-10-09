import os
import re
import threading

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from .india import india_only, rank_location_modes
from .nominatim import NominatimResolver


class Place(BaseModel):
    rank: int = Field(ge=1)
    name: str
    display_name: str | None = None
    latitude: float
    longitude: float
    sample_share: float = Field(ge=0.0, le=1.0)
    evidence_match: bool = False


class PlonkResult(BaseModel):
    model: str
    generated_samples: int = 0
    india_samples: int = 0
    india_sample_share: float = 0.0
    places: list[Place] = Field(default_factory=list)
    error: str | None = None


class PlonkIndiaService:
    """PLONK sampling, then India filtering, clustering into modes, and place names."""

    def __init__(self) -> None:
        self.model = os.getenv("PLONK_MODEL", "nicolas-dufour/PLONK_OSV_5M")
        self.sample_count = int(os.getenv("PLONK_SAMPLES", "1024"))
        self.num_steps = int(os.getenv("PLONK_NUM_STEPS", "32"))
        self.cfg = float(os.getenv("PLONK_CFG", "0.0"))
        self.cluster_radius_km = float(os.getenv("PLONK_CLUSTER_RADIUS_KM", "125"))
        self.resolver = NominatimResolver()
        self._pipeline = None
        self._pipeline_lock = threading.Lock()

    def predict(self, image_path, top_k: int = 5) -> PlonkResult:
        result = PlonkResult(model=self.model)
        try:
            with Image.open(image_path) as image:
                samples = self._get_pipeline()(
                    image.convert("RGB"), batch_size=self.sample_count, cfg=self.cfg, num_steps=self.num_steps
                )
            coordinates = [(float(lat), float(lon)) for lat, lon, *_ in samples.tolist()]
            indian = india_only(coordinates)
            modes = rank_location_modes(indian, top_k=top_k, cluster_radius_km=self.cluster_radius_km)
            labels = self.resolver.resolve_many(modes)
            result.places = [
                Place(
                    rank=rank,
                    name=label["name"] if label else f"India candidate {rank}",
                    display_name=label["display_name"] if label else None,
                    latitude=mode.latitude,
                    longitude=mode.longitude,
                    sample_share=mode.sample_share,
                )
                for rank, (mode, label) in enumerate(zip(modes, labels), start=1)
            ]
            result.generated_samples, result.india_samples = len(coordinates), len(indian)
            result.india_sample_share = len(indian) / len(coordinates) if coordinates else 0.0
            if len(result.places) < top_k:
                result.error = f"Only {len(result.places)} distinct India candidates in {len(coordinates)} samples"
        except Exception as exc:
            logger.exception("PLONK inference failed for {}", image_path)
            result.error = f"{type(exc).__name__}: {exc}"
        return result

    def _get_pipeline(self):
        with self._pipeline_lock:
            if self._pipeline is None:
                from plonk import PlonkPipeline  # heavy import; only when first needed

                self._pipeline = PlonkPipeline(model_path=self.model)
        return self._pipeline


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def matches_evidence(place: Place, evidence_terms: list[str]) -> bool:
    """True when the place's name appears in the OCR text (or vice versa)."""
    terms = [_normalize(term) for term in evidence_terms if term]
    evidence = " ".join(terms)
    parts = [
        _normalize(part)
        for value in (place.name, place.display_name or "")
        for part in value.split(",")
        if _normalize(part) not in {"", "india"}
    ]
    return any(
        (len(part) >= 4 and part in evidence) or any(len(term) >= 4 and term in part for term in terms)
        for part in parts
    )
