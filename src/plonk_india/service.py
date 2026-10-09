from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .india import india_only, rank_location_modes
from .models import PlonkIndiaResult, SimilarPlace
from .nominatim import NominatimResolver

logger = logging.getLogger(__name__)


class PlonkIndiaService:
    """Lazy PLONK inference followed by India filtering and mode ranking."""

    def __init__(
        self,
        *,
        model: str | None = None,
        sample_count: int | None = None,
        num_steps: int | None = None,
        cfg: float | None = None,
        cluster_radius_km: float | None = None,
        pipeline_factory: Callable[..., Any] | None = None,
        resolver: Any | None = None,
        image_loader: Callable[[Path], Any] | None = None,
    ) -> None:
        self.model = model or os.getenv("PLONK_MODEL", "nicolas-dufour/PLONK_OSV_5M")
        self.sample_count = (
            sample_count
            if sample_count is not None
            else self._env_int("PLONK_SAMPLES", 1024)
        )
        self.num_steps = (
            num_steps if num_steps is not None else self._env_int("PLONK_NUM_STEPS", 32)
        )
        self.cfg = cfg if cfg is not None else self._env_float("PLONK_CFG", 0.0)
        self.cluster_radius_km = (
            cluster_radius_km
            if cluster_radius_km is not None
            else self._env_float("PLONK_CLUSTER_RADIUS_KM", 125.0)
        )
        self.pipeline_factory = pipeline_factory
        self.resolver = resolver or NominatimResolver()
        self.image_loader = image_loader
        self._pipeline: Any = None
        self._pipeline_lock = threading.Lock()

    def predict(
        self,
        image_path: str | os.PathLike[str],
        *,
        top_k: int = 5,
        evidence_terms: list[str] | None = None,
    ) -> PlonkIndiaResult:
        result = PlonkIndiaResult(
            model=self.model,
            requested_samples=self.sample_count,
        )
        try:
            image = self._load_image(Path(image_path))
            predictions = self._get_pipeline()(
                image,
                batch_size=self.sample_count,
                cfg=self.cfg,
                num_steps=self.num_steps,
            )
            raw_coordinates = self._coordinates(predictions)
            indian_coordinates = india_only(raw_coordinates)
            modes = rank_location_modes(
                indian_coordinates,
                top_k=top_k,
                cluster_radius_km=self.cluster_radius_km,
            )
            resolved = self.resolver.resolve_many(modes)
            ranked = [
                (mode, label, self._matches_evidence(label, evidence_terms or []))
                for mode, label in zip(modes, resolved)
            ]
            ranked.sort(
                key=lambda candidate: (
                    not candidate[2],
                    -candidate[0].supporting_samples,
                )
            )
            places: list[SimilarPlace] = []
            for rank, (mode, label, evidence_match) in enumerate(ranked, start=1):
                name = label["name"] if label else f"India candidate {rank}"
                display_name = label.get("display_name") if label else None
                places.append(
                    SimilarPlace(
                        rank=rank,
                        name=name,
                        display_name=display_name,
                        latitude=mode.latitude,
                        longitude=mode.longitude,
                        sample_share=mode.sample_share,
                        supporting_samples=mode.supporting_samples,
                        evidence_match=evidence_match,
                        ranking_reason=(
                            "PLONK sample density plus OCR/Gemini place agreement"
                            if evidence_match
                            else "PLONK sample density inside India"
                        ),
                        map_url=self._map_url(mode.latitude, mode.longitude),
                    )
                )
            generated = len(raw_coordinates)
            india_count = len(indian_coordinates)
            return result.model_copy(
                update={
                    "generated_samples": generated,
                    "india_samples": india_count,
                    "india_sample_share": india_count / generated if generated else 0.0,
                    "locations": places,
                    "error": (
                        None
                        if len(places) == top_k
                        else f"Only {len(places)} distinct India candidates were present "
                        f"in {generated} generated samples"
                    ),
                }
            )
        except Exception as exc:
            logger.exception("PLONK India inference failed for %s", image_path)
            return result.model_copy(update={"error": f"{type(exc).__name__}: {exc}"})

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            with self._pipeline_lock:
                if self._pipeline is None:
                    if self.pipeline_factory is None:
                        from plonk import PlonkPipeline

                        factory = PlonkPipeline
                    else:
                        factory = self.pipeline_factory
                    self._pipeline = factory(model_path=self.model)
        return self._pipeline

    def _load_image(self, path: Path) -> Any:
        if self.image_loader is not None:
            return self.image_loader(path)
        from PIL import Image

        with Image.open(path) as image:
            return image.convert("RGB")

    @staticmethod
    def _coordinates(predictions: Any) -> list[tuple[float, float]]:
        values = predictions.tolist() if hasattr(predictions, "tolist") else predictions
        coordinates: list[tuple[float, float]] = []
        for value in values:
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            coordinates.append((float(value[0]), float(value[1])))
        return coordinates

    @staticmethod
    def _map_url(latitude: float, longitude: float) -> str:
        return (
            "https://www.openstreetmap.org/"
            f"?mlat={latitude:.6f}&mlon={longitude:.6f}"
            f"#map=12/{latitude:.6f}/{longitude:.6f}"
        )

    @staticmethod
    def _matches_evidence(
        label: dict[str, str] | None, evidence_terms: list[str]
    ) -> bool:
        if not label or not evidence_terms:
            return False

        def normalize(value: str) -> str:
            return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())

        evidence = normalize(" ".join(term for term in evidence_terms if term))
        if not evidence:
            return False
        label_parts = [
            normalize(part)
            for value in label.values()
            for part in value.split(",")
            if normalize(part) not in {"", "india"}
        ]
        normalized_terms = [normalize(term) for term in evidence_terms if term]
        return any(
            (len(part) >= 4 and part in evidence)
            or any(len(term) >= 4 and term in part for term in normalized_terms)
            for part in label_parts
        )

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return default
