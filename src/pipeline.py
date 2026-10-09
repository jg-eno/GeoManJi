"""GeoManJi pipeline: runs every stage on one photo and merges the results.

    GeoCLIP  ──────────────────┐
    PLONK India ───────────────┤  run in parallel
    OpenCV ──► Gemini OCR ─────┘
                 │
    evidence matching ──► Gemini fusion ──► final places
"""

from concurrent.futures import ThreadPoolExecutor

from geoclip import GeoCLIP
from loguru import logger

from . import cv
from .gemini import Gemini, OCRResult
from .plonk_india import PlonkIndiaService, PlonkResult, matches_evidence


class GeoPipeline:
    def __init__(self) -> None:
        self.geoclip = GeoCLIP()
        self.plonk = PlonkIndiaService()
        self.gemini = Gemini()

    def run(self, image_path) -> dict:
        # GeoCLIP and PLONK are independent model runs; CV -> OCR (a network call) overlaps with them.
        with ThreadPoolExecutor(max_workers=2) as pool:
            geoclip_job = pool.submit(self._geoclip, image_path)
            plonk_job = pool.submit(self.plonk.predict, image_path)
            cv_result, crops = cv.analyze(image_path)
            ocr = self.gemini.read_text(image_path, crops)
            geoclip, plonk = geoclip_job.result(), plonk_job.result()

        terms = [ocr.text, ocr.area_guess or "", *(clue.value for clue in ocr.clues)]
        for place in plonk.places:
            place.evidence_match = matches_evidence(place, terms)

        final = self.gemini.fuse(image_path, fusion_context(cv_result, ocr, geoclip, plonk))
        return {
            "final": final.model_dump(mode="json"),
            "geoclip": geoclip,
            "plonk": plonk.model_dump(mode="json"),
            "ocr": ocr.model_dump(mode="json"),
            "cv": cv_result,
        }

    def _geoclip(self, image_path) -> dict:
        try:
            gps, probability = self.geoclip.predict(str(image_path), top_k=1)
            latitude, longitude = gps[0]
            return {"latitude": float(latitude), "longitude": float(longitude), "probability": float(probability[0])}
        except Exception as exc:
            logger.exception("GeoCLIP inference failed for {}", image_path)
            return {"error": f"{type(exc).__name__}: {exc}"}


def fusion_context(cv_result: dict, ocr: OCRResult, geoclip: dict, plonk: PlonkResult) -> dict:
    """Compact summary of every stage for the fusion prompt."""
    return {
        "cv": {
            "sharpness": cv_result.get("sharpness"),
            "blurry": cv_result.get("blurry"),
            "text_regions": len(cv_result.get("text_regions", [])),
            "number_plates": len(cv_result.get("plates", [])),
        },
        "ocr": {
            "text": ocr.text,
            "languages": [language.name for language in ocr.languages],
            "area_guess": ocr.area_guess,
            "area_evidence": ocr.area_evidence,
            "clues": [{"type": clue.type, "value": clue.value} for clue in ocr.clues],
            "error": ocr.error,
        },
        "geoclip": {key: round(value, 4) if isinstance(value, float) else value for key, value in geoclip.items()},
        "plonk": [
            {
                "rank": place.rank,
                "name": place.name,
                "display_name": place.display_name,
                "latitude": round(place.latitude, 4),
                "longitude": round(place.longitude, 4),
                "sample_share": round(place.sample_share, 3),
                "evidence_match": place.evidence_match,
            }
            for place in plonk.places
        ],
    }
