"""GeoManJi pipeline: runs every stage on one photo and merges the results.

    GeoCLIP  ──────────────────┐
    PLONK India ───────────────┤  run in parallel
    OpenCV ──► Gemini OCR ─────┘
                 │
    evidence matching ──► Gemini fusion ──► final places
"""

import re
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
        states = plate_states(terms)
        terms += [state["state"] for state in states]
        for place in plonk.places:
            place.evidence_match = matches_evidence(place, terms)

        final = self.gemini.fuse(image_path, fusion_context(cv_result, ocr, geoclip, plonk, states))
        return {
            "final": final.model_dump(mode="json"),
            "plate_states": states,
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


# Registration codes on Indian number plates (Motor Vehicles Act), including the older OR/UA/DN codes still on the road.
STATE_CODES = {
    "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam",
    "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh", "DD": "Dadra and Nagar Haveli and Daman and Diu",
    "DN": "Dadra and Nagar Haveli and Daman and Diu", "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat",
    "HP": "Himachal Pradesh", "HR": "Haryana", "JH": "Jharkhand", "JK": "Jammu and Kashmir", "KA": "Karnataka",
    "KL": "Kerala", "LA": "Ladakh", "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya", "MN": "Manipur",
    "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "OR": "Odisha", "PB": "Punjab",
    "PY": "Puducherry", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana",
    "TG": "Telangana", "UK": "Uttarakhand", "UA": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal",
}
# State code, district number, series letters, four-digit number: "TN 09 BY 9726", "MH12-AB-1234".
PLATE_PATTERN = re.compile(r"\b([A-Z]{2})[\s.-]*(\d{1,2})[\s.-]*([A-Z]{1,3})[\s.-]*(\d{4})\b")


def plate_states(texts: list[str]) -> list[dict]:
    """Indian plates in the OCR text, each with the state its code belongs to."""
    found = {}
    for match in PLATE_PATTERN.finditer(" ".join(texts).upper()):
        if match[1] in STATE_CODES:
            plate = " ".join(match.groups())
            found[plate] = {"plate": plate, "code": match[1], "state": STATE_CODES[match[1]]}
    return list(found.values())


def fusion_context(cv_result: dict, ocr: OCRResult, geoclip: dict, plonk: PlonkResult, states: list[dict]) -> dict:
    """Compact summary of every stage for the fusion prompt."""
    return {
        "cv": {
            "sharpness": cv_result.get("sharpness"),
            "blurry": cv_result.get("blurry"),
            "text_regions": len(cv_result.get("text_regions", [])),
            "morphology_regions": cv_result.get("morphology_regions"),
            "db_text_boxes": cv_result.get("db_regions"),
        },
        "plate_states": states,
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


if __name__ == "__main__":
    states = plate_states(["Plate: TN 09 BY 9726", "mh12-ab-1234, NO 883, XX 01 AB 1234", "TN09BY9726"])
    assert [s["state"] for s in states] == ["Tamil Nadu", "Maharashtra"], states
    print("ok")
