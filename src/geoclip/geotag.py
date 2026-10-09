from loguru import logger

from geoclip import GeoCLIP
from src.ocr import OCRPipeline
from src.ocr.models import LocationPrediction, OCRResult
from src.plonk_india import PlonkIndiaService


class GeoTag:
    def __init__(
        self,
        ocr_pipeline: OCRPipeline | None = None,
        plonk_service: PlonkIndiaService | None = None,
    ):
        self.model = GeoCLIP()
        self.ocr = ocr_pipeline or OCRPipeline()
        self.plonk = plonk_service or PlonkIndiaService()

    def predict(self, img):
        top_pred_gps, top_pred_prob = self.model.predict(img, top_k=1)
        lat, lon = top_pred_gps[0]
        logger.info("Latitude : {} Longitude : {}", lat, lon)
        return {
            "Latitude": float(lat),
            "Longitude": float(lon),
            "Probability": float(top_pred_prob[0]),
        }

    def predict_with_evidence(self, img) -> dict:
        location = self.predict(img)
        ocr_result = self.ocr.extract(img)
        evidence_terms = [
            value
            for value in (
                ocr_result.text,
                ocr_result.area_guess,
                *(clue.value for clue in ocr_result.clues),
            )
            if value
        ]
        plonk_result = self.plonk.predict(
            img,
            top_k=5,
            evidence_terms=evidence_terms,
        )
        prediction = self._location_prediction(ocr_result, location)
        ocr_payload = ocr_result.model_dump(mode="json")
        plonk_payload = plonk_result.model_dump(mode="json")
        return {
            **location,
            "ocr": ocr_payload,
            "languages": ocr_payload["languages"],
            "primary_language": ocr_payload["primary_language"],
            "area_guess": ocr_payload["area_guess"],
            "area_confidence": ocr_payload["area_confidence"],
            "area_evidence": ocr_payload["area_evidence"],
            "analysis": {
                "source": ocr_payload["analysis_source"],
                "error": ocr_payload["analysis_error"],
                "summary": ocr_payload["summary"],
            },
            "evidence": [clue.model_dump(mode="json") for clue in ocr_result.clues],
            "location_prediction": prediction.model_dump(
                mode="json", exclude_none=True
            ),
            "similar_places": plonk_payload["locations"],
            "plonk": {
                key: value for key, value in plonk_payload.items() if key != "locations"
            },
            "map": {
                "provider": "OpenStreetMap",
                "latitude": location["Latitude"],
                "longitude": location["Longitude"],
                "locations": plonk_payload["locations"],
            },
        }

    @staticmethod
    def _location_prediction(
        ocr_result: OCRResult, location: dict
    ) -> LocationPrediction:
        primary = ocr_result.primary_language
        place_name = ocr_result.area_guess
        place_confidence = ocr_result.area_confidence
        evidence = ocr_result.area_evidence

        if place_name is None:
            place_clues = [
                clue for clue in ocr_result.clues if clue.type in _CLUE_PRIORITY_TYPES
            ]
            if place_clues:
                best = max(
                    place_clues,
                    key=lambda clue: clue.confidence or 0.0,
                )
                place_name = best.value
                place_confidence = best.confidence
                evidence = evidence or best.evidence

        return LocationPrediction(
            latitude=location["Latitude"],
            longitude=location["Longitude"],
            probability=location.get("Probability"),
            language=primary.name if primary else None,
            language_code=primary.code if primary else None,
            place_name=place_name,
            place_confidence=place_confidence,
            evidence=evidence,
            ocr_text=ocr_result.text,
            clues=[clue.model_copy() for clue in ocr_result.clues],
        )


_CLUE_PRIORITY_TYPES = {
    "place",
    "area",
    "postal",
    "road",
    "landmark",
    "business",
    "transport",
    "sign",
}
