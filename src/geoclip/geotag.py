from geoclip import GeoCLIP
from loguru import logger

from src.ocr import OCRPipeline


class GeoTag:
    def __init__(self, ocr_pipeline: OCRPipeline | None = None):
        self.model = GeoCLIP()
        self.ocr = ocr_pipeline or OCRPipeline()

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
        ocr_payload = ocr_result.model_dump(mode="json")
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
            "map": {
                "latitude": location["Latitude"],
                "longitude": location["Longitude"],
            },
        }
