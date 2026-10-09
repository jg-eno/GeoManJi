"""Gemini calls: OCR on the photo (plus CV crops) and the final fusion of every stage."""

import json
import os
import re
import threading
import time
from collections.abc import Sequence
from typing import Any

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv()

MAX_UPLOAD_SIDE = 1600  # photos are downscaled before upload; plenty for reading signs


class LanguageInfo(BaseModel):
    name: str = Field(description="Language name")
    script: str | None = Field(default=None, description="Writing script, e.g. Latin, Devanagari")


class Clue(BaseModel):
    type: str = Field(description="Category: place, area, postal, road, landmark, business, transport, plate, sign")
    label: str = Field(description="Human-readable description of the clue")
    value: str = Field(description="The detected text or entity")
    evidence: str | None = Field(default=None, description="Visible text supporting the clue")


class OCRResult(BaseModel):
    text: str = Field(default="", description="Verbatim transcription of all visible text")
    languages: list[LanguageInfo] = Field(default_factory=list)
    clues: list[Clue] = Field(default_factory=list)
    area_guess: str | None = Field(default=None, description="Likely area, only when the text supports it")
    area_evidence: str | None = Field(default=None, description="Text supporting the area guess")
    error: str | None = None


class FinalPlace(BaseModel):
    rank: int = Field(ge=1)
    name: str = Field(description="Most specific supported place, e.g. 'MG Road, Bengaluru, Karnataka'")
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    confidence: float = Field(ge=0.0, le=1.0, description="Honest probability that this place is correct")
    reasoning: str = Field(description="One or two sentences naming the evidence used")
    sources: list[str] = Field(
        default_factory=list,
        description="Pipeline inputs supporting this place: image, cv, ocr, geoclip, plonk",
    )


class FinalPrediction(BaseModel):
    places: list[FinalPlace] = Field(default_factory=list, description="Up to three places, best first")
    error: str | None = None


OCR_PROMPT = """Read the text in this photograph.

The first image is the full photograph. Any further images are full-resolution crops of text regions and number plates found by a computer-vision detector; use them to read small text.

1. Transcribe all visible text into text, preserving the original wording, line breaks and scripts.
2. List every language and writing script in the text.
3. Extract location clues that are explicitly visible: place names, administrative areas, postal codes, roads, transport terms, landmarks, businesses, number plates and signs.
4. If the text supports a likely area, return it in area_guess with the exact supporting text in area_evidence.
5. Do not invent a location or clue. Return only JSON matching the schema.
"""

FUSION_PROMPT = """You are the final step of a photo geolocation pipeline. Combine the photograph with the outputs of the earlier stages below and give your best final location predictions.

Stages:
- cv: classical OpenCV analysis (sharpness, whether the image is blurry, number of text regions and number plates found).
- ocr: text read from the photo, its languages, location clues, and an area guess.
- geoclip: one coordinate from an image-to-GPS retrieval model, with its probability.
- plonk: up to five candidate areas in India from a generative geolocation model. sample_share is the fraction of the model's samples in that area, not a calibrated probability. evidence_match means the area's name matched the OCR text.

Rules:
1. Return up to 3 places, best first. Be as specific as the evidence allows (landmark, street, neighbourhood, city), otherwise a city or state.
2. Weigh the evidence. Strongest: readable place names, number-plate state codes, PIN or phone codes, language and script, recognisable landmarks. Next: model candidates that agree with each other or with the text. Weakest: a single model guess on its own.
3. If a place matches a PLONK or GeoCLIP candidate you may reuse its coordinates; otherwise give the coordinates of the place you name.
4. The app targets India. If the evidence clearly shows the photo is outside India, say so in reasoning and predict the real location.
5. confidence is your honest probability that the place is correct; the values need not sum to 1. Do not invent evidence.
6. In reasoning, name the specific evidence used. In sources, list which of image, cv, ocr, geoclip, plonk support the place.

Pipeline outputs (JSON):
"""


class Gemini:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.timeout_ms = int(os.getenv("GEMINI_TIMEOUT_MS", "30000"))
        self.max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
        self._client = None
        self._client_lock = threading.Lock()

    def read_text(self, image_path, crops: Sequence[bytes] = ()) -> OCRResult:
        parts = [types.Part.from_bytes(data=crop, mime_type="image/jpeg") for crop in crops]
        return self._ask(image_path, [*parts, OCR_PROMPT], OCRResult)

    def fuse(self, image_path, context: dict[str, Any]) -> FinalPrediction:
        prompt = FUSION_PROMPT + json.dumps(context, ensure_ascii=False, indent=1)
        return self._ask(image_path, [prompt], FinalPrediction)

    def _ask(self, image_path, parts: list[Any], schema: type[BaseModel]):
        """Send the photo plus extra parts; parse the reply into `schema`, or return it with `error` set."""
        if not self.api_key:
            return schema(error="GEMINI_API_KEY is not configured")
        try:
            contents = [_photo_part(image_path), *parts]
            return schema.model_validate(_parse_json(self._generate(contents, schema)))
        except Exception as exc:
            logger.exception("Gemini {} request failed", schema.__name__)
            return schema(error=f"{type(exc).__name__}: {exc}")

    def _generate(self, contents: list[Any], schema: type[BaseModel]) -> str:
        response_schema = schema.model_json_schema()
        response_schema["properties"].pop("error", None)  # set by us, never by the model
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.1,
            max_output_tokens=2048,
        )
        for attempt in range(self.max_retries + 1):
            try:
                response = self._get_client().models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                return (response.text or "").strip()
            except Exception as exc:
                # Retry rate limits, server errors, timeouts and dropped connections.
                code = exc.code if isinstance(exc, errors.APIError) else None
                name = type(exc).__name__.lower()
                retryable = code == 429 or (code or 0) >= 500 or "timeout" in name or "connect" in name
                if attempt >= self.max_retries or not retryable:
                    raise
                delay = 2**attempt
                logger.warning("Gemini {} (attempt {}), retrying in {}s", code or name, attempt + 1, delay)
                time.sleep(delay)

    def _get_client(self):
        with self._client_lock:
            if self._client is None:
                self._client = genai.Client(
                    api_key=self.api_key, http_options=types.HttpOptions(timeout=self.timeout_ms)
                )
        return self._client


def _photo_part(image_path) -> types.Part:
    """Downscale large photos so uploads stay small and never hit size limits."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    scale = MAX_UPLOAD_SIDE / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    return types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")


def _parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Gemini response did not contain a JSON object")
    return json.loads(text[start : end + 1])
