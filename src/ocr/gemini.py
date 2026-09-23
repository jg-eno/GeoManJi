from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from dotenv import load_dotenv
from loguru import logger

from .models import GeminiAnalysis

load_dotenv()

_IMAGE_MIME_FALLBACK = "image/png"
_MAX_IMAGE_BYTES = 4 * 1024 * 1024


class GeminiAnalyzer:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_ms: int | None = None,
        max_retries: int | None = None,
        retry_delay_ms: int | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.timeout_ms = self._env_int("GEMINI_TIMEOUT_MS", 30000) if timeout_ms is None else timeout_ms
        self.max_retries = (
            self._env_int("GEMINI_MAX_RETRIES", 2) if max_retries is None else max_retries
        )
        self.retry_delay_ms = (
            self._env_int("GEMINI_RETRY_DELAY_MS", 1000)
            if retry_delay_ms is None
            else retry_delay_ms
        )
        self.client_factory = client_factory
        self._client: Any = None
        self._client_lock = threading.Lock()

    def analyze(self, ocr_text: str) -> tuple[GeminiAnalysis | None, str | None]:
        if not ocr_text.strip():
            return None, None
        if not self.api_key:
            return None, "GEMINI_API_KEY is not configured"

        try:
            response_text = self._generate(contents=self._prompt(ocr_text))
            payload = self._parse_json(response_text)
            return GeminiAnalysis.model_validate(payload), None
        except Exception as exc:
            logger.exception("Gemini text analysis failed")
            return None, f"{type(exc).__name__}: {exc}"

    def analyze_image(
        self, image_path: str | os.PathLike[str]
    ) -> tuple[GeminiAnalysis | None, str | None]:
        path = Path(image_path)
        if not self.api_key:
            return None, "GEMINI_API_KEY is not configured"

        try:
            image_bytes, mime_type = self._read_image(path)
            contents = [
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                self._image_prompt(),
            ]
            response_text = self._generate(contents=contents)
            payload = self._parse_json(response_text)
            return GeminiAnalysis.model_validate(payload), None
        except Exception as exc:
            logger.exception("Gemini image analysis failed for {}", path)
            return None, f"{type(exc).__name__}: {exc}"

    def _generate(self, contents: Any) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._get_client().models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeminiAnalysis.model_json_schema(),
                        temperature=0.1,
                        max_output_tokens=2048,
                    ),
                )
                return (response.text or "").strip()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries or not self._is_retryable(exc):
                    raise
                delay = max(self.retry_delay_ms, 0) * (2 ** attempt) / 1000.0
                logger.warning(
                    "Retryable Gemini error on attempt {}/{}: {}; retrying in {:.0f}s",
                    attempt + 1,
                    self.max_retries + 1,
                    f"{type(exc).__name__}: {exc}",
                    delay,
                )
                time.sleep(delay)
        raise last_exc  # pragma: no cover - guarded by loop range above

    @staticmethod
    def _read_image(path: Path) -> tuple[bytes, str]:
        image_bytes = path.read_bytes()
        if not image_bytes:
            raise ValueError(f"Image is empty: {path}")
        if len(image_bytes) > _MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image exceeds {_MAX_IMAGE_BYTES} bytes ({len(image_bytes)}): {path}"
            )
        mime_type, _ = mimetypes.guess_type(str(path))
        return image_bytes, (mime_type or _IMAGE_MIME_FALLBACK)

    def _get_client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    http_options = None
                    if self.timeout_ms > 0:
                        http_options = types.HttpOptions(timeout=self.timeout_ms)
                    if self.client_factory is not None:
                        self._client = self.client_factory(
                            api_key=self.api_key,
                            http_options=http_options,
                        )
                    else:
                        self._client = genai.Client(
                            api_key=self.api_key,
                            http_options=http_options,
                        )
        return self._client

    @staticmethod
    def _prompt(ocr_text: str) -> str:
        lines = "\n".join(
            f"{index}. {line}"
            for index, line in enumerate(ocr_text.splitlines(), start=1)
        )
        return f"""Analyze OCR text extracted from a photograph.

Tasks:
1. Identify every language and writing script represented in the text.
2. Extract location-relevant clues that are explicitly supported by the OCR text, such as place names, administrative areas, postal codes, roads, transport terms, landmarks, businesses, and signs.
3. If the OCR evidence supports a likely area, return that area and the exact supporting OCR text in area_evidence.
4. Do not invent a location, coordinates, or clue. Mark uncertain OCR text as low confidence.
5. Return only JSON matching the supplied schema. Use the original OCR wording in evidence.

OCR text:
{lines}
"""

    @staticmethod
    def _image_prompt() -> str:
        return """You are reading OCR text from a photograph. First transcribe every scrap of visible text verbatim into ocr_text, preserving original wording, line breaks, and script. Then fill the structured fields.

Tasks:
1. Transcribe all visible text into ocr_text, preserving the original wording and scripts.
2. Identify every language and writing script represented in the text. For each detected line, return it in detections with an estimated confidence (0-1) and an axis-aligned bounding_box in pixel coordinates [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] ordered clockwise from the top-left.
3. Extract location-relevant clues that are explicitly visible in the image: place names, administrative areas, postal codes, roads, transport terms, landmarks, businesses, and signs.
4. If the evidence supports a likely area, return that area and the exact supporting text in area_evidence.
5. Do not invent a location, coordinates, or clue. Mark uncertain text as low confidence.
6. Return only JSON matching the supplied schema.
"""

    @staticmethod
    def _parse_json(response_text: str) -> dict[str, Any]:
        text = response_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Gemini response did not contain a JSON object")
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Gemini response JSON must be an object")
        return payload

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, genai_errors.ServerError):
            code = getattr(exc, "code", None)
            if isinstance(code, int):
                return code == 429 or 500 <= code < 600
            return True
        name = type(exc).__name__.lower()
        return (
            "ratelimit" in name
            or "timeout" in name
            or "connection" in name
            or "servererror" in name
        )
