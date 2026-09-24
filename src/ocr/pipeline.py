from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from loguru import logger

from .gemini import GeminiAnalyzer
from .models import LanguageInfo, OCRClue, OCRDetection, OCRResult


_FALLBACK_SCRIPT_RANGES = {
    "Kannada": (0x0C80, 0x0CFF),
    "Devanagari": (0x0900, 0x097F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Malayalam": (0x0D00, 0x0D7F),
    "Arabic": (0x0600, 0x06FF),
    "Latin": (0x0041, 0x007A),
}


class OCRPipeline:
    def __init__(
        self,
        *,
        min_confidence: float | None = None,
        gemini_analyzer: GeminiAnalyzer | None = None,
    ) -> None:
        self.min_confidence = (
            self._env_float("OCR_MIN_CONFIDENCE", 0.1)
            if min_confidence is None
            else min_confidence
        )
        self.gemini = gemini_analyzer or GeminiAnalyzer()

    def extract(self, image_path: str | os.PathLike[str]) -> OCRResult:
        try:
            analysis, analysis_error = self.gemini.analyze_image(image_path)

            if analysis is not None:
                detections = self._normalize_detections(analysis.detections)
                text = analysis.ocr_text or self._text_from_detections(detections)
                languages = self._normalize_languages(analysis.languages, text)
                scripts = self._scripts_from_languages(languages) or detect_scripts(text)
                clues = self._normalize_clues(analysis.clues, detections)
                clues.extend(self._language_clues(languages, seen_clues=clues))
                area_guess = analysis.area_guess
                area_confidence = analysis.area_confidence
                area_evidence = analysis.area_evidence
                summary = analysis.summary
                analysis_source = "gemini"
            else:
                text = ""
                detections = []
                languages = self._fallback_languages(text, detections)
                scripts = detect_scripts(text)
                clues = self._language_clues(languages, seen_clues=[])
                area_guess = None
                area_confidence = None
                area_evidence = None
                summary = None
                analysis_source = "fallback"

            return OCRResult(
                text=text,
                detections=detections,
                languages=languages,
                primary_language=self._primary_language(languages),
                scripts=scripts,
                clues=clues,
                area_guess=area_guess,
                area_confidence=area_confidence,
                area_evidence=area_evidence,
                summary=summary,
                analysis_source=analysis_source,
                analysis_error=analysis_error,
            )
        except Exception as exc:
            logger.exception("OCR inference failed for {}", image_path)
            return OCRResult(error=f"{type(exc).__name__}: {exc}")

    def _normalize_detections(
        self, detections: Sequence[OCRDetection]
    ) -> list[OCRDetection]:
        normalized: list[OCRDetection] = []
        for detection in detections:
            text = " ".join(str(detection.text).split())
            if not text:
                continue
            if detection.confidence is not None and detection.confidence < self.min_confidence:
                continue
            normalized_box = self._normalize_box(detection.bounding_box)
            if normalized_box is None:
                continue
            normalized.append(
                OCRDetection(
                    text=text,
                    confidence=detection.confidence,
                    bounding_box=normalized_box,
                    reading_order=detection.reading_order,
                )
            )
        normalized.sort(
            key=lambda detection: (
                min(point[1] for point in detection.bounding_box),
                min(point[0] for point in detection.bounding_box),
            )
        )
        return normalized

    @staticmethod
    def _text_from_detections(detections: Sequence[OCRDetection]) -> str:
        return "\n".join(detection.text for detection in detections)

    def _normalize_languages(
        self, languages: Sequence[LanguageInfo], full_text: str
    ) -> list[LanguageInfo]:
        normalized: list[LanguageInfo] = []
        seen: set[tuple[str, str | None, str | None]] = set()
        for language in languages:
            script = language.script or self._script_for_text(language.evidence or "")
            key = (language.name, language.code, script)
            if not language.name.strip() or key in seen:
                continue
            seen.add(key)
            normalized.append(
                language.model_copy(
                    update={
                        "script": script,
                        "evidence": language.evidence or self._evidence_for_text(full_text),
                        "source": "gemini",
                    }
                )
            )
        return normalized

    def _normalize_clues(
        self, clues: Sequence[OCRClue], detections: Sequence[OCRDetection]
    ) -> list[OCRClue]:
        normalized: list[OCRClue] = []
        seen: set[tuple[str, str]] = set()
        for clue in clues:
            if not clue.value.strip():
                continue
            evidence = clue.evidence or self._matching_evidence(clue, detections)
            confidence = clue.confidence
            if confidence is None:
                confidence = self._matching_confidence(evidence, detections)
            key = (clue.type, clue.value.strip())
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                clue.model_copy(
                    update={
                        "evidence": evidence,
                        "confidence": confidence,
                        "source": clue.source or "gemini",
                    }
                )
            )
        return normalized

    def _language_clues(
        self, languages: Sequence[LanguageInfo], *, seen_clues: Sequence[OCRClue]
    ) -> list[OCRClue]:
        seen = {(clue.type, clue.value) for clue in seen_clues}
        clues: list[OCRClue] = []
        for language in languages:
            label = (
                f"{language.script} script detected"
                if language.script
                else f"{language.name} text detected"
            )
            value = language.script or language.name
            key = ("language", value)
            if key in seen:
                continue
            seen.add(key)
            clues.append(
                OCRClue(
                    type="language",
                    label=label,
                    value=value,
                    confidence=language.confidence,
                    evidence=language.evidence,
                    source=language.source or "fallback",
                )
            )
        return clues

    def _fallback_languages(
        self, text: str, detections: Sequence[OCRDetection]
    ) -> list[LanguageInfo]:
        languages: list[LanguageInfo] = []
        for script in detect_scripts(text):
            evidence = self._evidence_for_script(script, detections, text)
            languages.append(
                LanguageInfo(
                    name=f"{script} script",
                    script=script,
                    evidence=evidence,
                    source="fallback",
                )
            )
        return languages

    @staticmethod
    def _scripts_from_languages(languages: Sequence[LanguageInfo]) -> list[str]:
        scripts: list[str] = []
        for language in languages:
            if language.script and language.script not in scripts:
                scripts.append(language.script)
        return scripts

    @staticmethod
    def _primary_language(languages: Sequence[LanguageInfo]) -> LanguageInfo | None:
        if not languages:
            return None
        return max(
            languages,
            key=lambda language: language.confidence
            if language.confidence is not None
            else -1.0,
        )

    @staticmethod
    def _normalize_box(box: Any) -> list[list[float]] | None:
        try:
            points = [list(point) for point in box]
            if len(points) != 4 or any(len(point) != 2 for point in points):
                return None
            return [[float(point[0]), float(point[1])] for point in points]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _script_for_text(text: str) -> str | None:
        scripts = detect_scripts(text)
        return scripts[0] if len(scripts) == 1 else None

    @staticmethod
    def _evidence_for_text(text: str) -> str | None:
        stripped = text.strip()
        return stripped[:240] if stripped else None

    @staticmethod
    def _matching_evidence(
        clue: OCRClue, detections: Sequence[OCRDetection]
    ) -> str | None:
        value = clue.value.casefold()
        for detection in detections:
            if value in detection.text.casefold() or detection.text.casefold() in value:
                return detection.text
        return None

    @staticmethod
    def _matching_confidence(
        evidence: str | None, detections: Sequence[OCRDetection]
    ) -> float | None:
        if not evidence:
            return None
        normalized = evidence.casefold()
        confidences = [
            detection.confidence
            for detection in detections
            if detection.confidence is not None
            and (
                normalized in detection.text.casefold()
                or detection.text.casefold() in normalized
            )
        ]
        return max(confidences, default=None)

    @staticmethod
    def _evidence_for_script(
        script: str, detections: Sequence[OCRDetection], full_text: str
    ) -> str | None:
        start, end = _FALLBACK_SCRIPT_RANGES[script]
        for detection in detections:
            if any(start <= ord(character) <= end for character in detection.text):
                return detection.text
        return OCRPipeline._evidence_for_text(full_text)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default


def detect_scripts(text: str) -> list[str]:
    scripts: list[str] = []
    for script, (start, end) in _FALLBACK_SCRIPT_RANGES.items():
        if any(start <= ord(character) <= end for character in text):
            scripts.append(script)
    return scripts
