from __future__ import annotations

from pydantic import BaseModel, Field


class OCRDetection(BaseModel):
    text: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bounding_box: list[list[float]]
    reading_order: int | None = Field(default=None, ge=0)


class OCRClue(BaseModel):
    type: str = Field(description="Category of the detected location clue")
    label: str = Field(description="Human-readable description of the clue")
    value: str = Field(description="Detected text or entity")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: str | None = Field(default=None, description="OCR text supporting the clue")
    source: str | None = Field(default=None, description="Analysis source for the clue")


class LanguageInfo(BaseModel):
    name: str = Field(description="Detected language name")
    code: str | None = Field(default=None, description="ISO language code when known")
    script: str | None = Field(default=None, description="Writing script name when known")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: str | None = Field(default=None, description="OCR text supporting the detection")
    source: str | None = Field(default=None, description="Analysis source for the detection")


class GeminiAnalysis(BaseModel):
    ocr_text: str = Field(default="", description="Verbatim transcription of text read from the image")
    detections: list[OCRDetection] = Field(
        default_factory=list,
        description="Detected text lines with bounding boxes and confidence",
    )
    languages: list[LanguageInfo] = Field(default_factory=list)
    clues: list[OCRClue] = Field(default_factory=list)
    area_guess: str | None = Field(
        default=None,
        description="Likely area only when supported by the OCR evidence",
    )
    area_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    area_evidence: str | None = Field(
        default=None,
        description="OCR text supporting the area guess",
    )
    summary: str | None = Field(default=None)
    source: str = "gemini"
    error: str | None = None


class LocationPrediction(BaseModel):
    latitude: float = Field(description="Predicted latitude from GeoCLIP")
    longitude: float = Field(description="Predicted longitude from GeoCLIP")
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    language: str | None = Field(default=None, description="Detected language name")
    language_code: str | None = Field(default=None, description="Detected ISO language code")
    place_name: str | None = Field(default=None, description="Detected place/area name from the image text")
    place_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: str | None = Field(default=None, description="OCR text supporting the location")
    ocr_text: str | None = Field(default=None, description="Full detected text used for the prediction")
    clues: list[OCRClue] = Field(default_factory=list)


class OCRResult(BaseModel):
    text: str = ""
    detections: list[OCRDetection] = Field(default_factory=list)
    languages: list[LanguageInfo] = Field(default_factory=list)
    primary_language: LanguageInfo | None = None
    scripts: list[str] = Field(default_factory=list)
    clues: list[OCRClue] = Field(default_factory=list)
    area_guess: str | None = None
    area_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    area_evidence: str | None = None
    summary: str | None = None
    analysis_source: str | None = None
    analysis_error: str | None = None
    error: str | None = None

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())
