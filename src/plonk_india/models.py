from __future__ import annotations

from pydantic import BaseModel, Field


class SimilarPlace(BaseModel):
    rank: int = Field(ge=1)
    name: str
    display_name: str | None = None
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    sample_share: float = Field(ge=0.0, le=1.0)
    supporting_samples: int = Field(ge=1)
    evidence_match: bool = False
    ranking_reason: str
    map_url: str


class PlonkIndiaResult(BaseModel):
    source: str = "plonk"
    model: str
    country_filter: str = "India"
    requested_samples: int = Field(ge=1)
    generated_samples: int = Field(default=0, ge=0)
    india_samples: int = Field(default=0, ge=0)
    india_sample_share: float = Field(default=0.0, ge=0.0, le=1.0)
    locations: list[SimilarPlace] = Field(default_factory=list)
    error: str | None = None
