from __future__ import annotations

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class TextResponse(BaseModel):
    label: str
    confidence: float


class ImageRequest(BaseModel):
    image_base64: str = Field(min_length=1)


class ImageResponse(BaseModel):
    ai_generated_prob: float
    provenance: dict


class FuseRequest(BaseModel):
    text: str = ""
    image_base64: str = ""


class FuseResponse(BaseModel):
    confidence: float
    band: str
    explanation: str
    used_signals: list[str]
    breakdown: dict
    disclaimer: str = "This is an estimate, not an oracle."
