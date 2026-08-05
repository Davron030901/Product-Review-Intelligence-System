"""Request/response models. These ARE the frontend contract."""
from typing import List, Optional

from pydantic import BaseModel, Field

from src.config import ISSUE_LABELS


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Raw review text.", examples=["Arrived late and the box was crushed."])
    category: Optional[str] = Field(None, description="Optional product category, passed through to the response.")


class BatchRequest(BaseModel):
    reviews: List[str] = Field(..., max_length=500, description="Up to 500 reviews per call.")
    category: Optional[str] = None


class Sentiment(BaseModel):
    label: str = Field(..., description="negative | neutral | positive | unknown")
    confidence: float


class Issue(BaseModel):
    category: str
    confidence: float


class AnalyzeResponse(BaseModel):
    sentiment: Sentiment
    issues: List[Issue]
    low_confidence: bool = Field(..., description="True when a human should look at this result.")
    reasons: List[str] = Field(default_factory=list, description="Why it was flagged. Empty when confident.")
    input_category: Optional[str] = None
    word_count: int
    truncated: bool = False
    model_version: str
    model_backend: str
    processed_at: str


class BatchResponse(BaseModel):
    results: List[AnalyzeResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backend: Optional[str] = None
    model_version: Optional[str] = None


class TaxonomyResponse(BaseModel):
    issue_labels: List[str] = ISSUE_LABELS
    sentiment_labels: List[str]
