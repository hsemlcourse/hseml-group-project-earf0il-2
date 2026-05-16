from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    # ``model_*`` collides with Pydantic's protected namespace; silence the warning.
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="'ok' if the service is ready to serve predictions")
    model_loaded: bool
    model_path: Optional[str] = None


class InfoResponse(BaseModel):
    name: str
    framework: str
    input_shape: List[int] = Field(..., description="Expected input tensor shape excluding batch dim, e.g. [32, 16, 3]")
    classes: List[str]
    preprocessing: List[str] = Field(
        default_factory=lambda: ["denoise_median(ksize=3, passes=2)", "crop_sides(left=8, right=8)", "normalize(/255)"],
        description="Preprocessing pipeline applied to the raw input before inference",
    )


class PredictionResponse(BaseModel):
    label: str = Field(..., description="Top-1 predicted digit label")
    label_index: int = Field(..., ge=0, le=9)
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(
        ...,
        description="Class probabilities for each of the 10 digit classes",
    )
    inference_time_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
