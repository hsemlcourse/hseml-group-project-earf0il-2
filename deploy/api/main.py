import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .preprocess import INPUT_SHAPE, preprocess_for_model
from .schemas import (
    ErrorResponse,
    HealthResponse,
    InfoResponse,
    PredictionResponse,
)

logger = logging.getLogger("svhn-api")
logging.basicConfig(level=logging.INFO)

# Project root: deploy/api/main.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "models" / "resnet_like_final.keras"))
META_PATH = Path(os.environ.get("META_PATH", PROJECT_ROOT / "data" / "meta"))

DEFAULT_CLASSES: List[str] = [str(i) for i in range(10)]


class _State:
    model: Optional[Any] = None
    classes: List[str] = DEFAULT_CLASSES
    model_loaded: bool = False


state = _State()


def _load_classes() -> List[str]:
    """Load class names from meta file, fallback to '0'..'9'."""
    try:
        if META_PATH.exists():
            with open(META_PATH, "rb") as f:
                meta = pickle.load(f)
            names = meta.get("label_names")
            if names is not None:
                return [str(x) for x in names]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load meta from %s: %s", META_PATH, exc)
    return DEFAULT_CLASSES


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Loading model from %s", MODEL_PATH)
    try:
        # Import tensorflow lazily — avoids slow import for /health probes during build.
        from tensorflow.keras.models import load_model  # type: ignore

        state.model = load_model(str(MODEL_PATH))
        state.model_loaded = True
        logger.info("Model loaded; input_shape=%s", state.model.input_shape)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Model load failed: %s", exc)
        state.model = None
        state.model_loaded = False

    state.classes = _load_classes()
    logger.info("Classes: %s", state.classes)
    yield
    state.model = None


app = FastAPI(
    title="SVHN-Noisy Digit Classifier",
    description="ResNet-like CNN trained on SVHN with salt-and-pepper noise (val acc 0.9348).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.model_loaded else "degraded",
        model_loaded=state.model_loaded,
        model_path=str(MODEL_PATH),
    )


@app.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    return InfoResponse(
        name="resnet_like_final",
        framework="tensorflow.keras",
        input_shape=list(INPUT_SHAPE),
        classes=state.classes,
        preprocessing=[
            "resize to 32x32 RGB",
            "median blur ksize=3 x2 passes",
            "crop sides [:, 8:24, :] -> 32x16",
            "normalize /255 -> float32",
        ],
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise ValueError("Empty file")
        batch, _ = preprocess_for_model(image_bytes, return_intermediate=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    t0 = time.perf_counter()
    probs = state.model.predict(batch, verbose=0)[0]
    probs = np.asarray(probs, dtype=np.float64)
    # Ensure normalised (model already outputs softmax, but be defensive).
    if probs.sum() <= 0:
        raise HTTPException(status_code=500, detail="Model produced invalid output")
    probs = probs / probs.sum()

    idx = int(np.argmax(probs))
    classes = state.classes if len(state.classes) == len(probs) else DEFAULT_CLASSES
    label = classes[idx]
    prob_map = {classes[i]: float(probs[i]) for i in range(len(probs))}
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return PredictionResponse(
        label=label,
        label_index=idx,
        confidence=float(probs[idx]),
        probabilities=prob_map,
        inference_time_ms=round(elapsed_ms, 3),
    )
