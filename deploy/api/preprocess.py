"""Inference-time preprocessing — mirrors the training pipeline."""
import io
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image

TARGET_HW: Tuple[int, int] = (32, 32)  # (height, width) before crop_sides
INPUT_SHAPE: Tuple[int, int, int] = (32, 16, 3)  # post-crop network input

# Pillow >=10 moved resampling constants into Image.Resampling; fall back gracefully.
_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR  # type: ignore[attr-defined]


def _to_rgb_32x32(image_bytes: bytes) -> np.ndarray: # type: ignore
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # PIL.resize expects (W, H); TARGET_HW is (H, W) -> swap explicitly.
    img = img.resize((TARGET_HW[1], TARGET_HW[0]), _BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def preprocess_for_model(
    image_bytes: bytes,
    return_intermediate: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:

    raw = _to_rgb_32x32(image_bytes)  # (32, 32, 3) uint8

    # denoise_median: medianBlur expects uint8, ksize odd, applied per-image
    denoised = raw.copy()
    for _ in range(2):
        denoised = cv2.medianBlur(denoised, 3)

    # crop_sides(left=8, right=8): (32, 32, 3) → (32, 16, 3)
    cropped = denoised[:, 8:24, :]

    # normalize
    batch = (cropped.astype(np.float32) / 255.0)[None, ...]  # (1, 32, 16, 3)

    intermediates = {}
    if return_intermediate:
        intermediates = {"raw": raw, "denoised": denoised, "cropped": cropped}
    return batch, intermediates
