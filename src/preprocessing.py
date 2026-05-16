"""Image preprocessing utilities: denoising, cropping, leakage-aware splits."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .utils import SEED


def denoise_median(images: np.ndarray, ksize: int = 3, passes: int = 2) -> np.ndarray:
    """Apply median blur to an array of images of shape (N, H, W, C).

    Median filtering is a classic remedy for salt-and-pepper noise — exactly the
    type of noise that was added to the modified SVHN dataset.

    Parameters
    ----------
    images : np.ndarray, dtype uint8 / float
        Input images (N, H, W, C) or (N, H, W).
    ksize : int
        Kernel size for median blur (must be odd).
    passes : int
        Number of times to apply the filter.
    """
    import cv2

    out = images.copy()
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    for i in range(len(out)):
        img = out[i]
        for _ in range(passes):
            img = cv2.medianBlur(img, ksize)
        out[i] = img
    return out


def crop_sides(images: np.ndarray, left: int = 8, right: int = 8) -> np.ndarray:
    """Crop ``left`` and ``right`` pixel columns to remove neighbouring digits.

    SVHN images often contain edges of adjacent house-numbers; cropping helps
    the model focus on the central digit.
    """
    return images[:, :, left: images.shape[2] - right, :]


def normalize(images: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0, 1] floats."""
    return images.astype(np.float32) / 255.0


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.1,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val split with fixed seed.

    Notes on data leakage prevention:
    * The split is performed **before** any scaling/PCA fitting so that the
      validation set remains unseen during fitting.
    * Stratification preserves class balance.
    * The Kaggle test set (``data_test``) is held out and **never** touches
      training data.
    """
    return train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y, shuffle=True
    )


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten (N, H, W, C) -> (N, H*W*C) for classical ML models."""
    return images.reshape(len(images), -1)


def to_grayscale(images: np.ndarray) -> np.ndarray:
    """Convert RGB images to luminance grayscale (N, H, W)."""
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (images.astype(np.float32) @ weights)


def hog_features(images: np.ndarray) -> np.ndarray:
    """Compute HOG descriptors per image — a basic feature-engineering example."""
    from skimage.feature import hog

    feats = []
    gray = to_grayscale(images)
    for img in gray:
        feats.append(
            hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                feature_vector=True,
            )
        )
    return np.asarray(feats, dtype=np.float32)
