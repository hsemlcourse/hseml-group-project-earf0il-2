"""Utilities: seed fixation, data loading, plotting helpers."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fix seed for ``random``, ``numpy`` and (if installed) ``tensorflow``.

    Addresses the *fixed seed* reproducibility criterion.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # tensorflow is optional for unit tests
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:  # pragma: no cover
        pass


def project_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


def load_raw_data(data_dir: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load pickled raw data dictionaries.

    The dataset stores three pickled Python dicts:

    * ``data_train`` — keys: ``images`` (N, 32, 32, 3) ``float32``, ``labels`` (N,) ``uint8``, ``section`` ``str``.
    * ``data_test``  — keys: ``images`` (N, 32, 32, 3) ``float32``, ``section`` ``str``. (no labels — Kaggle hold-out)
    * ``meta``       — keys: ``label_names``.

    Returns
    -------
    train, test, meta : dict
    """
    data_dir = Path(data_dir) if data_dir is not None else project_root() / "data"
    train = pd.read_pickle(data_dir / "data_train")
    test = pd.read_pickle(data_dir / "data_test")
    meta = pd.read_pickle(data_dir / "meta")
    return train, test, meta


def get_arrays(
    data_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Convenience: load raw data and unpack into arrays.

    Returns
    -------
    X_train : ndarray (N, 32, 32, 3) float32 in [0, 255]
    y_train : ndarray (N,) int64
    X_test  : ndarray (M, 32, 32, 3) float32 in [0, 255]
    y_test  : Optional[ndarray] — labels for test if present, otherwise ``None``
    meta    : dict with at least ``label_names``
    """
    train, test, meta = load_raw_data(data_dir)
    X_train = np.asarray(train["images"], dtype=np.float32)
    y_train = np.asarray(train["labels"], dtype=np.int64)
    X_test = np.asarray(test["images"], dtype=np.float32)
    y_test = np.asarray(test["labels"], dtype=np.int64) if "labels" in test else None
    return X_train, y_train, X_test, y_test, meta
