"""Smoke tests for the SVHN classification pipeline."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow ``import src`` when tests are launched from repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import preprocessing, utils  # noqa: E402
from src.modeling import evaluate, make_knn, make_logreg  # noqa: E402


def test_set_global_seed_is_deterministic():
    utils.set_global_seed(42)
    a = np.random.rand(5)
    utils.set_global_seed(42)
    b = np.random.rand(5)
    np.testing.assert_allclose(a, b)


def test_normalize_range():
    img = (np.random.rand(2, 32, 32, 3) * 255).astype(np.uint8)
    out = preprocessing.normalize(img)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_crop_sides_shape():
    img = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    out = preprocessing.crop_sides(img, left=8, right=8)
    assert out.shape == (4, 32, 16, 3)


def test_flatten_shape():
    img = np.zeros((3, 32, 16, 3), dtype=np.float32)
    out = preprocessing.flatten_images(img)
    assert out.shape == (3, 32 * 16 * 3)


def test_stratified_split_preserves_classes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = np.tile(np.arange(10), 20)
    X_tr, X_val, y_tr, y_val = preprocessing.stratified_split(X, y, val_size=0.2, seed=42)
    assert len(X_tr) + len(X_val) == 200
    # all classes present in both splits
    assert set(y_tr.tolist()) == set(range(10))
    assert set(y_val.tolist()) == set(range(10))


def test_evaluate_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])
    m = evaluate(y_true, y_pred)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["f1_macro"] <= 1.0
    assert pytest.approx(m["accuracy"], rel=1e-6) == 5 / 6


def test_classical_models_factories_run():
    """Quickly check that classical-ML factories return scikit-learn estimators."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = rng.integers(0, 3, size=60)
    for clf in (make_logreg(seed=42), make_knn(n_neighbors=3)):
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (60,)
