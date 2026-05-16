"""Smoke tests for the FastAPI service.

The heavy TensorFlow model is replaced with a deterministic ``_DummyModel``
so the suite stays fast and TF-free.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest
from PIL import Image

# fastapi.testclient may be missing in environments without the deploy extras.
fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


def _png_bytes(
    size: Tuple[int, int] = (32, 32),
    color: Tuple[int, int, int] = (128, 128, 128),
) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _DummyModel:
    """Lightweight stand-in for the Keras model used in /predict."""

    input_shape: Tuple[Any, ...] = (None, 32, 16, 3)

    def predict(self, x: Any, verbose: int = 0) -> Any:  # noqa: ARG002
        # Deterministic distribution that peaks at class 7.
        batch_size = int(x.shape[0])
        probs = np.full((batch_size, 10), 0.05, dtype=np.float32)
        probs[:, 7] = 0.55
        return probs


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:  # noqa: ARG001
    # Make sure the repo root is on sys.path so ``deploy.api.main`` is importable.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    from deploy.api import main as api_main

    with TestClient(api_main.app) as c:
        # Lifespan has already run and may have loaded a real model — override it
        # with the deterministic dummy AFTER entering the context.
        api_main.state.model = _DummyModel()
        api_main.state.model_loaded = True
        api_main.state.classes = [str(i) for i in range(10)]
        yield c


def test_health(client: Any) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_info(client: Any) -> None:
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["input_shape"] == [32, 16, 3]
    assert body["classes"] == [str(i) for i in range(10)]


def test_predict_smoke(client: Any) -> None:
    files = {"file": ("test.png", _png_bytes(), "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == "7"
    assert body["label_index"] == 7
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == {str(i) for i in range(10)}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-5


def test_predict_invalid_payload(client: Any) -> None:
    files = {"file": ("bad.png", b"not an image", "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400
