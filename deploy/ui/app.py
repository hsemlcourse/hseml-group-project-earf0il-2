import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# Allow running from repo root or from inside ``deploy/ui``.
_DEPLOY_DIR = Path(__file__).resolve().parents[1]
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

from api.preprocess import preprocess_for_model  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SVHN Noisy Digit Classifier",
    page_icon="🔢",
    layout="wide",
)

st.title("🔢 SVHN Noisy Digit Classifier")
st.caption(
    "ResNet-like CNN trained on SVHN with salt-and-pepper noise · "
    "val accuracy 0.9348, macro-F1 0.9296"
)

with st.sidebar:
    st.subheader("Service")
    st.code(API_URL)
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.ok:
            health = r.json()
            st.success(f"status: {health.get('status')}")
            st.json(health)
        else:
            st.error(f"/health returned {r.status_code}")
    except requests.RequestException as exc:
        st.error(f"API unreachable: {exc}")

    st.markdown("---")
    st.subheader("Preprocessing pipeline")
    st.markdown(
        "1. resize → **32×32 RGB**\n"
        "2. medianBlur (k=3) × 2 passes\n"
        "3. crop sides `[:, 8:24, :]` → **32×16**\n"
        "4. normalize `/255` → float32"
    )

uploaded = st.file_uploader(
    "Upload a digit image (PNG / JPG)",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    accept_multiple_files=False,
)

if uploaded is not None:
    image_bytes = uploaded.read()

    # Show local visualisation of pipeline (does not need the API).
    try:
        _, intermediates = preprocess_for_model(image_bytes, return_intermediate=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not decode image: {exc}")
        st.stop()

    st.subheader("Preprocessing visualisation")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.caption("Original")
        st.image(Image.open(io.BytesIO(image_bytes)), use_column_width=True)
    with c2:
        st.caption("Resized 32×32")
        st.image(intermediates["raw"], use_column_width=True)
    with c3:
        st.caption("Denoised (median ×2)")
        st.image(intermediates["denoised"], use_column_width=True)
    with c4:
        st.caption("Cropped 32×16")
        st.image(intermediates["cropped"], use_column_width=True)

    st.subheader("Prediction")
    with st.spinner("Calling API..."):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded.name, image_bytes, uploaded.type or "image/png")},
                timeout=30,
            )
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    if not resp.ok:
        st.error(f"API error {resp.status_code}: {resp.text}")
        st.stop()

    result = resp.json()
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted digit", result["label"])
    m2.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
    m3.metric("Inference time", f"{result['inference_time_ms']:.1f} ms")

    probs = result["probabilities"]
    df = (
        pd.DataFrame({"class": list(probs.keys()), "probability": list(probs.values())})
        .sort_values("class")
        .set_index("class")
    )
    st.bar_chart(df)

    with st.expander("Raw response"):
        st.json(result)
else:
    st.info("Upload an image to get a prediction. Best with 32×32 RGB digit crops.")
