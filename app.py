"""Streamlit interface for interactive underwater enhancement."""
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from underwater_enhancement.pipeline import PipelineConfig, run_pipeline, resize_image
from underwater_enhancement.image_io import image_analysis
from underwater_enhancement.metrics import quality_metrics

st.set_page_config(page_title="Underwater Image Enhancement", layout="wide")
st.title("Hybrid Underwater Image Enhancement and Super-Resolution")
st.caption("Classical digital image processing with optional OpenCV DNN super-resolution.")
upload = st.file_uploader("Upload an underwater image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if upload:
    raw = np.frombuffer(upload.getvalue(), np.uint8)
    source = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if source is None:
        st.error("The uploaded file could not be decoded as an image.")
        st.stop()
    with st.sidebar:
        st.header("Pipeline controls")
        white = st.checkbox("Gray-world white balance", True)
        correction = st.slider("Color correction blend", 0.0, 1.0, 0.0, 0.05)
        haze = st.slider("Underwater haze reduction", 0.0, 1.0, 0.0, 0.05)
        contrast = st.selectbox("Contrast enhancement", ["clahe", "histogram", "none"])
        clahe_clip = st.slider("CLAHE clip limit", 0.5, 8.0, 2.0, 0.1)
        clahe_grid = st.slider("CLAHE tile grid", 2, 16, 8)
        denoise = st.selectbox("Denoising", ["none", "median", "gaussian", "bilateral"])
        denoise_k = st.select_slider("Filter kernel size", options=[3, 5, 7, 9], value=5)
        interpolation = st.selectbox("Interpolation", ["nearest", "bilinear", "bicubic"])
        scale = st.slider("Resize scale", 0.25, 4.0, 1.0, 0.25)
        sharpening = st.selectbox("Sharpening", ["unsharp", "laplacian", "none"])
        amount = st.slider("Sharpening amount", 0.0, 2.0, 0.6, 0.05)
        sr = st.checkbox("Optional EDSR deep super-resolution")
        model = st.text_input("Local OpenCV EDSR .pb model path", help="Requires a compatible model and OpenCV contrib build.") if sr else ""
    config = PipelineConfig(white_balance=white, color_correction_strength=correction,
        haze_strength=haze, contrast=contrast, clahe_clip=clahe_clip, clahe_grid=clahe_grid,
        denoise_method=denoise, denoise_kernel=denoise_k, interpolation=interpolation,
        scale=scale, sharpening=sharpening, sharpen_amount=amount,
        super_resolution=sr, super_resolution_model=model)
    try:
        enhanced, stages = run_pipeline(source, config)
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
        st.stop()
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.json(image_analysis(source))
    with right:
        st.subheader("Enhanced")
        st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.json(image_analysis(enhanced))
    st.subheader("Intermediate results")
    stage_names = [name for name in stages if name not in ("Original", "Enhanced output")]
    if stage_names:
        cols = st.columns(min(3, len(stage_names)))
        for i, name in enumerate(stage_names):
            with cols[i % len(cols)]:
                st.caption(name)
                st.image(cv2.cvtColor(stages[name], cv2.COLOR_BGR2RGB), use_container_width=True)
    st.subheader("No-reference image quality indicators")
    st.write("UCIQE and UIQM are heuristic metrics; higher scores do not guarantee faithful or pleasing enhancement.")
    st.dataframe(pd.DataFrame([quality_metrics(enhanced)]), use_container_width=True)
    encoded_ok, encoded = cv2.imencode(".png", enhanced)
    if encoded_ok:
        st.download_button("Download enhanced PNG", encoded.tobytes(), file_name="enhanced_underwater.png", mime="image/png")
    st.download_button("Download comparison stages (CSV)", pd.DataFrame([
        {"stage": name, "width": im.shape[1], "height": im.shape[0], "mean_pixel": float(im.mean())}
        for name, im in stages.items()]).to_csv(index=False).encode(),
        file_name="stage_summary.csv", mime="text/csv")
else:
    st.info("Upload a locally sourced image for processing. This project does not bundle sample images; check dataset terms before use.")
