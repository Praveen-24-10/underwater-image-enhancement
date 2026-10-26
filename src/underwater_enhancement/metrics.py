"""Reference-based fidelity and no-reference underwater quality metrics."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _pair(reference: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if reference is None or test is None or reference.size == 0 or test.size == 0:
        raise ValueError("Both images must be non-empty.")
    if reference.shape != test.shape:
        raise ValueError("Reference and test images must have identical dimensions and channels.")
    if reference.dtype != np.uint8:
        reference = np.clip(reference, 0, 255).astype(np.uint8)
    if test.dtype != np.uint8:
        test = np.clip(test, 0, 255).astype(np.uint8)
    return reference, test


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    a, b = _pair(reference, test)
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    error = mse(reference, test)
    return float("inf") if error == 0 else float(10 * np.log10((255.0 ** 2) / error))


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    a, b = _pair(reference, test)
    if min(a.shape[:2]) < 3:
        raise ValueError("SSIM requires images at least 3x3 pixels.")
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        return float(structural_similarity(a, b, channel_axis=2, data_range=255))
    return float(structural_similarity(a, b, data_range=255))


def uciqe(image: np.ndarray) -> float:
    """UCIQE no-reference metric: weighted chroma spread, saturation, luminance contrast.

    Formula commonly reported: 0.4680*contrast_chroma + 0.2745*mean_saturation
    + 0.2576*contrast_luminance. Higher is often interpreted as better, but
    it does not measure semantic fidelity and may reward over-enhancement.
    """
    if image is None or image.size == 0:
        raise ValueError("Image must be non-empty.")
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0
    a, b = lab[:, :, 1] - 128, lab[:, :, 2] - 128
    chroma = np.sqrt(a * a + b * b)
    chroma_contrast = float(np.percentile(chroma, 99) - np.percentile(chroma, 1)) / 255.0
    saturation = chroma / np.sqrt(chroma * chroma + L * L * (255.0 ** 2) + 1e-12)
    luminance_contrast = float(np.percentile(L, 99) - np.percentile(L, 1))
    return 0.4680 * chroma_contrast + 0.2745 * float(saturation.mean()) + 0.2576 * luminance_contrast


def uiqm(image: np.ndarray) -> float:
    """Approximate UIQM (UICM, UISM, UIConM components); higher is heuristic only."""
    if image is None or image.size == 0:
        raise ValueError("Image must be non-empty.")
    bgr = np.asarray(image, dtype=np.uint8)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rg, yb = rgb[:, :, 0] - rgb[:, :, 1], 0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]
    uicm = float(np.sqrt(rg.var() + yb.var()) + 0.3 * (rg.mean() + yb.mean()))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    uism = float(np.mean(cv2.magnitude(gx, gy)))
    blocks = gray[:gray.shape[0] // 8 * 8, :gray.shape[1] // 8 * 8].reshape(-1, 8, gray.shape[1] // 8, 8)
    if blocks.size:
        local = blocks.transpose(0, 2, 1, 3).reshape(-1, 8, 8)
        means = local.mean(axis=(1, 2))
        contrasts = (local.max(axis=(1, 2)) - local.min(axis=(1, 2))) / (local.max(axis=(1, 2)) + local.min(axis=(1, 2)) + 1e-6)
        uiconm = float(np.mean(contrasts * np.log1p(np.abs(means))))
    else:
        uiconm = 0.0
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


def quality_metrics(image: np.ndarray) -> dict[str, float]:
    return {"UCIQE": uciqe(image), "UIQM (approx.)": uiqm(image)}
