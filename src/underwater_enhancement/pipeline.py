"""Image enhancement operations and a configurable composable pipeline.

Images in this module use OpenCV BGR uint8 layout unless a function says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np


def validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input image is empty or is not a NumPy array.")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("Expected a grayscale, BGR, or BGRA image.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    """Scale each channel so its spatial mean matches the mean of all channels."""
    image = validate_image(image)
    if image.ndim == 2:
        return image.copy()
    means = image.astype(np.float32).mean(axis=(0, 1))
    target = float(means.mean())
    gains = target / np.maximum(means, 1.0)
    return np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def color_correction(image: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """Blend gray-world balancing with the source to avoid over-correction."""
    image = validate_image(image)
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    balanced = gray_world_white_balance(image)
    return cv2.addWeighted(image, 1.0 - strength, balanced, strength, 0)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Equalize luminance (Y in YCrCb) so color is less likely to shift."""
    image = validate_image(image)
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Apply CLAHE to luminance; clip_limit limits local histogram amplification."""
    image = validate_image(image)
    if clip_limit <= 0 or tile_grid < 1:
        raise ValueError("clip_limit and tile_grid must be positive")
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def reduce_underwater_haze(image: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """Approximate haze correction by attenuating veiling-light airlight.

    For each channel, estimate a high-percentile airlight A and map
    J=(I-A*(1-t))/max(t, eps), with a luminance-derived transmission t.
    This is a lightweight heuristic, not a physical underwater inversion.
    """
    image = validate_image(image)
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    if strength == 0:
        return image.copy()
    src = image.astype(np.float32)
    if image.ndim == 2:
        luminance = src
        airlight = float(np.percentile(src, 95))
    else:
        luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        airlight = np.percentile(src.reshape(-1, 3), 95, axis=0).reshape(1, 1, 3)
    norm_luma = luminance / 255.0
    transmission = np.clip(1.0 - strength * (0.72 - 0.5 * norm_luma), 0.25, 1.0)
    if image.ndim == 3:
        transmission = transmission[:, :, None]
    corrected = (src - airlight * (1.0 - transmission)) / transmission
    return np.clip(corrected, 0, 255).astype(np.uint8)


def denoise(image: np.ndarray, method: str, kernel_size: int = 5, sigma: float = 1.2,
            bilateral_d: int = 9, color_sigma: float = 60, space_sigma: float = 60) -> np.ndarray:
    image = validate_image(image)
    method = method.lower()
    if method == "none":
        return image.copy()
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")
    if method == "median":
        return cv2.medianBlur(image, kernel_size)
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=float(sigma))
    if method == "bilateral":
        return cv2.bilateralFilter(image, int(bilateral_d), float(color_sigma), float(space_sigma))
    raise ValueError(f"Unknown denoising method: {method}")


def resize_image(image: np.ndarray, scale: float, method: str = "bicubic") -> np.ndarray:
    image = validate_image(image)
    if not 0.1 <= scale <= 8:
        raise ValueError("scale must be between 0.1 and 8")
    interpolation = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR,
                     "bicubic": cv2.INTER_CUBIC}.get(method.lower())
    if interpolation is None:
        raise ValueError(f"Unknown interpolation method: {method}")
    h, w = image.shape[:2]
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interpolation)


def sharpen(image: np.ndarray, method: str = "unsharp", amount: float = 1.0,
            radius: float = 1.2) -> np.ndarray:
    """Laplacian adds -alpha*L(I); unsharp masking adds amount*(I-Gaussian(I))."""
    image = validate_image(image)
    if amount < 0 or radius <= 0:
        raise ValueError("amount must be nonnegative and radius positive")
    src = image.astype(np.float32)
    if method.lower() == "laplacian":
        lap = cv2.Laplacian(src, cv2.CV_32F, ksize=3)
        out = src - float(amount) * lap
    elif method.lower() == "unsharp":
        blurred = cv2.GaussianBlur(src, (0, 0), float(radius))
        out = src + float(amount) * (src - blurred)
    elif method.lower() == "none":
        return image.copy()
    else:
        raise ValueError(f"Unknown sharpening method: {method}")
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class PipelineConfig:
    white_balance: bool = True
    color_correction_strength: float = 0.0
    haze_strength: float = 0.0
    contrast: str = "clahe"  # none, histogram, clahe
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    denoise_method: str = "none"  # none, median, gaussian, bilateral
    denoise_kernel: int = 5
    interpolation: str = "bicubic"
    scale: float = 1.0
    sharpening: str = "unsharp"  # none, laplacian, unsharp
    sharpen_amount: float = 0.6
    sharpen_radius: float = 1.2
    super_resolution: bool = False
    super_resolution_model: str = ""


def run_pipeline(image: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run selected stages in a documented order and retain every intermediate."""
    current = validate_image(image)
    stages: dict[str, np.ndarray] = {"Original": current.copy()}
    if config.white_balance:
        current = gray_world_white_balance(current)
        stages["Gray-world white balance"] = current
    if config.color_correction_strength > 0:
        current = color_correction(current, config.color_correction_strength)
        stages["Color correction"] = current
    if config.haze_strength > 0:
        current = reduce_underwater_haze(current, config.haze_strength)
        stages["Haze reduction"] = current
    if config.contrast == "histogram":
        current = histogram_equalization(current)
        stages["Histogram equalization"] = current
    elif config.contrast == "clahe":
        current = clahe_enhancement(current, config.clahe_clip, config.clahe_grid)
        stages["CLAHE"] = current
    elif config.contrast != "none":
        raise ValueError("contrast must be none, histogram, or clahe")
    current = denoise(current, config.denoise_method, config.denoise_kernel)
    if config.denoise_method != "none":
        stages[f"{config.denoise_method.title()} filtering"] = current
    if config.scale != 1.0:
        current = resize_image(current, config.scale, config.interpolation)
        stages[f"{config.interpolation.title()} resize ({config.scale:g}x)"] = current
    if config.super_resolution:
        current = deep_super_resolve(current, config.super_resolution_model or None)
        stages["Optional deep super-resolution"] = current
    current = sharpen(current, config.sharpening, config.sharpen_amount, config.sharpen_radius)
    if config.sharpening != "none":
        stages[f"{config.sharpening.title()} sharpening"] = current
    stages["Enhanced output"] = current
    return current, stages


def deep_super_resolve(image: np.ndarray, model_path: str | None = None, scale: int = 4) -> np.ndarray:
    """Optional OpenCV DNN EDSR; requires a separately downloaded compatible model."""
    if not model_path:
        raise RuntimeError("Deep SR needs a local OpenCV-compatible .pb model; see README.")
    import os
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Super-resolution model not found: {model_path}")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", int(scale))
    return sr.upsample(validate_image(image))
