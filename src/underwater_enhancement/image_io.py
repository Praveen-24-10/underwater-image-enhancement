"""Validated image input/output helpers."""
from pathlib import Path
import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if image is None or image.size == 0:
        raise ValueError("Cannot save an empty image.")
    if not cv2.imwrite(str(target), image):
        raise OSError(f"Could not write image: {target}")
    return target


def image_analysis(image: np.ndarray) -> dict[str, object]:
    if image is None or image.size == 0:
        raise ValueError("Image is empty.")
    return {"width": int(image.shape[1]), "height": int(image.shape[0]),
            "channels": 1 if image.ndim == 2 else int(image.shape[2]),
            "dtype": str(image.dtype), "mean_bgr": [float(x) for x in image.reshape(-1, image.shape[-1] if image.ndim == 3 else 1).mean(axis=0)],
            "std": float(image.std()), "min": int(image.min()), "max": int(image.max())}
