"""Compare configured methods over a folder; writes metrics only from supplied files."""
from pathlib import Path
import argparse
import sys
import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from underwater_enhancement.image_io import load_image, save_image
from underwater_enhancement.metrics import quality_metrics, psnr, ssim, mse
from underwater_enhancement.pipeline import PipelineConfig, run_pipeline

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--reference-dir", type=Path, help="Optional matching ground-truth folder for PSNR/SSIM/MSE")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(p for p in args.input_dir.rglob("*") if p.suffix.lower() in EXTS):
        source = load_image(path)
        variants = {
            "original": source,
            "gray_world_clahe": run_pipeline(source, PipelineConfig(white_balance=True, contrast="clahe", sharpening="none"))[0],
            "histogram": run_pipeline(source, PipelineConfig(white_balance=False, contrast="histogram", sharpening="none"))[0],
            "bicubic_2x": run_pipeline(source, PipelineConfig(white_balance=False, contrast="none", scale=2, interpolation="bicubic", sharpening="none"))[0],
        }
        for method, image in variants.items():
            stem = f"{path.stem}_{method}"
            save_image(args.output_dir / f"{stem}.png", image)
            row = {"file": path.name, "method": method, **quality_metrics(image)}
            if args.reference_dir:
                refpath = args.reference_dir / path.name
                if refpath.exists():
                    ref = load_image(refpath)
                    if ref.shape != image.shape:
                        row.update({"MSE": None, "PSNR": None, "SSIM": None,
                                    "reference_status": "shape mismatch; register/align manually"})
                    else:
                        row.update({"MSE": mse(ref, image), "PSNR": psnr(ref, image), "SSIM": ssim(ref, image),
                                    "reference_status": "matched dimensions; alignment still must be verified"})
                else:
                    row.update({"MSE": None, "PSNR": None, "SSIM": None, "reference_status": "reference missing"})
            records.append(row)
    if not records:
        parser.error(f"No supported images found in {args.input_dir}")
    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "comparison_results.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        cols = [c for c in ("UCIQE", "UIQM (approx.)", "PSNR", "SSIM") if c in frame and frame[c].notna().any()]
        if cols:
            summary = frame.groupby("method")[cols].mean()
            ax = summary.plot(kind="bar", subplots=True, layout=(len(cols), 1), figsize=(9, 3 * len(cols)), legend=False)
            plt.tight_layout()
            plt.savefig(args.output_dir / "metric_comparison.png", dpi=160)
            plt.close("all")
    except ImportError:
        print("matplotlib unavailable: saved CSV, skipped plot")
    print(f"Saved actual measurements for {len(frame)} image-method pairs to {args.output_dir}")

if __name__ == "__main__":
    main()
