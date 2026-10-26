"""Small reproducible parameter sweep over a user-provided image folder."""
from pathlib import Path
import argparse
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from underwater_enhancement.image_io import load_image, save_image
from underwater_enhancement.metrics import quality_metrics
from underwater_enhancement.pipeline import PipelineConfig, run_pipeline

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image_dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "experiments")
    a = p.parse_args()
    if not a.image_dir.is_dir(): p.error(f"Not a directory: {a.image_dir}")
    files = [x for x in a.image_dir.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    if not files: p.error("No images found")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [("no_enhancement", PipelineConfig(white_balance=False, contrast="none", sharpening="none")),
             ("grayworld_only", PipelineConfig(white_balance=True, contrast="none", sharpening="none")),
             ("grayworld_clahe", PipelineConfig(white_balance=True, contrast="clahe", sharpening="none")),
             ("haze_clahe_unsharp", PipelineConfig(white_balance=True, haze_strength=0.5, contrast="clahe", sharpening="unsharp"))]
    rows=[]
    for f in files:
        im=load_image(f)
        for label, cfg in cases:
            result,_=run_pipeline(im,cfg)
            save_image(a.output_dir / f"{f.stem}_{label}.png",result)
            rows.append({"file":f.name,"method":label,**quality_metrics(result)})
    df=pd.DataFrame(rows); df.to_csv(a.output_dir/"experiment_results.csv",index=False)
    try:
        import matplotlib.pyplot as plt
        df.groupby("method")[ ["UCIQE", "UIQM (approx.)"] ].mean().plot(kind="bar", title="Mean no-reference metrics on supplied images")
        plt.tight_layout(); plt.savefig(a.output_dir/"experiment_plot.png",dpi=160); plt.close()
    except ImportError: pass
    print(f"Measured {len(rows)} outputs from {len(files)} supplied images; results are in {a.output_dir}")
if __name__ == "__main__": main()
