# Hybrid Underwater Image Enhancement and Super-Resolution Using Digital Image Processing and Deep Learning

A CPU-friendly college project implementing a configurable underwater image enhancement pipeline with OpenCV, a Streamlit UI, evaluation metrics, and reproducible experiment scripts. The deep-learning super-resolution option is optional and requires a separate compatible model file.

## Features

- Image validation, analysis, and PNG download.
- Gray-world white balancing and blended color correction.
- Luminance histogram equalization and CLAHE.
- Lightweight heuristic haze reduction.
- Median, Gaussian, or bilateral filtering.
- Nearest-neighbor, bilinear, and bicubic resizing.
- Laplacian or unsharp-mask sharpening.
- Optional OpenCV DNN EDSR super-resolution, CPU-capable.
- MSE, PSNR, SSIM for same-sized reference images; UCIQE and approximate UIQM without references.
- Batch comparisons, CSV tables, and plots computed from images supplied by the user.

## Windows installation

Use Python 3.10 or newer. In PowerShell, first change to the project folder. These commands work when starting in `C:\Users\tbunn`:

```powershell
Set-Location C:\Users\tbunn\DIP
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

Calling the virtual environment's Python executable directly avoids needing the `py` launcher, activation policy changes, or Streamlit on PATH. You can optionally activate the environment with `.\.venv\Scripts\Activate.ps1` from the project folder.

## Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Upload a locally available underwater image, choose processing stages in the sidebar, and download the enhanced PNG. The app does not bundle images; use datasets only when their license and citation conditions permit.

## Batch evaluation and experiments

Place images in a folder you control. These scripts create output folders automatically and never claim results for absent input data.

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate.py .\my_images --output-dir .\outputs\evaluation
.\.venv\Scripts\python.exe .\scripts\evaluate.py .\degraded --reference-dir .\ground_truth --output-dir .\outputs\paired_evaluation
.\.venv\Scripts\python.exe .\scripts\experiments.py .\my_images --output-dir .\outputs\experiments
```

Paired metrics are recorded only when a matching file name exists in the reference folder. Resizing a reference to align dimensions changes the comparison protocol, so document it in a formal report. A meaningful benchmark requires registered image pairs and consistent capture/processing conditions. No experimental results are included in this repository because no sample data was provided.

## Optional deep super-resolution

The pipeline accepts a local OpenCV DNN EDSR `.pb` model path through `PipelineConfig.super_resolution_model`. The model must match the selected architecture and scale. In the UI, enter its local path and enable the EDSR option. OpenCV builds without `cv2.dnn_superres` cannot run this feature; use a compatible `opencv-contrib-python` installation in place of `opencv-python` if needed (do not install both packages together). Model weights are not included and must be obtained from a source whose license permits your use.

Super-resolution estimates plausible high-frequency detail from learned priors. Those details were not measured by the camera and must not be treated as recovered ground truth. It can invent texture, alter small objects, and increase memory/time use. The classical pipeline runs on CPU; no GPU is required.

## Method notes and limitations

- Gray-world assumes average scene reflectance is approximately neutral; strongly colored scenes can be misbalanced.
- Color correction blends the source and gray-world result. It is a practical control rather than a calibrated underwater color model.
- Haze reduction uses percentile airlight and luminance-based transmission as a simple approximation; it is not a full physical water-attenuation/dehazing model.
- UCIQE and UIQM are no-reference indicators, not universal perceptual truth. This UIQM is explicitly approximate and should not be compared blindly with values from other implementations.
- PSNR, SSIM, and MSE only make sense against a suitable aligned reference. Enhancement is not expected to maximize these when the input itself is the reference.
- Color enhancement can amplify noise, clip highlights, and create halos. Inspect intermediate images and tune conservatively.

## Project structure

```text
app.py                         Streamlit interface
src/underwater_enhancement/    Processing, metrics, and image I/O
scripts/evaluate.py            Method comparison and metric plots
scripts/experiments.py         Reproducible parameter/configuration sweep
tests/                          Unit tests for core processing and metrics
docs/                           Report, presentation, and viva outlines
outputs/                        Generated results (ignored by Git)
```

## Verification

Run the automated unit checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

Check Streamlit starts with `.\.venv\Scripts\python.exe -m streamlit run .\app.py`. Metrics and plots are based on files actually supplied to the scripts; generated outputs are not evidence of quality without a documented dataset and protocol.
