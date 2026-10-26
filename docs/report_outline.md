# Project report outline

1. **Title and abstract** — state the problem, proposed classical pipeline, optional EDSR component, and measured evaluation protocol.
2. **Introduction** — underwater light attenuation, wavelength-dependent color loss, backscatter, low contrast, and motivation.
3. **Objectives and scope** — modular enhancement, interactive parameter selection, fair comparisons, CPU operation.
4. **Theory** — gray-world gains; luminance histogram equalization; CLAHE clipping and tiles; haze model `I(x)=J(x)t(x)+A(1-t(x))`; denoising; interpolation; Laplacian/unsharp mask; EDSR limitations.
5. **System design** — architecture diagram, module responsibilities, data flow, input validation and output handling.
6. **Implementation** — Python/OpenCV choices, parameter defaults, UI, experiment scripts, optional model interface.
7. **Experimental protocol** — name/version/license of dataset, image count and dimensions, reference alignment, hardware, exact parameters, runtime measurement procedure.
8. **Results** — insert actual per-image CSV summaries and generated plots; report MSE/PSNR/SSIM only for paired ground truth and UCIQE/UIQM with their limitations. Do not pre-fill results.
9. **Discussion** — color casts, detail/noise trade-offs, failure cases, metric disagreements, compute cost.
10. **Conclusion and future work** — calibrated color models, physical underwater restoration, learned benchmark comparisons.
11. **References and appendices** — citations, license notices, configuration and viva preparation.
