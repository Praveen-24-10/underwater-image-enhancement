# Viva questions and short answers

1. **Why does underwater imagery have a color cast?** Water attenuates wavelengths differently; red light is commonly lost faster than blue/green, depending on water and distance.
2. **What is the gray-world assumption?** The average scene reflectance is approximately neutral, so channel means can be balanced. It fails on scenes dominated by one color.
3. **How does histogram equalization work?** It remaps intensity using the cumulative distribution to spread occupied intensity values.
4. **Why use CLAHE?** It equalizes local tiles while clipping histogram peaks to limit noise amplification and global over-contrast.
5. **What is the haze model?** Observed intensity is direct scene radiance attenuated by transmission plus airlight/backscatter: `I=Jt+A(1-t)`.
6. **Why bilateral filtering?** It averages nearby pixels with both spatial and intensity similarity weights, smoothing while retaining many edges; it is slower than Gaussian blur.
7. **Compare interpolation methods.** Nearest copies a sample; bilinear uses a 2x2 neighborhood; bicubic uses a 4x4 neighborhood and often looks smoother but may ring.
8. **What does unsharp masking do?** It adds a scaled high-frequency residual, `I + k(I-G(I))`.
9. **When can PSNR and SSIM be used?** With a suitable, registered reference of matching size; otherwise they do not measure enhancement quality.
10. **What do UCIQE/UIQM measure?** They are no-reference combinations of color, saturation, contrast, and sharpness cues; they are imperfect proxies.
11. **Can super-resolution recover true missing details?** No. Learned SR predicts plausible texture and may hallucinate content; it is not ground-truth recovery.
12. **Why make deep learning optional?** Classical stages run on CPU without weights, while SR adds model, license, dependency, memory, and compute requirements.
