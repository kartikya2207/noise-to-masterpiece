# Noise → Masterpiece: Building Stable Diffusion from First Principles

My attempt to build a text-to-image diffusion model from scratch — VAE, then DDPM, then CLIP
conditioning — writing every component by hand instead of calling a pre-built pipeline.

> **Viewing the notebooks:** GitHub's preview can't render the larger notebooks (the Week-4 one
> embeds all its figures and is ~1 MB, so you'll see *"Unable to render code block"*). Use nbviewer
> to see them fully rendered:
> [all notebooks](https://nbviewer.org/github/kartikya2207/noise-to-masterpiece/tree/main/notebooks/) ·
> [Week 4](https://nbviewer.org/github/kartikya2207/noise-to-masterpiece/blob/main/notebooks/Week4_ConvVAE_MNIST.ipynb) ·
> [Week 3](https://nbviewer.org/github/kartikya2207/noise-to-masterpiece/blob/main/notebooks/Week3_Milestone1_VAE_2D.ipynb) ·
> [Week 2](https://nbviewer.org/github/kartikya2207/noise-to-masterpiece/blob/main/notebooks/Week2_VAE_Loss.ipynb)

## What it is

A week-by-week course project ("Noise → Masterpiece") whose end goal is a small Stable-Diffusion-style
model. Stable Diffusion runs diffusion in the latent space of a VAE, so the first phase is building
and understanding the VAE properly.

**Current status (mid-term):** Phase 1, the Variational Autoencoder, is done through Week 4 — the
loss, a 2D proof-of-concept, and a convolutional VAE on MNIST are implemented, trained and checked.
The diffusion phases are still ahead (see roadmap).

## What I did

**Weeks 0–1 — Foundations.** PyTorch basics (tensors, autograd, training loops) and the probability
the project leans on: Gaussians, Bayes' rule, KL divergence.

**Week 2 — The VAE loss, from scratch** (`Week2_VAE_Loss.ipynb`).
- Derived and implemented the closed-form KL between `N(μ, σ²)` and `N(0, I)`, and asserted it matches
  `torch.distributions.kl_divergence`.
- Implemented the reparameterization trick (`z = μ + σ·ε`) and checked that 10,000 samples recover
  the target mean and std.
- Combined them into the ELBO (BCE reconstruction + KL), plus written answers on why `log p(x)` is
  intractable, what happens without the KL term, and how gradients flow.

**Week 3 — Milestone 1: a VAE on 2D "two moons"** (`Week3_Milestone1_VAE_2D.ipynb`).
- Built the required `LinearVAE` baseline and showed *why* it fails: generation collapses to one
  elliptical blob, because a linear map of a Gaussian is always a Gaussian.
- Swapped in an MLP encoder/decoder (same skeleton, same loss), which captures both crescents.
- With full-strength KL (`β = 1`) the two moons collapsed onto one curve; lowering the KL weight to
  `β = 0.15` let the latent keep enough information to recover both modes.
- Deliverables: reconstructions, latent map, generation from `N(0, I)`, and latent interpolation
  from one moon to the other.

**Week 4 — Convolutional VAE on MNIST** (`Week4_ConvVAE_MNIST.ipynb`).
- Same skeleton with `Linear → Conv/ConvTranspose` (`28→14→7` and back) and `MSE → BCE`, since
  MNIST pixels are in `[0, 1]`.
- Kept the latent at 2D on purpose so I could draw the latent map and the decoded manifold grid.
- Trained 10 epochs on a Colab GPU; loss per image fell from 207.91 to 159.36. Outputs:
  reconstructions, a test-set latent map coloured by digit, the 20×20 manifold grid of digits
  morphing into each other, and 64 digits generated from pure noise.

## What I learnt

- **Verify the math before trusting the model.** Checking my KL against `torch.distributions` and
  checking reparameterized sample statistics took a few lines and meant that when training later
  looked odd, I knew the loss wasn't the problem.
- **"It doesn't work" can be the result.** The LinearVAE's single blob wasn't a bug to hide — it's
  a clean demonstration that capacity, not training, was the limit. Explaining the failure taught me
  more than the working model did.
- **β is a real trade-off.** Strong KL gives a tidy, sampleable latent but can squash information
  and merge modes; weak KL reconstructs better but the latent gets less regular. Seeing it collapse
  at `β = 1` made the β-VAE idea concrete.
- **Small details matter:** standardizing the 2D data to match the `N(0, I)` prior gave cleaner
  samples; using `reduction='sum'` on *both* loss terms stops one from silently dominating; and the
  likelihood choice (MSE for real values, BCE for `[0, 1]` pixels) follows from the data.
- **A 2D latent is a teaching tool, not a quality setting.** It gives great pictures but blurry
  reconstructions; a larger latent would be sharper but impossible to plot directly.
- **Practical:** notebooks with embedded figures quickly exceed GitHub's render limit — hence nbviewer.

## Repo structure

```
notebooks/
  Week2_VAE_Loss.ipynb            ELBO from scratch (KL, reparameterization, BCE) + reflections
  Week3_Milestone1_VAE_2D.ipynb   Linear vs MLP VAE on 2D moons, β choice, interpolation
  Week4_ConvVAE_MNIST.ipynb       Convolutional VAE on MNIST
requirements.txt
```

## How to run

Each notebook runs top to bottom in Google Colab (easiest — Week 4 picks up a GPU automatically;
MNIST downloads on first run into `data/`, which is git-ignored). Locally:

```bash
pip install -r requirements.txt
jupyter notebook
```

## Roadmap

| Phase | Weeks | Status |
|-------|-------|--------|
| VAE (loss → 2D → MNIST) | 2–4 | done |
| VAE on CelebA faces + latent surgery | 5–6 | next |
| Forward diffusion & noise schedules | 7–8 | upcoming |
| UNet denoiser + unconditional DDPM | 9–10 | upcoming |
| Class-conditional generation + CFG | 11 | upcoming |
| Text-to-image with CLIP | 12 | upcoming |
