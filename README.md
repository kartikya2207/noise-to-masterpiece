# Noise → Masterpiece: Building Stable Diffusion from First Principles

Implementing a text-to-image diffusion model from scratch — VAE, DDPM, and CLIP conditioning — with no
pre-built pipelines or wrappers. Every component written and trained by hand.

**Mid-term status:** Phase 1 (the Variational Autoencoder) complete through Week 4. The VAE loss, a 2D
proof-of-concept, and a convolutional VAE on MNIST are all implemented, trained, and verified.

---

## Progress (Weeks 0–4)

**Week 0–1 — Foundations.** PyTorch tooling (tensors, autograd, training loops) and the core probability
objects the project relies on: Gaussians, Bayes' rule, and KL divergence.

**Week 2 — The VAE loss, from scratch.** Implemented and *verified* the three pieces of the ELBO:
- closed-form KL divergence between `N(μ, σ²)` and `N(0, I)` — checked against `torch.distributions` (exact match)
- the reparameterization trick — verified sample mean/std recover the target Gaussian
- the full ELBO loss (BCE reconstruction + KL), with reflections on intractability, the KL term, and gradient flow

**Week 3 — Milestone 1: first generative model (2D).** A VAE trained on two interleaving moons.
- built a `LinearVAE` baseline and showed *why* it cannot generate a curved/multi-mode manifold (a linear map of
  a Gaussian is always a Gaussian → generation collapses to one blob)
- upgraded to an MLP VAE that captures both moons; ran a β (KL-weight) sweep and found `β = 0.15` recovers both
  modes where the full-strength KL collapses them
- deliverables: reconstruction, latent-space map, generation from `N(0, I)`, and latent interpolation

**Week 4 — Convolutional VAE on MNIST.** The same VAE skeleton with `Linear → Conv/ConvTranspose` and `MSE → BCE`.
- conv encoder (`28→14→7`) + transpose-conv decoder (`7→14→28`), 2D latent
- deliverables: sharper reconstructions, a latent map of the test set colored by digit, the decoded **latent
  manifold grid** (digits morphing smoothly across latent space), and generation from pure noise

---

## Repository structure

```
.
├── README.md
├── requirements.txt
└── notebooks/
    ├── Week2_VAE_Loss.ipynb           # ELBO from scratch (KL, reparam, BCE loss) + reflections
    ├── Week3_Milestone1_VAE_2D.ipynb  # Linear vs MLP VAE on 2D moons, β sweep, interpolation
    └── Week4_ConvVAE_MNIST.ipynb      # Convolutional VAE on MNIST
```

## Running

Each notebook runs top-to-bottom in Google Colab (preferred — the Week-4 notebook auto-detects a GPU).
For a local run:

```bash
pip install -r requirements.txt
jupyter notebook
```

## Roadmap

| Phase | Weeks | Status |
|-------|-------|--------|
| VAE (loss → 2D → MNIST) | 2–4 | ✅ done |
| VAE on CelebA faces + latent surgery | 5–6 | next |
| Forward diffusion & noise schedules | 7–8 | upcoming |
| UNet denoiser + unconditional DDPM | 9–10 | upcoming |
| Class-conditional generation + CFG | 11 | upcoming |
| Text-to-image with CLIP | 12 | upcoming |
