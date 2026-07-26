---
license: other
language:
- en
tags:
- image-generation
- gan
- pytorch
- synthetic-faces
- ffhq
---

# FaceForge DCGAN-64 — FFHQ

> Replace bracketed fields and remove this note before publishing a model.

## Model description

FaceForge DCGAN-64 is an educational, unconditional PyTorch DCGAN that creates
**64 × 64 synthetic face-like images** from 128-dimensional random latent
vectors. It is the project baseline, not a production portrait model, identity
system, or image editor.

- **Developed by:** [your name or organization]
- **Model type:** Unconditional DCGAN generator
- **Language:** English documentation; no text input
- **License / terms:** [state a model-specific license consistent with FFHQ's terms]
- **Code:** [GitHub repository URL]
- **Checkpoint revision:** [Git tag or commit SHA]

## Files

| File | Purpose |
| --- | --- |
| `generator_best.pt` | Generator state dictionary; load only through the matching FaceForge code. |
| `model_metadata.json` | Architecture contract, preprocessing settings, and training information. |

Do not upload FFHQ image files, an R3GAN checkpoint, generated samples that
could be confused with real people, or arbitrary Python pickles to this model
repository.

## Training

- **Dataset:** FFHQ, obtained separately under its applicable terms.
- **Training resolution:** 64 × 64 RGB; source files were 512 × 512 and were
  resized during loading.
- **Architecture:** DCGAN generator with a 128-D latent input. Use the exact
  `latent_dim` and `feature_maps` stored in `model_metadata.json`.
- **Objective:** BCE-with-logits adversarial objective with Adam optimizers.
- **Run:** [epochs, batch size, hardware, wall-clock time, date]
- **Preprocessing:** [confirm resize, crop policy, and normalization from metadata]

## Evaluation

Record only measured results from the exported `evaluation.json`:

| Metric | Result | Notes |
| --- | --- | --- |
| Generator loss | [value] | Training diagnostic, not a quality score. |
| Discriminator loss | [value] | Training diagnostic, not a quality score. |
| Diversity diagnostic | [value] | Pixel-space heuristic, not a realism metric. |
| FID | Not measured | Do not replace with an estimate. |

Inspect fixed-noise grids for artifacts and collapse. Losses, a small sample
grid, and pixel diversity do not demonstrate realism, fairness, privacy, or
fitness for a real-world deployment.

## Intended use

Suitable only for education, controlled GAN demonstrations, and comparisons
with stronger pretrained models. All outputs must be visibly labelled
synthetic.

## Out-of-scope use

Do not use this model for face recognition, identity inference, impersonation,
deception, profiling, biometric decisions, or claims that a generated image
depicts a real person. It should not be used as evidence of fairness, privacy,
or demographic representation.

## Limitations and bias

This is a low-resolution baseline trained on a limited face dataset. Outputs
can be blurred, artifacted, repetitive, or unrepresentative. The model can
inherit dataset bias and may generate people-like images with unintended visual
similarities. It has not received a demographic, safety, privacy, or benchmark
evaluation.

## Citation and acknowledgements

Credit FFHQ according to its official dataset documentation and terms. For the
optional visual reference in the FaceForge app, cite BrownVC R3GAN separately;
it is not this model and is not redistributed here.
