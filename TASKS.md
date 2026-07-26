# TASKS.md

## Project
Build a web app that trains and demonstrates a GAN capable of generating realistic synthetic human faces from random latent vectors.

## 1. Project Setup
- [x] Create a clean Python project structure.
- [x] Add dependencies for TensorFlow or PyTorch, image processing, plotting, and the web framework.
- [x] Add configuration for dataset paths, image size, latent dimension, epochs, batch size, and checkpoints.

## 2. Dataset Pipeline
- [x] Load a human-face dataset such as CelebA or FFHQ.
- [x] Detect or center-crop faces when required.
- [x] Resize images to the selected training resolution.
- [x] Normalize pixel values to the model output range.
- [x] Build shuffled, batched, and prefetched data loaders.
- [x] Add a dataset preview and validation checks.

## 3. GAN Architecture
- [x] Build the Generator to transform random latent vectors into face images.
- [x] Build the Discriminator to classify real and generated images.
- [x] Use stable layers, activations, initialization, and normalization.
- [x] Add model summaries and shape tests.

## 4. Training Pipeline
- [x] Implement adversarial training for both models.
- [x] Generate random latent vectors during training.
- [x] Track Generator and Discriminator losses.
- [x] Save checkpoints and fixed-noise sample grids.
- [x] Support resume-from-checkpoint.
- [x] Add protections for unstable training, NaNs, and mode collapse indicators.
- [x] Add a local CUDA training CLI with AMP, terminal tqdm progress, atomic per-epoch checkpoints, and resume support.

## 5. Evaluation
- [x] Compare generated samples across training epochs.
- [x] Plot Generator and Discriminator loss curves.
- [ ] Calculate FID where practical.
- [x] Review diversity, realism, artifacts, and possible mode collapse.
- [x] Save final sample grids and evaluation results.

## 6. Web App
- [x] Build pages for Overview, Generate Faces, Training Progress, Model Evaluation, and Project Details.
- [x] Allow users to select the number of faces and random seed.
- [x] Generate faces using the trained Generator.
- [x] Display face grids with regenerate and download actions.
- [x] Show checkpoints, training samples, losses, and evaluation metrics.
- [x] Add loading, error, empty, and unavailable-model states.
- [x] Add an inference-only, same-seed side-by-side DCGAN/R3GAN comparison API and UI.
- [x] Install and live-validate the official R3GAN FFHQ-256 checkpoint on this Windows GPU.

## 7. UI Quality
- [x] Use the **Impeccable skill** to design and implement the complete UI.
- [x] Create a polished AI product interface rather than a basic notebook-style dashboard.
- [x] Ensure responsive layouts, accessible contrast, clear hierarchy, consistent spacing, and reusable components.
- [x] Avoid exposing unsafe or misleading identity claims; clearly label all faces as synthetic.

## 8. UI Evaluation
- [x] Use **Chrome DevTools MCP** to evaluate the finished web app.
- [ ] Test responsiveness, accessibility, keyboard navigation, console errors, network failures, layout shifts, and performance.
- [ ] Verify all generation controls, loading states, downloads, navigation, and error handling.
- [ ] Fix all critical and high-priority issues before completion.

## 9. Testing
- [x] Add tests for preprocessing, model shapes, checkpoint loading, inference, and invalid inputs.
- [x] Run a small end-to-end training smoke test.
- [x] Verify deterministic generation when using a fixed seed.

## 10. Documentation and Delivery
- [x] Add setup, dataset, training, evaluation, and run instructions.
- [x] Document architecture choices and limitations.
- [x] Prepare a GitHub-safe source release with CI, license boundaries, ignored data/weights, and a Hugging Face model-card template.
- [ ] Include sample outputs and screenshots.
- [x] Deliver the trained model, source code, evaluation results, and final web app.

## Definition of Done
- [x] The GAN trains without pipeline failures.
- [x] The Generator produces recognizable and varied synthetic faces.
- [x] Training samples and losses are saved and viewable.
- [x] The web app supports reliable face generation from random noise.
- [ ] The UI is built with Impeccable and passes Chrome DevTools MCP evaluation.
