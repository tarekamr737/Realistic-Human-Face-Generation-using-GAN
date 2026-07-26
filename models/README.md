# Local model artifacts

This directory is deliberately Git-ignored except for this guide. It may contain:

- `generator_best.pt` — FaceForge DCGAN generator checkpoint.
- `model_metadata.json` — architecture and compatibility metadata.
- `training_history.json`, `evaluation.json`, and `final_samples.png` — local training evidence.
- `r3gan/network-snapshot-final.pkl` — separately downloaded official R3GAN checkpoint.

The FastAPI app is inference-only: it loads exported generators and never trains or fine-tunes in the browser or API.

Do not commit model checkpoints, FFHQ images, generated portraits, or downloaded R3GAN source to this code repository. For a reviewed FaceForge checkpoint, use the Hugging Face model-release template at [`../docs/MODEL_CARD.md`](../docs/MODEL_CARD.md) and publish only the selected DCGAN artifacts. The R3GAN checkpoint must remain at its official upstream location.
