# FaceForge AI

<img width="1920" height="1080" alt="Screenshot 2026-07-26 164044" src="https://github.com/user-attachments/assets/2cd5a725-25a4-4241-b991-b0e9b45e3943" />

<img width="1920" height="1080" alt="Screenshot 2026-07-26 164118" src="https://github.com/user-attachments/assets/f279bdfb-9d68-4b0a-90a0-97f008b1dbbf" />

<img width="1920" height="1080" alt="Screenshot 2026-07-26 164139" src="https://github.com/user-attachments/assets/d6eb91ca-7e06-4b88-918a-af85e97c9acb" />


FaceForge AI is an educational GAN project for synthetic face-like image generation.
It pairs reproducible FFHQ training workflows with a local, **inference-only** web application.

The application compares two local generators:

- **FaceForge DCGAN-64** — this project's 64 × 64 PyTorch baseline.
- **R3GAN FFHQ-256** — an optional official pretrained reference from BrownVC.

> **Safety:** Every output is synthetic. Do not present generated images as real people
> or use either model for identity, biometric, impersonation, profiling, or decision-making purposes.

## Highlights

- Inference-only FastAPI application; no browser-triggered training or fine-tuning.
- Reproducible seed-based generation of 1, 4, 8, or 16 images.
- Side-by-side DCGAN/R3GAN comparison with a shared numeric seed.
- Clearly labelled synthetic output and image/grid downloads.
- GPU-aware DCGAN training with AMP, `tqdm`, checkpoints, samples, and resume.
- Complete Google Colab workflow for validation, training, and export.

## Project structure

| Path | Purpose |
| --- | --- |
| [`app/`](app) | FastAPI service and responsive inference interface. |
| [`faceforge/`](faceforge) | Models, data, training, inference, export, and R3GAN integration. |
| [`notebooks/faceforge_ffhq_training.ipynb`](notebooks/faceforge_ffhq_training.ipynb) | GPU Colab workflow. |
| [`scripts/train_local.py`](scripts/train_local.py) | Local 64 × 64 DCGAN training CLI. |
| [`scripts/setup_local_gpu.ps1`](scripts/setup_local_gpu.ps1) | CUDA PyTorch setup. |
| [`scripts/setup_r3gan_comparison.ps1`](scripts/setup_r3gan_comparison.ps1) | R3GAN installer. |
| [`models/`](models) | Local-only artifacts; model weights are never committed. |
| [`docs/`](docs) | Model-card template and release checklist. |

## Requirements

- Keep the repository, virtual environment, dataset, artifacts, caches, and runs on **D:**.
- Python 3.12 or newer.
- NVIDIA GPU and CUDA PyTorch for practical local training.
- FFHQ data only when training, obtained under its [official terms](https://github.com/NVlabs/ffhq-dataset).

## Local inference

### 1. Create the environment on D:

```powershell
New-Item -ItemType Directory -Force -Path .\tmp\pip-cache | Out-Null
$env:PIP_CACHE_DIR = "$PWD\tmp\pip-cache"
$env:TEMP = "$PWD\tmp"
$env:TMP = "$PWD\tmp"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
```

The environment variables keep package downloads and temporary files on D: rather than filling C:.

### 2. Add the reviewed FaceForge export

Extract these artifacts into [`models/`](models):

```text
models/
├── generator_best.pt
├── model_metadata.json
├── training_history.json        # optional training view
├── evaluation.json              # optional evaluation view
└── final_samples.png            # optional evidence
```

The generator must match `model_metadata.json`. The app validates that contract and never accepts a browser-supplied checkpoint path.

### 3. Start the app

```powershell
$env:PIP_CACHE_DIR = "$PWD\tmp\pip-cache"
$env:TEMP = "$PWD\tmp"
$env:TMP = "$PWD\tmp"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). A missing or incompatible checkpoint creates an explicit unavailable state.

For the same application version, checkpoint, device type, seed, image count, and latent range, DCGAN generation is deterministic. CPU and GPU output can differ slightly because of floating-point behavior.

## Optional R3GAN comparison

R3GAN is a separately sourced reference model. Review the [upstream repository](https://github.com/brownvc/R3GAN) before use.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_r3gan_comparison.ps1
```

The installer keeps the official source, checkpoint, and caches on D:

```text
third_party/R3GAN/                         # official source checkout
models/r3gan/network-snapshot-final.pkl    # official FFHQ-256 checkpoint
tmp/r3gan-torch-extensions/                # optional extension cache
tmp/huggingface/                           # download cache
```

The default path uses R3GAN's upstream PyTorch reference operators. A separate CUDA Toolkit and Visual Studio C++ Build Tools are not required. Set `FACEFORGE_R3GAN_CUSTOM_OPS=1` only to build optional faster CUDA extensions.

The same numeric seed is supplied to both models, but this is a reproducible comparison—not a paired benchmark. DCGAN uses a 128-D latent vector and produces 64 × 64 images; R3GAN uses a different latent space and produces 256 × 256 images.

> R3GAN checkpoints are Python pickles. Load only the official checkpoint obtained through the setup script; untrusted pickle files are unsafe.

## Train in Google Colab

Use the complete [training notebook](notebooks/faceforge_ffhq_training.ipynb).

1. Open it in Google Colab and select a GPU runtime.
2. Run the setup cell.
3. Make aligned FFHQ images available at `MyDrive/FaceForge/ffhq/` under the data terms.
4. Run the remaining cells in order to validate data, train, monitor `tqdm`, checkpoint, sample, and export.
5. Review generated grids and metrics before exporting a candidate model.
6. Download `faceforge_model_export.zip` and extract it into local `models/`.

## Train locally on D:

The trainer reads original **512 × 512 FFHQ files** from `data/` and resizes them while loading to train the project's 64 × 64 DCGAN. It is suitable for a 4 GB laptop GPU and is not native 512 × 512 training.

### 1. Install CUDA PyTorch

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_local_gpu.ps1
```

Confirm that the final line reports the NVIDIA GPU.

### 2. Start training

```powershell
.\.venv\Scripts\python.exe .\scripts\train_local.py `
  --dataset .\data `
  --run-dir .\runs\ffhq_dcgan64 `
  --export-dir .\models `
  --epochs 80 `
  --batch-size 64 `
  --workers 4 `
  --snapshot-every 5
```

Nested `tqdm` bars show batch losses and overall epoch progress. `runs\ffhq_dcgan64\checkpoints\last.pt` is atomically updated after each completed epoch.

### 3. Resume and publish a reviewed candidate

```powershell
.\.venv\Scripts\python.exe .\scripts\train_local.py `
  --dataset .\data `
  --run-dir .\runs\ffhq_dcgan64 `
  --export-dir .\models `
  --epochs 80 `
  --batch-size 64 `
  --workers 4 `
  --snapshot-every 5 `
  --resume .\runs\ffhq_dcgan64\checkpoints\last.pt `
  --publish-final
```

Inspect `runs\ffhq_dcgan64\samples\epoch_*.png` before using `--publish-final`. The flag writes the reviewed generator and metadata to local `models/` for the inference app.

## Testing

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --check .\app\static\app.js
```

The suite does not require FFHQ data, an exported DCGAN checkpoint, or R3GAN assets.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Service health status. |
| `GET /api/model` | FaceForge DCGAN availability and metadata. |
| `GET /api/models` | Availability and metadata for both models. |
| `POST /api/generate` | Generate from `count`, `seed`, and `truncation`. |
| `GET /api/training` | Exported training history when available. |
| `GET /api/evaluation` | Exported evaluation data when available. |

The API has no training route and never accepts a model path, URL, or serialized object from a browser request.

## Limitations

FaceForge DCGAN-64 is an instructional baseline. Results can be blurred, artifacted, repetitive, or unrepresentative. Losses and pixel-diversity diagnostics help monitor training but do not prove realism, fairness, privacy, demographic coverage, or deployment readiness.

R3GAN can create sharper images because it is stronger and pretrained at a higher resolution—not because the models are equivalent. Neither system is appropriate for identity-related or high-impact decisions.

## Publishing

Publish the **code** on GitHub and a reviewed **FaceForge DCGAN checkpoint** in a separate Hugging Face model repository. Git intentionally excludes FFHQ images, generated portraits, model weights, R3GAN assets, virtual environments, and caches.

Before release, follow [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) and use [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) as the model repository README. It must contain exact measured results, data provenance, limitations, and terms consistent with FFHQ.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for release boundaries and [`SECURITY.md`](SECURITY.md) for responsible disclosure guidance.

## License

Original source code and documentation are licensed under the [MIT License](LICENSE). This does not grant rights to FFHQ data, trained weights, generated artifacts, R3GAN code, or R3GAN checkpoints; those materials remain subject to their own terms.
