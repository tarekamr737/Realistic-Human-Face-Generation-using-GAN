# Contributing to FaceForge AI

Thanks for improving the inference-only demonstration.

## Local workflow

Keep the repository, virtual environment, cache, FFHQ data, checkpoints, and
temporary outputs on the D: drive. Create a virtual environment, install
`requirements.txt`, and run:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run `node --check app/static/app.js` after editing client JavaScript. Keep the
API inference-only: browser requests must never accept arbitrary checkpoint
paths, start training, fine-tune models, or load untrusted pickle files.

## Pull requests

- Explain the user-facing and safety impact.
- Add or update tests for behavior changes.
- Preserve the synthetic-image label and limitations.
- Do not commit `.env` files, FFHQ data, weights, generated faces, `runs/`,
  `tmp/`, or `third_party/` assets.
- Do not vendor R3GAN. Use its official setup path and honor upstream terms.

Use clear commit messages and keep unrelated formatting changes separate from
functional changes.
