# Release checklist

## GitHub code repository

- [ ] Confirm `git status --short` has no unexpected files.
- [ ] Confirm `git diff --cached --check` succeeds.
- [ ] Confirm `git ls-files` contains no FFHQ data, checkpoint, generated
      output, virtual environment, cache, `runs/`, or `third_party/` files.
- [ ] Run `python -m unittest discover -s tests -v` in the project virtual
      environment.
- [ ] Run `node --check app/static/app.js`.
- [ ] Read `LICENSE`, `THIRD_PARTY_NOTICES.md`, `SECURITY.md`, and the README.
- [ ] Replace placeholder repository links and release metadata.
- [ ] Create a repository description and topics such as `gan`, `pytorch`,
      `fastapi`, `synthetic-media`, and `ffhq`.

## Hugging Face model repository

- [ ] Confirm the specific checkpoint is reviewed and compatible with the
      current code (`generator_best.pt` plus `model_metadata.json`).
- [ ] Copy `MODEL_CARD.md` to the model repository's `README.md`.
- [ ] Replace every bracketed placeholder with exact training details and
      measured evaluation values; leave unavailable metrics as not measured.
- [ ] State dataset provenance and the model terms consistent with FFHQ.
- [ ] Upload only the intended checkpoint and metadata, not FFHQ data, R3GAN,
      caches, local paths, credentials, or unreviewed generated outputs.
- [ ] Test a fresh clone/download with the FaceForge inference app.

## Do not release

- FFHQ images, metadata, or names.
- The local R3GAN checkout or its downloaded `.pkl` checkpoint.
- `.env` files, Hugging Face access tokens, or browser screenshots with them.
- Untrusted `torch.load` or Python-pickle files.
