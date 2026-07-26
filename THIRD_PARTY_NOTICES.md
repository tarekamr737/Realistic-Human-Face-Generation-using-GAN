# Third-party notices and release boundaries

FaceForge AI is source code. It does not distribute the FFHQ dataset, trained
weights, generated images, the R3GAN source checkout, or the R3GAN checkpoint.

## FFHQ

The project can train on Flickr-Faces-HQ (FFHQ). Review the official
[FFHQ dataset repository](https://github.com/NVlabs/ffhq-dataset), license,
credits, and restrictions before downloading, training, or distributing a
derivative checkpoint. The dataset bundle is published under CC BY-NC-SA 4.0;
individual images can have separate creator credits and terms. FFHQ is not
intended for face-recognition use.

Do not upload the FFHQ image files, names, metadata, or derived claims of
identity to GitHub. Any public model card must disclose its FFHQ training data,
limitations, and the applicable terms.

## R3GAN comparison

The optional comparison is from [BrownVC R3GAN](https://github.com/brownvc/R3GAN)
and its official [FFHQ-256 model repository](https://huggingface.co/brownvc/R3GAN-FFHQ-256x256).
The local setup script downloads it separately. It is not bundled, relicensed,
or offered as a FaceForge artifact. Keep its attribution, source terms, and
checkpoint conditions intact; do not upload the checkpoint from this project.

## Dependencies

Runtime and developer dependencies are declared in `requirements.txt`. Their
licenses remain their own. Review them before a commercial or redistributed
release.
