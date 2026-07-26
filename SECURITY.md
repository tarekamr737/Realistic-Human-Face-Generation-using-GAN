# Security policy

Please do not publish security vulnerabilities, model-access bypasses, exposed
credentials, or unsafe pickle-loading issues in a public issue. Report them
privately to the repository owner with a minimal reproduction and affected
version.

The service is designed to load only configured local checkpoints. Treat every
PyTorch or Python-pickle checkpoint from an untrusted source as unsafe. Never
add API routes that let a browser supply a filesystem path, model URL, or
serialized object to load.

Rotate any accidentally exposed credential immediately, remove it from the
repository history before publishing, and update the affected service.
