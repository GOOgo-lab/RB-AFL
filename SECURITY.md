# Security notes

PyTorch checkpoints and cached Python pickle objects can execute code when loaded. Use only artifacts produced locally or downloaded from a trusted release whose SHA-256 has been verified. Never run untrusted checkpoints, attack caches, registries, shell scripts, or configuration files with elevated privileges.

Ed25519 private keys must be generated and stored outside the repository. The `.gitignore` blocks common key extensions, but users remain responsible for secret management and key rotation.

To report a vulnerability, contact the repository maintainers privately before opening a public issue.

