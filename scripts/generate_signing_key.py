#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rbafl.signing import generate_keypair


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an Ed25519 registry-signing key pair")
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--public-key", required=True)
    args = parser.parse_args()
    generate_keypair(args.private_key, args.public_key)
    print(f"private key: {args.private_key}")
    print(f"public key:  {args.public_key}")
    print("Keep the private key outside the repository.")


if __name__ == "__main__":
    main()

