#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from rbafl.evaluation import register_one
from rbafl.signing import sign_record
from rbafl.watermark import load_registry, save_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Register Z = W XOR B from a clean vector dataset")
    parser.add_argument("--vector", required=True)
    parser.add_argument("--watermark", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--identity", default="")
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--density-sigma", type=float, default=3.0)
    parser.add_argument("--bit-length", type=int, default=256)
    parser.add_argument(
        "--threshold-mode",
        choices=["median", "mean", "zero"],
        default="median",
    )
    parser.add_argument("--private-key", default="", help="Optional Ed25519 private PEM key")
    parser.add_argument("--signer-id", default="", help="Required when --private-key is used")
    parser.add_argument("--owner-id", default="")
    parser.add_argument("--watermark-reference", default="")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    record = register_one(
        args.vector,
        args.watermark,
        args.checkpoint,
        args.registry,
        identity=args.identity or None,
        grid_size=args.grid_size,
        density_sigma=args.density_sigma,
        bit_length=args.bit_length,
        threshold_mode=args.threshold_mode,
        device=args.device,
    )
    if args.private_key:
        if not args.signer_id:
            raise ValueError("--signer-id is required with --private-key")
        registry = load_registry(args.registry)
        signed = sign_record(record, args.private_key, signer_id=args.signer_id,
                             owner_id=args.owner_id or None, watermark_reference=args.watermark_reference or None)
        registry_records = [
            signed if item.get("record_id") == record.get("record_id") else item
            for item in registry.get("records", [])
        ]
        save_registry(args.registry, registry_records)
        record = signed
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
