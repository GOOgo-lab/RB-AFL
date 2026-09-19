#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rbafl.evaluation import verify_one
from rbafl.signing import verify_record_signature, verify_center_record
from rbafl.watermark import (
    bit_sha256,
    file_sha256,
    find_record,
    load_registry,
    watermark_image_to_bits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a suspicious/attacked vector dataset")
    parser.add_argument("--vector", required=True)
    parser.add_argument("--watermark", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--record", required=True, help="Registered identity or record_id")
    parser.add_argument("--recovered-image", default="")
    parser.add_argument(
        "--threshold-file",
        required=True,
        help="Frozen calibration/threshold_summary.json",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--public-key", default="", help="Optional Ed25519 public PEM key")
    parser.add_argument("--center-record", default="", help="Center-signed JSON envelope")
    parser.add_argument("--center-public-key", default="", help="Independently trusted center key")
    args = parser.parse_args()
    threshold_path = Path(args.threshold_file).expanduser().resolve()
    lock_path = threshold_path.parent / "threshold_lock.sha256"
    if not lock_path.is_file():
        raise RuntimeError("threshold_lock.sha256 is required beside the threshold file")
    expected_digest = lock_path.read_text(encoding="utf-8").strip().split()[0]
    if hashlib.sha256(threshold_path.read_bytes()).hexdigest() != expected_digest:
        raise RuntimeError("Threshold-file checksum verification failed")
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    if (
        payload.get("split_name") != "calibration"
        or payload.get("test_data_used_for_threshold_selection") is not False
        or payload.get("selection_policy") != "predeclared_fixed"
    ):
        raise RuntimeError("Threshold artifact is not the frozen calibration-only protocol output")
    threshold = float(payload["selected_threshold"])
    if threshold != 0.75:
        raise ValueError("A valid calibrated NC threshold in [0,1] is required")
    registry = load_registry(args.registry)
    record = find_record(registry, args.record)
    bit_length = int(record["bit_length"])
    watermark_bits, _, _ = watermark_image_to_bits(args.watermark, bit_length)
    if bit_length != 256 or int(watermark_bits.sum()) != 45:
        raise ValueError("Eligible watermark must contain exactly 45 one bits and 211 zero bits")
    watermark_digest = bit_sha256(watermark_bits)
    if payload.get("watermark_sha256") != watermark_digest:
        raise RuntimeError("Threshold artifact was calibrated with a different watermark")
    if record.get("copyright_watermark_sha256") != watermark_digest:
        raise RuntimeError("Registry record uses a different watermark")
    checkpoint_digest = file_sha256(args.checkpoint)
    checkpoint_hashes = dict(payload.get("checkpoint_sha256_by_seed", {}))
    if checkpoint_digest not in set(str(value) for value in checkpoint_hashes.values()):
        raise RuntimeError("Checkpoint is not one of the E5 models used for threshold calibration")
    if record.get("checkpoint_sha256") != checkpoint_digest:
        raise RuntimeError("Registry record uses a different checkpoint")
    if record.get("threshold_mode") != payload.get("quantization"):
        raise RuntimeError("Registry quantization differs from the frozen protocol")
    if args.center_record:
        if not args.public_key or not args.center_public_key:
            raise ValueError("Center mode needs both user and trusted center public keys")
        envelope = json.loads(Path(args.center_record).read_text(encoding="utf-8"))
        verify_center_record(envelope, args.center_public_key, args.public_key)
        if envelope["record"] != record:
            raise ValueError("Center-signed record differs from registry record")
    elif args.public_key:
        verify_record_signature(record, args.public_key)
    result = verify_one(
        args.vector,
        args.watermark,
        args.checkpoint,
        args.registry,
        args.record,
        output_recovered_image=args.recovered_image or None,
        nc_threshold=float(threshold),
        device=args.device,
    )
    result["authentication_level"] = "center_and_user_signature" if args.center_record else ("user_signature_only" if args.public_key else "content_match_only")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
