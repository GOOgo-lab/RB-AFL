from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from PIL import Image


def as_bits(value: Sequence[int] | np.ndarray, length: int | None = None) -> np.ndarray:
    bits = (np.asarray(value).reshape(-1) != 0).astype(np.uint8)
    if length is not None and bits.size != int(length):
        raise ValueError(f"Expected {length} bits, received {bits.size}")
    return bits




def bit_distribution_stats(bits: Sequence[int]) -> Dict[str, float | int | bool]:
    """Return publication-facing bit-balance / degeneracy diagnostics.

    The binary entropy is in bits and reaches 1.0 for a perfectly balanced
    sequence.  These statistics are descriptive; they are not used to alter
    registration or verification decisions.
    """
    value = as_bits(bits)
    n = int(value.size)
    ones = int(value.sum())
    zeros = int(n - ones)
    one_ratio = float(ones / n) if n else 0.0
    if one_ratio <= 0.0 or one_ratio >= 1.0:
        entropy = 0.0
    else:
        entropy = float(
            -one_ratio * np.log2(one_ratio)
            - (1.0 - one_ratio) * np.log2(1.0 - one_ratio)
        )
    return {
        "bit_length": n,
        "zero_bits": zeros,
        "one_bits": ones,
        "one_ratio": one_ratio,
        "binary_entropy_bits": entropy,
        "hamming_weight": ones,
        "is_all_zero": bool(ones == 0),
        "is_all_one": bool(zeros == 0),
        "minority_bit_ratio": float(min(one_ratio, 1.0 - one_ratio)) if n else 0.0,
    }

def xor_bits(a: Sequence[int] | np.ndarray, b: Sequence[int] | np.ndarray) -> np.ndarray:
    left = as_bits(a)
    right = as_bits(b, left.size)
    return np.bitwise_xor(left, right).astype(np.uint8)


def ber_score(reference: Sequence[int], recovered: Sequence[int]) -> float:
    a = as_bits(reference)
    b = as_bits(recovered, a.size)
    return float(np.mean(a != b))


def nc_score(reference: Sequence[int], recovered: Sequence[int]) -> float:
    a = as_bits(reference).astype(np.float64)
    b = as_bits(recovered, a.size).astype(np.float64)
    if not np.any(a):
        raise ValueError("Reference watermark must have nonzero L2 norm")
    if np.array_equal(a, b):
        return 1.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-15:
        return 0.0
    value = float(np.dot(a, b) / denominator)
    return float(np.clip(value, 0.0, 1.0))


def bit_accuracy(reference: Sequence[int], recovered: Sequence[int]) -> float:
    return 1.0 - ber_score(reference, recovered)


def bit_sha256(bits: Sequence[int]) -> str:
    value = as_bits(bits)
    payload = np.packbits(value, bitorder="big").tobytes()
    return hashlib.sha256(payload + str(value.size).encode("ascii")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_dataset_sha256(path: str | Path) -> str:
    """Path-independent digest of a vector file and Shapefile sidecars."""

    source = Path(path).expanduser().resolve()
    files = [source]
    if source.suffix.lower() == ".shp":
        files.extend(
            candidate
            for extension in (".shx", ".dbf", ".prj", ".cpg")
            if (candidate := source.with_suffix(extension)).is_file()
        )
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.suffix.lower().encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def encode_bits(bits: Sequence[int]) -> str:
    return base64.b64encode(np.packbits(as_bits(bits), bitorder="big").tobytes()).decode("ascii")


def decode_bits(payload: str, bit_length: int) -> np.ndarray:
    packed = np.frombuffer(base64.b64decode(payload.encode("ascii")), dtype=np.uint8)
    return np.unpackbits(packed, bitorder="big")[: int(bit_length)].astype(np.uint8)


def factor_grid_shape(bit_length: int) -> Tuple[int, int]:
    width = int(np.ceil(np.sqrt(bit_length)))
    while width > 1 and bit_length % width:
        width -= 1
    height = int(np.ceil(bit_length / width))
    return width, height


def watermark_image_to_bits(path: str | Path, bit_length: int) -> Tuple[np.ndarray, int, int]:
    width, height = factor_grid_shape(bit_length)
    image = Image.open(path).convert("L").resize((width, height), Image.Resampling.NEAREST)
    array = np.asarray(image, dtype=np.uint8)
    bits = (array.reshape(-1) >= 128).astype(np.uint8)[:bit_length]
    if bits.size < bit_length:
        bits = np.pad(bits, (0, bit_length - bits.size), mode="wrap")
    return bits, width, height


def save_bits_image(bits: Sequence[int], width: int, height: int, path: str | Path) -> None:
    value = as_bits(bits)
    canvas = np.zeros(width * height, dtype=np.uint8)
    canvas[: min(canvas.size, value.size)] = value[: canvas.size] * 255
    Image.fromarray(canvas.reshape(height, width), mode="L").save(path)


def create_registry_record(
    identity: str,
    source_path: str | Path,
    checkpoint_path: str | Path,
    watermark_bits: Sequence[int],
    feature_bits: Sequence[int],
    experiment: Dict[str, object],
    channel_config: Dict[str, object],
    threshold_mode: str,
    timing: Dict[str, float] | None = None,
    space: Dict[str, int] | None = None,
) -> Dict[str, object]:
    watermark = as_bits(watermark_bits)
    feature = as_bits(feature_bits, watermark.size)
    if not np.any(watermark):
        raise ValueError("Reference watermark must be nonzero")
    zero = xor_bits(watermark, feature)
    record_id = hashlib.sha256(
        f"{identity}|{file_sha256(checkpoint_path)}|{bit_sha256(zero)}".encode("utf-8")
    ).hexdigest()[:24]
    from .fields import CHANNEL_SCHEMA
    channel_config = {**channel_config, "channel_schema": CHANNEL_SCHEMA}
    configuration = {"channel_config": channel_config, "threshold_mode": threshold_mode, "bit_length": int(watermark.size)}
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    return {
        "config_sha256": config_hash,
        "record_id": record_id,
        "identity": identity,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name_at_registration": Path(source_path).name,
        "source_file_sha256": file_sha256(source_path),
        "source_dataset_sha256": vector_dataset_sha256(source_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_name": Path(checkpoint_path).name,
        "experiment": experiment,
        "channel_config": channel_config,
        "threshold_mode": threshold_mode,
        "bit_length": int(watermark.size),
        "zero_watermark_bits_b64": encode_bits(zero),
        "zero_watermark_sha256": bit_sha256(zero),
        "copyright_watermark_sha256": bit_sha256(watermark),
        "copyright_watermark_stats": bit_distribution_stats(watermark),
        "feature_bit_stats": bit_distribution_stats(feature),
        "zero_watermark_stats": bit_distribution_stats(zero),
        "timing": timing or {},
        "space": space or {},
    }


def save_registry(path: str | Path, records: Iterable[Dict[str, object]]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "rbafl_zero_watermark_registry_v1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": list(records),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def load_registry(path: str | Path) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "rbafl_zero_watermark_registry_v1.0.0":
        raise ValueError("Unsupported registry format")
    return payload


def find_record(registry: Dict[str, object], identity_or_id: str) -> Dict[str, object]:
    matches = [
        record
        for record in registry.get("records", [])
        if record.get("identity") == identity_or_id or record.get("record_id") == identity_or_id
    ]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one registry record for '{identity_or_id}', found {len(matches)}")
    return matches[0]


def recover_watermark(record: Dict[str, object], feature_bits: Sequence[int]) -> np.ndarray:
    validate_record_integrity(record)
    length = int(record["bit_length"])
    zero = decode_bits(str(record["zero_watermark_bits_b64"]), length)
    return xor_bits(zero, as_bits(feature_bits, length))


def verify_recovered(
    record: Dict[str, object],
    recovered_bits: Sequence[int],
    reference_watermark_bits: Sequence[int],
    nc_threshold: float,
) -> Dict[str, object]:
    if not np.isfinite(nc_threshold) or not 0 <= nc_threshold <= 1:
        raise ValueError("NC threshold must be in [0,1]")
    reference = as_bits(reference_watermark_bits, int(record["bit_length"]))
    if bit_sha256(reference) != record["copyright_watermark_sha256"]:
        raise ValueError("The supplied copyright watermark does not match the registered watermark hash")
    recovered = as_bits(recovered_bits, reference.size)
    nc = nc_score(reference, recovered)
    ber = ber_score(reference, recovered)
    return {
        "nc": nc,
        "ber": ber,
        "bit_accuracy": 1.0 - ber,
        "threshold": float(nc_threshold),
        "passed": bool(np.any(recovered) and nc >= nc_threshold),
    }


def validate_record_integrity(record: Dict[str, object]) -> None:
    """Validate Hcfg and HZ before reconstruction (authenticity also needs signatures)."""
    from .fields import CHANNEL_SCHEMA
    config = record["channel_config"]
    if config.get("channel_schema") != CHANNEL_SCHEMA:
        raise ValueError("Legacy or incompatible geometric fields; rebuild registry and retrain")
    payload = {"channel_config": config, "threshold_mode": record["threshold_mode"],
               "bit_length": int(record["bit_length"])}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    if digest != record.get("config_sha256"):
        raise ValueError("Registered configuration hash mismatch")
    zero = decode_bits(str(record["zero_watermark_bits_b64"]), int(record["bit_length"]))
    if len(zero) != int(record["bit_length"]) or bit_sha256(zero) != record.get("zero_watermark_sha256"):
        raise ValueError("Registered zero-watermark hash mismatch")
