"""Optional Ed25519 signing for registry records.

Signatures authenticate who submitted a record and detect later tampering.  They
do not prove ownership of public geographic data; provenance documents and an
independent registration-time service remain necessary.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Ed25519 support requires the optional 'cryptography' package"
        ) from exc
    return serialization, Ed25519PrivateKey, Ed25519PublicKey


def generate_keypair(private_key_path: str | Path, public_key_path: str | Path) -> None:
    serialization, Ed25519PrivateKey, _ = _crypto()
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_path = Path(private_key_path).expanduser().resolve()
    public_path = Path(public_key_path).expanduser().resolve()
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    try:
        private_path.chmod(0o600)
    except OSError:
        pass
    public_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def _canonical_payload(record: Mapping[str, Any]) -> bytes:
    work = copy.deepcopy(dict(record))
    signature = dict(work.get("registration_signature", {}))
    signature.pop("value_b64", None)
    work["registration_signature"] = signature
    return json.dumps(
        work,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sign_record(
    record: Mapping[str, Any],
    private_key_path: str | Path,
    *,
    signer_id: str,
    owner_id: str | None = None,
    watermark_reference: str | None = None,
) -> dict[str, Any]:
    serialization, _, _ = _crypto()
    private_key = serialization.load_pem_private_key(
        Path(private_key_path).expanduser().read_bytes(), password=None
    )
    public_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signed = copy.deepcopy(dict(record))
    if not signer_id:
        raise ValueError("Registrant identity is required")
    signed["registrant_id"] = str(signer_id)
    signed["owner_id"] = str(owner_id or signer_id)
    signed["watermark_reference"] = str(watermark_reference or "sha256:" + str(record.get("copyright_watermark_sha256", "")))
    signed["model_reference"] = "sha256:" + str(record.get("checkpoint_sha256", ""))
    signed["config_reference"] = "sha256:" + str(record.get("config_sha256", ""))
    signed["user_public_key_der_b64"] = base64.b64encode(public_der).decode("ascii")
    signed["registration_signature"] = {
        "algorithm": "Ed25519",
        "payload_encoding": "canonical-json-sha256-v1.1",
        "signer_id": str(signer_id),
        "signed_utc": datetime.now(timezone.utc).isoformat(),
        "public_key_sha256": hashlib.sha256(public_der).hexdigest(),
    }
    signature = private_key.sign(hashlib.sha256(_canonical_payload(signed)).digest())
    signed["registration_signature"]["value_b64"] = base64.b64encode(signature).decode("ascii")
    return signed


def verify_record_signature(record: Mapping[str, Any], public_key_path: str | Path) -> bool:
    serialization, _, _ = _crypto()
    signature = dict(record.get("registration_signature", {}))
    if signature.get("algorithm") != "Ed25519" or not signature.get("value_b64"):
        raise ValueError("Registry record does not contain an Ed25519 signature")
    if signature.get("payload_encoding") != "canonical-json-sha256-v1.1":
        raise ValueError("Unsupported signature encoding")
    public_key = serialization.load_pem_public_key(
        Path(public_key_path).expanduser().read_bytes()
    )
    public_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    fingerprint = hashlib.sha256(public_der).hexdigest()
    if fingerprint != signature.get("public_key_sha256"):
        raise ValueError("Public key fingerprint does not match the signed record")
    public_key.verify(
        base64.b64decode(str(signature["value_b64"]), validate=True),
        hashlib.sha256(_canonical_payload(record)).digest(),
    )
    return True



def issue_center_record(record, user_public_key_path, center_private_key_path, *,
                        center_id, certificate_reference, timestamp_utc=None):
    """Eq. 19 envelope. Call only after external identity/provenance/conflict review.

    This local reference implementation signs a supplied/current UTC timestamp;
    it is not an independently operated IPR center or trusted timestamp service.
    """
    verify_record_signature(record, user_public_key_path)
    serialization, _, _ = _crypto()
    private_key = serialization.load_pem_private_key(Path(center_private_key_path).read_bytes(), password=None)
    public_der = private_key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    envelope = {"format": "rbafl_center_record_v1.1", "record": copy.deepcopy(dict(record)),
                "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
                "center_id": str(center_id), "certificate_reference": str(certificate_reference),
                "center_public_key_sha256": hashlib.sha256(public_der).hexdigest()}
    if not center_id or not certificate_reference:
        raise ValueError("Center identity and certificate reference are required")
    _validate_center_time(envelope["timestamp_utc"])
    envelope["center_signature_b64"] = base64.b64encode(private_key.sign(_center_payload(envelope))).decode("ascii")
    return envelope


def _center_payload(envelope):
    unsigned = {k: v for k, v in envelope.items() if k != "center_signature_b64"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _validate_center_time(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("Timestamp must explicitly specify UTC")


def verify_center_record(envelope, center_public_key_path, user_public_key_path):
    serialization, _, _ = _crypto()
    if envelope.get("format") != "rbafl_center_record_v1.1":
        raise ValueError("Unsupported center record")
    public = serialization.load_pem_public_key(Path(center_public_key_path).read_bytes())
    der = public.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    if hashlib.sha256(der).hexdigest() != envelope.get("center_public_key_sha256"):
        raise ValueError("Center trust anchor mismatch")
    public.verify(base64.b64decode(envelope["center_signature_b64"], validate=True), _center_payload(envelope))
    _validate_center_time(envelope["timestamp_utc"])
    verify_record_signature(envelope["record"], user_public_key_path)
    return True
