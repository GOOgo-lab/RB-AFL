"""Issue a local reference center envelope after external authorization review."""
import argparse
import json
from pathlib import Path
from rbafl.signing import issue_center_record
from rbafl.watermark import find_record, load_registry, validate_record_integrity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("registry", "record", "user-public-key", "center-private-key", "center-id", "certificate-reference", "output"):
        p.add_argument("--" + field, required=True)
    args = p.parse_args()
    record = find_record(load_registry(args.registry), args.record)
    validate_record_integrity(record)
    envelope = issue_center_record(record, args.user_public_key, args.center_private_key,
                                   center_id=args.center_id, certificate_reference=args.certificate_reference)
    Path(args.output).write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
