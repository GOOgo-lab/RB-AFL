from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("cryptography"), "cryptography is an optional full dependency")
class SigningTests(unittest.TestCase):
    def test_tampering_is_detected(self) -> None:
        from cryptography.exceptions import InvalidSignature

        from rbafl.signing import generate_keypair, sign_record, verify_record_signature

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private_key = root / "private.key"
            public_key = root / "public.pem"
            generate_keypair(private_key, public_key)
            signed = sign_record(
                {"record_id": "r1", "identity": "example", "zero_watermark_bits_b64": "AA=="},
                private_key,
                signer_id="author-1",
            )
            self.assertTrue(verify_record_signature(signed, public_key))
            signed["identity"] = "tampered"
            with self.assertRaises(InvalidSignature):
                verify_record_signature(signed, public_key)


if __name__ == "__main__":
    unittest.main()

