from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from desktop.backend.updates.manifest import (
    InvalidManifestSignature,
    LocalUpdateState,
    UpdateNotApplicable,
    select_asset,
    verify_manifest,
)


def _manifest_bytes(
    *,
    version: str = "1.2.0",
    minimum_launcher: str = "1.0.0",
    supported_from: list[str] | None = None,
) -> bytes:
    body = {
        "schemaVersion": 1,
        "keyId": "release-key",
        "version": version,
        "channel": "stable",
        "platform": "windows-x64",
        "draft": False,
        "prerelease": False,
        "minimumLauncherVersion": minimum_launcher,
        "minimumSupportedVersion": "1.0.0",
        "releaseNotes": "测试更新",
        "assets": {
            "app": {
                "url": "https://example.com/finesub-app.zip",
                "size": 10,
                "sha256": "1" * 64,
                "supportedFrom": supported_from or ["1.0.0"],
            },
            "full": {
                "url": "https://example.com/finesub-full.zip",
                "size": 20,
                "sha256": "2" * 64,
            },
        },
    }
    return json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


@pytest.fixture
def signing_material():
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trusted = {"release-key": base64.b64encode(public).decode("ascii")}
    return private, trusted


def test_manifest_accepts_signature_from_trusted_key(signing_material) -> None:
    signing_key, trusted_keys = signing_material
    body = _manifest_bytes()

    parsed = verify_manifest(body, signing_key.sign(body), trusted_keys)

    assert parsed.version == "1.2.0"


def test_manifest_rejects_modified_body(signing_material) -> None:
    signing_key, trusted_keys = signing_material
    body = _manifest_bytes()
    signature = signing_key.sign(body)

    with pytest.raises(InvalidManifestSignature):
        verify_manifest(
            body.replace(b"1.2.0", b"1.2.1"),
            signature,
            trusted_keys,
        )


def test_manifest_rejects_unknown_signing_key(signing_material) -> None:
    signing_key, _ = signing_material
    body = _manifest_bytes()

    with pytest.raises(InvalidManifestSignature):
        verify_manifest(body, signing_key.sign(body), {})


def test_launcher_version_forces_full_update(signing_material) -> None:
    signing_key, trusted_keys = signing_material
    body = _manifest_bytes(minimum_launcher="2.0.0")
    manifest = verify_manifest(body, signing_key.sign(body), trusted_keys)

    selected = select_asset(
        manifest,
        LocalUpdateState(
            version="1.0.0",
            launcher_version="1.5.0",
            channel="stable",
            platform="windows-x64",
        ),
    )

    assert selected == "full"


def test_compatible_version_uses_small_app_update(signing_material) -> None:
    signing_key, trusted_keys = signing_material
    body = _manifest_bytes(supported_from=["1.0.0"])
    manifest = verify_manifest(body, signing_key.sign(body), trusted_keys)

    selected = select_asset(
        manifest,
        LocalUpdateState(
            version="1.0.0",
            launcher_version="1.0.0",
            channel="stable",
            platform="windows-x64",
        ),
    )

    assert selected == "app"


def test_downgrade_is_rejected(signing_material) -> None:
    signing_key, trusted_keys = signing_material
    body = _manifest_bytes(version="1.0.0")
    manifest = verify_manifest(body, signing_key.sign(body), trusted_keys)

    with pytest.raises(UpdateNotApplicable):
        select_asset(
            manifest,
            LocalUpdateState(
                version="1.1.0",
                launcher_version="1.0.0",
                channel="stable",
                platform="windows-x64",
            ),
        )
