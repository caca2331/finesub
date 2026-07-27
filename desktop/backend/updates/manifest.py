from __future__ import annotations

import base64
import binascii
import json
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field

from desktop.backend.common.models import DownloadAsset


class InvalidManifestSignature(ValueError):
    pass


class UpdateNotApplicable(ValueError):
    pass


class UpdateAsset(DownloadAsset):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    supported_from: list[str] = Field(
        default_factory=list,
        alias="supportedFrom",
    )


class UpdateAssets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: UpdateAsset
    full: UpdateAsset


class UpdateManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[1] = Field(alias="schemaVersion")
    key_id: str = Field(alias="keyId")
    version: str
    channel: Literal["stable", "beta"]
    platform: Literal["windows-x64"]
    draft: bool = False
    prerelease: bool = False
    minimum_launcher_version: str = Field(alias="minimumLauncherVersion")
    minimum_supported_version: str = Field(alias="minimumSupportedVersion")
    release_notes: str = Field(default="", alias="releaseNotes")
    mandatory: bool = False
    assets: UpdateAssets


class LocalUpdateState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    launcher_version: str
    channel: Literal["stable", "beta"]
    platform: Literal["windows-x64"]


def _decode_public_key(encoded: str) -> bytes:
    try:
        key = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise InvalidManifestSignature("Trusted update key is not valid base64") from error
    if len(key) != 32:
        raise InvalidManifestSignature("Trusted Ed25519 public key must be 32 bytes")
    return key


def _decode_signature(signature: bytes) -> bytes:
    if len(signature) == 64:
        return signature
    stripped = signature.strip()
    try:
        decoded = base64.b64decode(stripped, validate=True)
    except (ValueError, binascii.Error) as error:
        raise InvalidManifestSignature("Update signature is malformed") from error
    if len(decoded) != 64:
        raise InvalidManifestSignature("Ed25519 signature must be 64 bytes")
    return decoded


def verify_manifest(
    manifest_bytes: bytes,
    signature_bytes: bytes,
    trusted_keys: dict[str, str],
    *,
    expected_channel: Literal["stable", "beta"] = "stable",
    expected_platform: Literal["windows-x64"] = "windows-x64",
) -> UpdateManifest:
    try:
        untrusted = json.loads(manifest_bytes)
        key_id = untrusted["keyId"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise InvalidManifestSignature("Update manifest is malformed") from error
    if not isinstance(key_id, str) or key_id not in trusted_keys:
        raise InvalidManifestSignature("Update manifest uses an unknown signing key")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            _decode_public_key(trusted_keys[key_id])
        )
        public_key.verify(_decode_signature(signature_bytes), manifest_bytes)
    except (InvalidSignature, ValueError) as error:
        raise InvalidManifestSignature("Update manifest signature is invalid") from error

    manifest = UpdateManifest.model_validate(untrusted)
    if manifest.draft:
        raise UpdateNotApplicable("Draft releases cannot be installed")
    if manifest.channel != expected_channel:
        raise UpdateNotApplicable(
            f"Manifest channel {manifest.channel!r} does not match {expected_channel!r}"
        )
    if manifest.platform != expected_platform:
        raise UpdateNotApplicable(
            f"Manifest platform {manifest.platform!r} is not supported"
        )
    if expected_channel == "stable" and manifest.prerelease:
        raise UpdateNotApplicable("Prereleases are not accepted on the stable channel")
    _version(manifest.version)
    _version(manifest.minimum_launcher_version)
    _version(manifest.minimum_supported_version)
    for supported in manifest.assets.app.supported_from:
        _version(supported)
    return manifest


def select_asset(
    manifest: UpdateManifest,
    local: LocalUpdateState,
) -> Literal["app", "full"]:
    if manifest.channel != local.channel or manifest.platform != local.platform:
        raise UpdateNotApplicable("Update channel or platform does not match")
    if _version(manifest.version) <= _version(local.version):
        raise UpdateNotApplicable("Update version must be newer than the installed version")
    if _version(local.launcher_version) < _version(
        manifest.minimum_launcher_version
    ):
        return "full"
    if _version(local.version) < _version(manifest.minimum_supported_version):
        return "full"
    if local.version not in manifest.assets.app.supported_from:
        return "full"
    return "app"


def _version(value: str) -> Version:
    try:
        return Version(value)
    except InvalidVersion as error:
        raise UpdateNotApplicable(f"Invalid update version: {value!r}") from error
