from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)


ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class ReleaseBuildConfig:
    version: str
    channel: Literal["stable", "beta"]
    key_id: str
    private_key_path: Path
    app_source: Path
    full_source: Path
    output_dir: Path
    minimum_launcher_version: str
    minimum_supported_version: str
    app_supported_from: list[str]
    release_notes: str
    repository: str = "caca2331/finesub"
    platform: Literal["windows-x64"] = "windows-x64"


@dataclass(frozen=True, slots=True)
class ReleaseArtifacts:
    app_zip: Path
    full_zip: Path
    manifest: Path
    manifest_sig: Path


def _archive_tree(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Release source does not exist: {source}")
    files = sorted(
        (path for path in source.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(
        destination,
        "w",
        compression=ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in files:
            if path.is_symlink():
                raise ValueError(f"Release archives cannot contain symlinks: {path}")
            relative = path.relative_to(source).as_posix()
            info = ZipInfo(relative, date_time=ZIP_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(
        path.expanduser().resolve().read_bytes(),
        password=None,
    )
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("Release key must be an Ed25519 private key")
    return key


def build_release(config: ReleaseBuildConfig) -> ReleaseArtifacts:
    output = config.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    asset_platform = "win-x64" if config.platform == "windows-x64" else config.platform
    app_name = f"finesub-app-{config.version}-{asset_platform}.zip"
    full_name = f"finesub-full-{config.version}-{asset_platform}.zip"
    app_zip = output / app_name
    full_zip = output / full_name
    _archive_tree(config.app_source, app_zip)
    _archive_tree(config.full_source, full_zip)

    tag = f"v{config.version}"
    release_root = (
        f"https://github.com/{config.repository}/releases/download/{tag}"
    )
    manifest_body = {
        "assets": {
            "app": {
                "sha256": _sha256(app_zip),
                "size": app_zip.stat().st_size,
                "supportedFrom": config.app_supported_from,
                "url": f"{release_root}/{app_name}",
            },
            "full": {
                "sha256": _sha256(full_zip),
                "size": full_zip.stat().st_size,
                "url": f"{release_root}/{full_name}",
            },
        },
        "channel": config.channel,
        "draft": False,
        "keyId": config.key_id,
        "minimumLauncherVersion": config.minimum_launcher_version,
        "minimumSupportedVersion": config.minimum_supported_version,
        "platform": config.platform,
        "prerelease": config.channel == "beta",
        "releaseNotes": config.release_notes,
        "schemaVersion": 1,
        "version": config.version,
    }
    manifest_bytes = json.dumps(
        manifest_body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest = output / "update-manifest.json"
    manifest.write_bytes(manifest_bytes)
    signature = _load_private_key(config.private_key_path).sign(manifest_bytes)
    manifest_sig = output / "update-manifest.sig"
    manifest_sig.write_bytes(base64.b64encode(signature) + b"\n")
    (output / f"{app_name}.sha256").write_text(
        f"{_sha256(app_zip)}  {app_name}\n",
        encoding="ascii",
        newline="\n",
    )
    (output / f"{full_name}.sha256").write_text(
        f"{_sha256(full_zip)}  {full_name}\n",
        encoding="ascii",
        newline="\n",
    )
    return ReleaseArtifacts(
        app_zip=app_zip,
        full_zip=full_zip,
        manifest=manifest,
        manifest_sig=manifest_sig,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build signed FineSub releases")
    parser.add_argument("--version", required=True)
    parser.add_argument("--channel", choices=("stable", "beta"), default="stable")
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--app-source", type=Path, required=True)
    parser.add_argument("--full-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dist/release"))
    parser.add_argument("--minimum-launcher", required=True)
    parser.add_argument("--minimum-supported", required=True)
    parser.add_argument("--supported-from", action="append", default=[])
    parser.add_argument("--release-notes", default="")
    parser.add_argument("--repository", default="caca2331/finesub")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifacts = build_release(
        ReleaseBuildConfig(
            version=args.version,
            channel=args.channel,
            key_id=args.key_id,
            private_key_path=args.private_key,
            app_source=args.app_source,
            full_source=args.full_source,
            output_dir=args.output_dir,
            minimum_launcher_version=args.minimum_launcher,
            minimum_supported_version=args.minimum_supported,
            app_supported_from=args.supported_from,
            release_notes=args.release_notes,
            repository=args.repository,
        )
    )
    print(artifacts.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
