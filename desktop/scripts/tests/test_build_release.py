from __future__ import annotations

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from desktop.backend.updates.manifest import verify_manifest
from desktop.scripts.build_release import ReleaseBuildConfig, build_release


def _write_app_source(root: Path, version: str) -> None:
    files = {
        "src/asr_playground/pipeline.py": "pipeline",
        "desktop/backend/worker/main.py": "worker",
        "desktop/frontend/out/index.html": "<html></html>",
        "pyproject.toml": "[project]\nname='finesub'\nversion='0.2.0'\n",
        "app-manifest.json": json.dumps(
            {"version": version, "platform": "windows-x64"},
            separators=(",", ":"),
        ),
    }
    for relative, body in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


def test_release_builder_emits_signed_assets(tmp_path: Path) -> None:
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "release-key.pem"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    app_source = tmp_path / "app-source"
    full_source = tmp_path / "full-source"
    _write_app_source(app_source, "0.2.0")
    full_source.mkdir()
    (full_source / "FineSub Desktop.exe").write_bytes(b"launcher")
    (full_source / "FineSub Desktop Updater.exe").write_bytes(b"updater")

    result = build_release(
        ReleaseBuildConfig(
            version="0.2.0",
            channel="stable",
            key_id="test-release-key",
            private_key_path=private_path,
            app_source=app_source,
            full_source=full_source,
            output_dir=tmp_path / "release",
            minimum_launcher_version="0.1.0",
            minimum_supported_version="0.1.0",
            app_supported_from=["0.1.0"],
            release_notes="FineSub Desktop 0.2.0",
        )
    )

    assert result.app_zip.name == "finesub-app-0.2.0-win-x64.zip"
    assert result.full_zip.name == "finesub-full-0.2.0-win-x64.zip"
    assert result.manifest_sig.is_file()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    parsed = verify_manifest(
        result.manifest.read_bytes(),
        result.manifest_sig.read_bytes(),
        {"test-release-key": base64.b64encode(public).decode("ascii")},
    )
    assert parsed.version == "0.2.0"
    assert parsed.assets.app.size == result.app_zip.stat().st_size


def test_release_zip_is_reproducible(tmp_path: Path) -> None:
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "release-key.pem"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    app_source = tmp_path / "app-source"
    full_source = tmp_path / "full-source"
    _write_app_source(app_source, "0.2.0")
    full_source.mkdir()
    (full_source / "FineSub Desktop.exe").write_bytes(b"launcher")
    common = dict(
        version="0.2.0",
        channel="stable",
        key_id="key",
        private_key_path=private_path,
        app_source=app_source,
        full_source=full_source,
        minimum_launcher_version="0.1.0",
        minimum_supported_version="0.1.0",
        app_supported_from=["0.1.0"],
        release_notes="same",
    )

    first = build_release(
        ReleaseBuildConfig(output_dir=tmp_path / "one", **common)
    )
    second = build_release(
        ReleaseBuildConfig(output_dir=tmp_path / "two", **common)
    )

    assert first.app_zip.read_bytes() == second.app_zip.read_bytes()
    assert first.full_zip.read_bytes() == second.full_zip.read_bytes()
