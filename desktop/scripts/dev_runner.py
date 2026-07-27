from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = REPOSITORY_ROOT / "desktop" / "frontend"
DEVELOPMENT_HOST = "127.0.0.1"
DEVELOPMENT_PORT = 3000


def _wait_for_frontend(process: subprocess.Popen[bytes], timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"Next.js exited before becoming ready (exit code {exit_code})."
            )
        try:
            with socket.create_connection(
                (DEVELOPMENT_HOST, DEVELOPMENT_PORT),
                timeout=0.5,
            ):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError("Next.js did not become ready within 60 seconds.")


def _stop_frontend(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None:
        raise RuntimeError("Node.js/npm was not found on PATH.")
    if not (FRONTEND_ROOT / "node_modules").is_dir():
        raise RuntimeError(
            "Frontend dependencies are missing. Run "
            ".\\desktop\\scripts\\setup-dev.ps1 first."
        )

    creation_flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )
    frontend = subprocess.Popen(
        [npm, "run", "dev", "--", "--port", str(DEVELOPMENT_PORT)],
        cwd=FRONTEND_ROOT,
        creationflags=creation_flags,
    )
    try:
        _wait_for_frontend(frontend)
        os.environ["FINESUB_APP_ROOT"] = str(REPOSITORY_ROOT)
        os.environ["FINESUB_DESKTOP_DEV_URL"] = (
            f"http://{DEVELOPMENT_HOST}:{DEVELOPMENT_PORT}"
        )
        if str(REPOSITORY_ROOT) not in sys.path:
            sys.path.insert(0, str(REPOSITORY_ROOT))
        from desktop.backend.launcher.main import main as launch_desktop

        return launch_desktop()
    finally:
        _stop_frontend(frontend)


if __name__ == "__main__":
    raise SystemExit(main())
