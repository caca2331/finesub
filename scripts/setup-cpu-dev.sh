#!/bin/sh
# CPU-only development environment for a Linux machine with no GPU, such as a
# cloud agent session: the packages ci.yml's ubuntu job installs (keep the two
# in step). Enough for `python -m pytest -q`; the GPU / ASR stack is not
# installed and its tests skip themselves. CPU torch is required, not optional:
# the suite's conftest imports the device module, which imports torch, so
# without it every test errors at collection. It comes from
# download.pytorch.org (PyPI's Linux torch drags in several GB of CUDA), so a
# sandbox has to allow that host.
#
# Meant to run at the start of every session (a Claude Code SessionStart hook,
# or a cloud environment's setup script), so it is idempotent: once the
# install matches pyproject.toml and this script it only re-exports PATH.
#
# Installs into the system Python, or into .venv/ when that Python is
# externally managed (PEP 668). The .venv/ sees the system site-packages, so
# whatever a cloud environment's cached setup step preinstalled there (CPU
# torch, above all) counts as already satisfied instead of downloading again. With .venv/, later commands need it on PATH:
# under Claude Code this script writes that to $CLAUDE_ENV_FILE; elsewhere it
# prints the line to add.
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

py=
for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
        py=$candidate
        break
    fi
done
if [ -z "$py" ]; then
    echo "setup-cpu-dev: no Python >= 3.12 on PATH (pyproject.toml requires it)" >&2
    exit 1
fi

managed=$("$py" -c 'import os, sysconfig; print(int(os.path.exists(os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED"))))')
if [ "$managed" = 1 ]; then
    # Checked by running pip, not by the file: a venv whose ensurepip was
    # missing (Debian without python3.x-venv) leaves bin/python behind.
    if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
        rm -rf .venv
        "$py" -m venv --system-site-packages .venv
    fi
    py=$root/.venv/bin/python
    if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
        echo "export PATH=\"$root/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
    else
        echo "setup-cpu-dev: installed into .venv/; put it on PATH: export PATH=\"$root/.venv/bin:\$PATH\"" >&2
    fi
fi

# Hooks are not cloned, so a fresh clone has no guard against pushing dev's
# history to the public remote. Installed every run (cheap), before the fast
# path; a different hook already there is the owner's and is left alone.
hooks=$(git rev-parse --git-path hooks 2>/dev/null) || hooks=
if [ -n "$hooks" ] && [ -z "$(git config --get core.hooksPath)" ]; then
    if [ ! -e "$hooks/pre-push" ]; then
        mkdir -p "$hooks"
        cp scripts/pre-push "$hooks/pre-push"
        chmod +x "$hooks/pre-push"
    elif ! cmp -s scripts/pre-push "$hooks/pre-push"; then
        echo "setup-cpu-dev: $hooks/pre-push differs from scripts/pre-push; left as is" >&2
    fi
fi

stamp_dir=${XDG_CACHE_HOME:-$HOME/.cache}
stamp=$stamp_dir/finesub-cpu-dev.stamp
want=$(cat pyproject.toml scripts/setup-cpu-dev.sh | "$py" -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())')
want="$want $py"
# The import guards against a stamp that outlived its environment (a recreated .venv/).
if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ] && "$py" -c 'import finesub' 2>/dev/null; then
    exit 0
fi

"$py" -m pip install -q -U pip setuptools
"$py" -m pip install -q -e ".[harness,dev]"
# numpy / soundfile belong to [asr], which is not installed here.
"$py" -m pip install -q numpy soundfile
if ! "$py" -m pip install -q "torch==2.11.0" "torchaudio==2.11.0" --index-url https://download.pytorch.org/whl/cpu; then
    # No stamp, so the next session tries again.
    echo "setup-cpu-dev: CPU torch did not install. Without it every test errors at collection." >&2
    echo "setup-cpu-dev: allow download.pytorch.org in the environment's network settings, then rerun this script." >&2
    exit 1
fi

mkdir -p "$stamp_dir"
echo "$want" > "$stamp"
