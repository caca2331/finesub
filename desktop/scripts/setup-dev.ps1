param(
    [string]$VenvPath = "",
    [switch]$DesktopOnly
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$FrontendRoot = Join-Path $RepositoryRoot "desktop\frontend"
$RuntimeLock = Join-Path $RepositoryRoot "desktop\runtime\pylock.win-py312.toml"

# The lock is what the shipped app installs too (RuntimeEnvironment.install), so
# installing from it here is what makes a dev box match a user's machine -- down
# to the patched CTranslate2, which the lock carries as a hashed direct URL.
$InstallPipeline = -not $DesktopOnly

if (-not $VenvPath) {
    $VenvPath = Join-Path $RepositoryRoot ".venv-desktop"
}
if (-not (Test-Path -LiteralPath $VenvPath -PathType Container)) {
    & py -3.12 -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to create the Python 3.12 environment."
    }
}

$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
& $PythonPath -m pip install --upgrade pip "uv==0.11.32"
if ($LASTEXITCODE -ne 0) {
    throw "pip/uv bootstrap failed with exit code $LASTEXITCODE"
}

Push-Location $RepositoryRoot
try {
    & $PythonPath -m pip install -e ".[desktop,dev]"
    if ($LASTEXITCODE -ne 0) {
        throw "Desktop dependency installation failed with exit code $LASTEXITCODE"
    }
    if ($InstallPipeline) {
        $UvPath = Join-Path $VenvPath "Scripts\uv.exe"
        & $UvPath pip install --python $PythonPath --requirement $RuntimeLock
        if ($LASTEXITCODE -ne 0) {
            throw "Pipeline dependency installation failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}

Push-Location $FrontendRoot
try {
    & npm ci
    if ($LASTEXITCODE -ne 0) {
        throw "npm ci failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "FineSub desktop development dependencies are ready."
if ($InstallPipeline) {
    Write-Host "The complete locked ASR and translation runtime is installed."
}
else {
    Write-Host "Desktop-only mode selected; subtitle processing dependencies were skipped."
}
Write-Host "Run: .\desktop\scripts\run-dev.ps1 -PythonPath `"$PythonPath`""
