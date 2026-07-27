param(
    [string]$VenvPath = "",
    [switch]$IncludePipeline
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$FrontendRoot = Join-Path $RepositoryRoot "desktop\frontend"

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
& $PythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed with exit code $LASTEXITCODE"
}

$Extras = if ($IncludePipeline) { ".[desktop,dev,asr,harness]" } else { ".[desktop,dev]" }
Push-Location $RepositoryRoot
try {
    & $PythonPath -m pip install -e $Extras
    if ($LASTEXITCODE -ne 0) {
        throw "Python dependency installation failed with exit code $LASTEXITCODE"
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
Write-Host "Run: .\desktop\scripts\run-dev.ps1 -PythonPath `"$PythonPath`""
