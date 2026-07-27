[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Frontend = Join-Path $RepoRoot "desktop\frontend"

Push-Location $Frontend
try {
    if (-not $SkipInstall) {
        & npm ci
        if ($LASTEXITCODE -ne 0) {
            throw "npm ci failed with exit code $LASTEXITCODE"
        }
    }
    $env:NEXT_TELEMETRY_DISABLED = "1"
    & npm run test
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend tests failed with exit code $LASTEXITCODE"
    }
    & npm run typecheck
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend typecheck failed with exit code $LASTEXITCODE"
    }
    & npm run build
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend build failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$Index = Join-Path $Frontend "out\index.html"
if (-not (Test-Path -LiteralPath $Index -PathType Leaf)) {
    throw "Static frontend was not generated: $Index"
}
if (-not $PythonPath) {
    $RepoPython = Join-Path $RepoRoot ".venv-desktop\Scripts\python.exe"
    if (Test-Path -LiteralPath $RepoPython -PathType Leaf) {
        $PythonPath = $RepoPython
    }
    else {
        $PythonPath = (Get-Command python.exe -ErrorAction Stop).Source
    }
}
& $PythonPath (Join-Path $PSScriptRoot "verify_static_export.py") $Index
if ($LASTEXITCODE -ne 0) {
    throw "Static WebView export validation failed."
}
Write-Host "FineSub frontend ready: $Index"
