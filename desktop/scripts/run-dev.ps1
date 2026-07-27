param(
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

if (-not $PythonPath) {
    $PythonPath = Join-Path $RepositoryRoot ".venv-desktop\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Desktop Python environment not found. Run .\desktop\scripts\setup-dev.ps1 first, or pass -PythonPath."
}

& $PythonPath (Join-Path $PSScriptRoot "dev_runner.py")
exit $LASTEXITCODE
