[CmdletBinding()]
param(
    [string]$OutputDirectory = "",
    [string]$Version = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Version) {
    $Version = (
        Get-Content -LiteralPath (Join-Path $RepoRoot "desktop\VERSION") -Raw
    ).Trim()
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $RepoRoot "dist\cli"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$Stage = Join-Path $OutputDirectory ".wheel-stage"

if (Test-Path -LiteralPath $Stage) {
    Remove-Item -LiteralPath $Stage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

function Copy-PythonTree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    foreach ($Item in Get-ChildItem -LiteralPath $Source) {
        if ($Item.PSIsContainer) {
            if (
                $Item.Name -eq "__pycache__" -or
                $Item.Name -eq "tests" -or
                $Item.Name.StartsWith(".")
            ) {
                continue
            }
            Copy-PythonTree `
                -Source $Item.FullName `
                -Destination (Join-Path $Destination $Item.Name)
        }
        elseif ($Item.Extension -ne ".pyc") {
            Copy-Item `
                -LiteralPath $Item.FullName `
                -Destination $Destination `
                -Force
        }
    }
}

# The shell package itself.
foreach ($ProjectFile in @("pyproject.toml", "MANIFEST.in", "README.md")) {
    Copy-Item `
        -LiteralPath (Join-Path $RepoRoot "cli\$ProjectFile") `
        -Destination $Stage `
        -Force
}
# License comes from the repository root so the wheel carries it (PEP 639).
Copy-Item `
    -LiteralPath (Join-Path $RepoRoot "LICENSE") `
    -Destination $Stage `
    -Force
# WriteAllText writes UTF-8 without BOM under both Windows PowerShell 5.1
# (the CI default, whose Set-Content lacks utf8NoBOM) and pwsh 7.
[System.IO.File]::WriteAllText((Join-Path $Stage "VERSION"), $Version)
Copy-PythonTree `
    -Source (Join-Path $RepoRoot "cli\src\finesub_cli") `
    -Destination (Join-Path $Stage "src\finesub_cli")

# Vendored pipeline sources: what PYTHONPATH points at inside the managed
# runtime. Kept under _vendor (not site-packages importable from the shell's
# venv) so the shell's own third-party packages can never shadow the
# lock-pinned versions in the runtime.
$Vendor = Join-Path $Stage "src\finesub_cli\_vendor"
foreach ($Package in @("asr_playground", "llm", "finesub_bootstrap")) {
    Copy-PythonTree `
        -Source (Join-Path $RepoRoot "src\$Package") `
        -Destination (Join-Path $Vendor "src\$Package")
}
Copy-Item `
    -LiteralPath (Join-Path $RepoRoot "desktop\runtime\pylock.win-py312.toml") `
    -Destination (Join-Path $Vendor "pylock.win-py312.toml") `
    -Force
Copy-Item `
    -LiteralPath (Join-Path $RepoRoot "desktop\resources\runtime-manifest.json") `
    -Destination (Join-Path $Vendor "runtime-manifest.json") `
    -Force

foreach ($RequiredFile in @(
    "src\finesub_cli\main.py",
    "src\finesub_cli\_vendor\pylock.win-py312.toml",
    "src\finesub_cli\_vendor\runtime-manifest.json",
    "src\finesub_cli\_vendor\src\asr_playground\pipeline.py",
    "src\finesub_cli\_vendor\src\llm\correction_translation.py",
    "src\finesub_cli\_vendor\src\llm\model_catalog.psv",
    "src\finesub_cli\_vendor\src\finesub_bootstrap\environment.py"
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $Stage $RequiredFile))) {
        throw "Wheel staging is incomplete; missing: $RequiredFile"
    }
}

& $Python -m build --wheel --outdir $OutputDirectory $Stage
if ($LASTEXITCODE -ne 0) {
    throw "Wheel build failed with exit code $LASTEXITCODE."
}

$Wheel = Get-ChildItem -LiteralPath $OutputDirectory -Filter "finesub-$Version-*.whl"
if (-not $Wheel) {
    throw "Expected wheel finesub-$Version-*.whl was not produced."
}
& $Python -c @"
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as wheel:
    names = wheel.namelist()
for required in (
    'finesub_cli/main.py',
    'finesub_cli/_vendor/pylock.win-py312.toml',
    'finesub_cli/_vendor/runtime-manifest.json',
    'finesub_cli/_vendor/src/asr_playground/pipeline.py',
    'finesub_cli/_vendor/src/llm/prompt_templates/',
):
    if not any(name.startswith(required) for name in names):
        raise SystemExit(f'wheel is missing {required}')
print('wheel contents verified')
"@ $Wheel[0].FullName
if ($LASTEXITCODE -ne 0) {
    throw "Wheel content verification failed."
}

Write-Host "FineSub CLI wheel ready:"
Write-Host $Wheel[0].FullName
