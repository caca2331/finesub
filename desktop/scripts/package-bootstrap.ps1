[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [Parameter(Mandatory = $true)]
    [string]$LauncherConfigPath,
    [Parameter(Mandatory = $true)]
    [string]$TrustedKeysPath
)

$ErrorActionPreference = "Stop"
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)

# What ships is decided by git, not by whatever happens to be on disk.
#
# This used to walk the source tree with -Force and copy everything whose
# extension was not .pyc/.pyo, skipping only three directory names. It never
# consulted .gitignore, so anything a maintainer left under src/ -- a *.log of
# real API responses, a .bak of a prompt, a scratch module -- went into a
# signed, publicly downloadable zip, invisible to `git status`, to the CI gate
# and to review. The local build already carried `finesub.egg-info/` and a
# gitignored `utils/` for exactly this reason.
function Get-TrackedRelativePaths {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$RelativeRoot
    )

    Push-Location -LiteralPath $RepoRoot
    try {
        $Tracked = & git ls-files --cached --full-name -- $RelativeRoot
        if ($LASTEXITCODE -ne 0) {
            throw "git ls-files failed for $RelativeRoot; refusing to guess what to ship"
        }
    }
    finally {
        Pop-Location
    }
    return @($Tracked | Where-Object { $_ })
}

function Copy-TrackedTree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$RelativeRoot,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $Prefix = $RelativeRoot.Replace('\', '/').TrimEnd('/') + '/'
    foreach ($Relative in Get-TrackedRelativePaths -RepoRoot $RepoRoot -RelativeRoot $RelativeRoot) {
        $Normalized = $Relative.Replace('\', '/')
        if (-not $Normalized.StartsWith($Prefix)) { continue }
        $Tail = $Normalized.Substring($Prefix.Length)
        # Tests are the one tracked thing an end user has no use for.
        if ($Tail -match '(^|/)tests(/|$)') { continue }
        # Belt and braces: byte-code should already be gitignored, but a
        # force-added one must still never reach a release.
        if ($Tail -match '(^|/)__pycache__(/|$)') { continue }
        if ($Tail -match '[.](pyc|pyo)$') { continue }
        $Target = Join-Path $Destination ($Tail.Replace('/', '\'))
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Target) | Out-Null
        Copy-Item -LiteralPath (Join-Path $RepoRoot ($Normalized.Replace('/', '\'))) `
            -Destination $Target -Force
    }
}

# Build outputs and secrets that are deliberately untracked yet must ship. Each
# is asserted to exist: a naive `git ls-files` whitelist would silently drop
# them and produce a package with no interface (frontend/out) or no update
# trust anchor (trusted-update-keys.json).
function Assert-RequiredUntracked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Paths
    )

    foreach ($Required in $Paths) {
        if (-not (Test-Path -LiteralPath $Required)) {
            throw "Required (untracked) build input is missing: $Required"
        }
    }
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    [System.IO.File]::WriteAllText(
        $Path,
        $Content + [System.Environment]::NewLine,
        (New-Object System.Text.UTF8Encoding($false))
    )
}

$LauncherDist = Join-Path $OutputDirectory "FineSub Desktop.dist"
if (-not (Test-Path -LiteralPath (Join-Path $LauncherDist "FineSub Desktop.exe") -PathType Leaf)) {
    throw "FineSub Desktop.exe was not generated."
}

$VersionRoot = Join-Path $LauncherDist "app\versions\$Version"
if (Test-Path -LiteralPath $VersionRoot) {
    Remove-Item -LiteralPath $VersionRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $VersionRoot | Out-Null
# Build inputs that are deliberately untracked. Named here so a missing one
# fails the build instead of shipping a package with no interface, or with
# no trust anchor for verifying its own updates.
$FrontendOut = Join-Path $RepoRoot "desktop\frontend\out"
$TrustedKeys = Join-Path $RepoRoot "desktop\resources\trusted-update-keys.json"
Assert-RequiredUntracked -Paths @($FrontendOut)

Copy-TrackedTree -RepoRoot $RepoRoot -RelativeRoot "src" `
    -Destination (Join-Path $VersionRoot "src")

# Pre-0.4.0 launchers (frozen 0.3.x/0.2.x exes in the field) hard-code
# src/asr_playground/pipeline.py in REQUIRED_APP_FILES and in
# resolve_application_source, and they are the code that validates and boots
# THIS payload after an app-incremental update. Without the stub they reject
# the payload outright, and a hybrid install (old exe + new app dir) cannot
# start. Existence is all they check; nothing imports it. Keep shipping it
# until in-app updates from pre-rename installs are explicitly dropped.
$LegacyPackageDir = Join-Path $VersionRoot "src\asr_playground"
New-Item -ItemType Directory -Force -Path $LegacyPackageDir | Out-Null
Write-Utf8NoBom -Path (Join-Path $LegacyPackageDir "pipeline.py") -Content @'
"""Compatibility placeholder for pre-0.4.0 launchers.

The package was renamed to ``finesub`` in 0.4.0. Launchers frozen before the
rename validate update payloads and locate the active application source by
checking that this file exists; they never import it. See
desktop/scripts/package-bootstrap.ps1 for why it is generated here.
"""

raise ImportError(
    "asr_playground was renamed to finesub in 0.4.0; "
    "this stub only satisfies pre-0.4.0 launchers' payload checks"
)
'@

$VersionDesktop = Join-Path $VersionRoot "desktop"
New-Item -ItemType Directory -Force -Path $VersionDesktop | Out-Null
Copy-Item -LiteralPath (Join-Path $RepoRoot "desktop\__init__.py") -Destination $VersionDesktop -Force
Copy-TrackedTree -RepoRoot $RepoRoot -RelativeRoot "desktop/backend" `
    -Destination (Join-Path $VersionDesktop "backend")
Copy-TrackedTree -RepoRoot $RepoRoot -RelativeRoot "desktop/resources" `
    -Destination (Join-Path $VersionDesktop "resources")
Copy-TrackedTree -RepoRoot $RepoRoot -RelativeRoot "desktop/runtime" `
    -Destination (Join-Path $VersionDesktop "runtime")
# Untracked on purpose -- the signing trust anchor is not in the repo, but a
# build without it can never verify an update.
if (Test-Path -LiteralPath $TrustedKeys) {
    Copy-Item -LiteralPath $TrustedKeys `
        -Destination (Join-Path $VersionDesktop "resources") -Force
}
# The built frontend: untracked by definition, and the whole interface.
Copy-Item -LiteralPath $FrontendOut `
    -Destination (Join-Path $VersionDesktop "frontend\out") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "pyproject.toml") -Destination $VersionRoot -Force

$AppManifest = @{
    version = $Version
    platform = "windows-x64"
} | ConvertTo-Json -Compress
Write-Utf8NoBom `
    -Path (Join-Path $VersionRoot "app-manifest.json") `
    -Content $AppManifest

$Pointer = @{
    current = $Version
    previous = $null
    pendingHealth = $false
} | ConvertTo-Json -Compress
$AppRoot = Join-Path $LauncherDist "app"
New-Item -ItemType Directory -Force -Path $AppRoot | Out-Null
Write-Utf8NoBom -Path (Join-Path $AppRoot "current.json") -Content $Pointer

# The command line for this installation. Both files sit at the package root:
# finesub.cmd finds the managed interpreter beside itself, finesub.py finds the
# active app sources under it.
foreach ($PackageCli in @("finesub.cmd", "finesub.py")) {
    Copy-Item `
        -LiteralPath (Join-Path $RepoRoot "desktop\assets\package-cli\$PackageCli") `
        -Destination $LauncherDist `
        -Force
}

$LauncherConfig = Get-Content -LiteralPath $LauncherConfigPath -Raw | ConvertFrom-Json
$LauncherConfig.appVersion = $Version
$LauncherConfig.launcherVersion = $Version
$LauncherConfigJson = $LauncherConfig | ConvertTo-Json -Depth 8 -Compress
Write-Utf8NoBom `
    -Path (Join-Path $LauncherDist "launcher.json") `
    -Content $LauncherConfigJson
Copy-Item -LiteralPath $TrustedKeysPath -Destination (Join-Path $LauncherDist "trusted-update-keys.json") -Force

Write-Host "FineSub onedir package: $LauncherDist"
