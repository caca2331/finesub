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

function Copy-ReleaseTree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    foreach ($Item in Get-ChildItem -LiteralPath $Source -Force) {
        if ($Item.PSIsContainer) {
            if (
                $Item.Name -eq "tests" -or
                $Item.Name -eq "__pycache__" -or
                $Item.Name.StartsWith(".tmp", [System.StringComparison]::OrdinalIgnoreCase)
            ) {
                continue
            }
            Copy-ReleaseTree `
                -Source $Item.FullName `
                -Destination (Join-Path $Destination $Item.Name)
        }
        elseif ($Item.Extension -notin @(".pyc", ".pyo")) {
            Copy-Item -LiteralPath $Item.FullName -Destination $Destination -Force
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
$UpdaterDist = Join-Path $OutputDirectory "FineSub Desktop Updater.dist"
if (-not (Test-Path -LiteralPath (Join-Path $LauncherDist "FineSub Desktop.exe") -PathType Leaf)) {
    throw "FineSub Desktop.exe was not generated."
}
if (-not (Test-Path -LiteralPath (Join-Path $UpdaterDist "FineSub Desktop Updater.exe") -PathType Leaf)) {
    throw "FineSub Desktop Updater.exe was not generated."
}

$UpdaterTarget = Join-Path $LauncherDist "updater"
if (Test-Path -LiteralPath $UpdaterTarget) {
    Remove-Item -LiteralPath $UpdaterTarget -Recurse -Force
}
Copy-ReleaseTree -Source $UpdaterDist -Destination $UpdaterTarget

# 0.2.1 and earlier validate and relaunch the historical executable names.
# Keep byte-identical aliases for one migration cycle so those launchers can
# consume the 0.2.2 Full package and then transition to the new product names.
Copy-Item `
    -LiteralPath (Join-Path $LauncherDist "FineSub Desktop.exe") `
    -Destination (Join-Path $LauncherDist "FineSub.exe") `
    -Force
Copy-Item `
    -LiteralPath (Join-Path $UpdaterTarget "FineSub Desktop Updater.exe") `
    -Destination (Join-Path $UpdaterTarget "FineSubUpdater.exe") `
    -Force

$VersionRoot = Join-Path $LauncherDist "app\versions\$Version"
if (Test-Path -LiteralPath $VersionRoot) {
    Remove-Item -LiteralPath $VersionRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $VersionRoot | Out-Null
Copy-ReleaseTree -Source (Join-Path $RepoRoot "src") -Destination (Join-Path $VersionRoot "src")

$VersionDesktop = Join-Path $VersionRoot "desktop"
New-Item -ItemType Directory -Force -Path $VersionDesktop | Out-Null
Copy-Item -LiteralPath (Join-Path $RepoRoot "desktop\__init__.py") -Destination $VersionDesktop -Force
Copy-ReleaseTree `
    -Source (Join-Path $RepoRoot "desktop\backend") `
    -Destination (Join-Path $VersionDesktop "backend")
Copy-ReleaseTree `
    -Source (Join-Path $RepoRoot "desktop\resources") `
    -Destination (Join-Path $VersionDesktop "resources")
Copy-ReleaseTree `
    -Source (Join-Path $RepoRoot "desktop\frontend\out") `
    -Destination (Join-Path $VersionDesktop "frontend\out")
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

$LauncherConfig = Get-Content -LiteralPath $LauncherConfigPath -Raw | ConvertFrom-Json
$LauncherConfig.appVersion = $Version
$LauncherConfig.launcherVersion = $Version
$LauncherConfigJson = $LauncherConfig | ConvertTo-Json -Depth 8 -Compress
Write-Utf8NoBom `
    -Path (Join-Path $LauncherDist "launcher.json") `
    -Content $LauncherConfigJson
Copy-Item -LiteralPath $TrustedKeysPath -Destination (Join-Path $LauncherDist "trusted-update-keys.json") -Force

Write-Host "FineSub onedir package: $LauncherDist"
