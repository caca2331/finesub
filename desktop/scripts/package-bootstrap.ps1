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
if (-not (Test-Path -LiteralPath (Join-Path $LauncherDist "FineSub Desktop.exe") -PathType Leaf)) {
    throw "FineSub Desktop.exe was not generated."
}

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
    -Source (Join-Path $RepoRoot "desktop\runtime") `
    -Destination (Join-Path $VersionDesktop "runtime")
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
