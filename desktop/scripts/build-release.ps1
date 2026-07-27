[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [ValidateSet("stable", "beta")][string]$Channel = "stable",
    [Parameter(Mandatory = $true)][string]$KeyId,
    [Parameter(Mandatory = $true)][string]$PrivateKeyPath,
    [string]$VenvPath = "",
    [string]$BootstrapDirectory = "",
    [string]$MinimumLauncherVersion = "0.2.3",
    [string]$MinimumSupportedVersion = "0.1.0",
    [string[]]$SupportedFrom = @("0.2.3"),
    [string]$ReleaseNotes = "",
    [string]$Repository = "caca2331/finesub",
    [switch]$SkipBootstrap
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $VenvPath) {
    if ($env:FINESUB_DESKTOP_VENV) {
        $VenvPath = $env:FINESUB_DESKTOP_VENV
    }
    else {
        $VenvPath = Join-Path $RepoRoot ".venv-desktop"
    }
}
$Python = Join-Path ([System.IO.Path]::GetFullPath($VenvPath)) "Scripts\python.exe"
if (-not $BootstrapDirectory) {
    $BootstrapDirectory = Join-Path $RepoRoot "dist\bootstrap"
    if ($BootstrapDirectory -match "[^\u0000-\u007F]") {
        $BootstrapDirectory = Join-Path `
            ([System.IO.Path]::GetTempPath()) `
            "finesub-build\$Version"
    }
}
$BootstrapDirectory = [System.IO.Path]::GetFullPath($BootstrapDirectory)
$Bootstrap = Join-Path $BootstrapDirectory "FineSub Desktop.dist"
if (-not $SkipBootstrap) {
    & (Join-Path $PSScriptRoot "build-bootstrap.ps1") `
        -VenvPath $VenvPath `
        -OutputDirectory $BootstrapDirectory `
        -Version $Version
}
$AppSource = Join-Path $Bootstrap "app\versions\$Version"
if (-not (Test-Path -LiteralPath $AppSource -PathType Container)) {
    throw "App source does not exist: $AppSource"
}
$Arguments = @(
    "-m", "desktop.scripts.build_release",
    "--version", $Version,
    "--channel", $Channel,
    "--key-id", $KeyId,
    "--private-key", ([System.IO.Path]::GetFullPath($PrivateKeyPath)),
    "--app-source", $AppSource,
    "--full-source", $Bootstrap,
    "--output-dir", (Join-Path $RepoRoot "dist\release"),
    "--minimum-launcher", $MinimumLauncherVersion,
    "--minimum-supported", $MinimumSupportedVersion,
    "--release-notes", $ReleaseNotes,
    "--repository", $Repository
)
foreach ($VersionValue in $SupportedFrom) {
    $Arguments += @("--supported-from", $VersionValue)
}
Push-Location $RepoRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Release asset generation failed."
    }
}
finally {
    Pop-Location
}
