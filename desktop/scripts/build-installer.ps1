[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ApplicationDirectory,
    [string]$OutputDirectory = "",
    [string]$Version = "0.2.7",
    [string]$InnoCompiler = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Definition = Join-Path $RepoRoot "desktop\installer\FineSubDesktop.iss"
$SetupIcon = Join-Path $RepoRoot "desktop\assets\finesub-desktop.ico"

function Resolve-InnoCompiler {
    param([string]$RequestedPath)

    if ($RequestedPath) {
        $Resolved = [System.IO.Path]::GetFullPath($RequestedPath)
        if (-not (Test-Path -LiteralPath $Resolved -PathType Leaf)) {
            throw "Inno Setup compiler not found: $Resolved"
        }
        return $Resolved
    }

    $Command = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }

    $Candidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
    )
    foreach ($Candidate in $Candidates) {
        if ($Candidate -and (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
            return $Candidate
        }
    }
    throw @"
Inno Setup 6 compiler (ISCC.exe) was not found.
Install Inno Setup 6 or pass -InnoCompiler with its full path.
"@
}

if (-not (Test-Path -LiteralPath $Definition -PathType Leaf)) {
    throw "Installer definition not found: $Definition"
}
if (-not (Test-Path -LiteralPath $SetupIcon -PathType Leaf)) {
    throw "FineSub Desktop setup icon not found: $SetupIcon"
}

$ApplicationDirectory = [System.IO.Path]::GetFullPath($ApplicationDirectory)
if (-not (Test-Path -LiteralPath $ApplicationDirectory -PathType Container)) {
    throw "Packaged application directory not found: $ApplicationDirectory"
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $RepoRoot "dist\installer"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)

$RequiredFiles = @(
    "FineSub Desktop.exe",
    "updater\FineSub Desktop Updater.exe",
    "app\current.json",
    "app\versions\$Version\desktop\resources\runtime-manifest.json",
    "app\versions\$Version\desktop\frontend\out\index.html",
    "app\versions\$Version\desktop\frontend\out\fonts\OFL.txt"
)
foreach ($RelativePath in $RequiredFiles) {
    $RequiredPath = Join-Path $ApplicationDirectory $RelativePath
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Installer input is incomplete; required file not found: $RequiredPath"
    }
}

$CurrentPointer = Get-Content `
    -LiteralPath (Join-Path $ApplicationDirectory "app\current.json") `
    -Raw `
    -Encoding UTF8 | ConvertFrom-Json
if ($CurrentPointer.current -ne $Version) {
    throw "Packaged app version '$($CurrentPointer.current)' does not match installer version '$Version'."
}

$Compiler = Resolve-InnoCompiler -RequestedPath $InnoCompiler
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$CompilerArguments = @(
    "/DAppSource=$ApplicationDirectory",
    "/DAppVersion=$Version",
    "/DOutputDir=$OutputDirectory",
    "/DSetupIcon=$SetupIcon"
)
$ChineseLanguageFile = Join-Path `
    (Split-Path -Parent $Compiler) `
    "Languages\ChineseSimplified.isl"
if (Test-Path -LiteralPath $ChineseLanguageFile -PathType Leaf) {
    $CompilerArguments += "/DIncludeChineseLanguage"
}
$CompilerArguments += $Definition
& $Compiler @CompilerArguments
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compilation failed with exit code $LASTEXITCODE."
}

$InstallerName = "FineSub-Desktop-$Version-Setup.exe"
$InstallerPath = Join-Path $OutputDirectory $InstallerName
if (-not (Test-Path -LiteralPath $InstallerPath -PathType Leaf)) {
    throw "Inno Setup completed without producing the expected installer: $InstallerPath"
}

Write-Host "FineSub Desktop installer ready:"
Write-Host $InstallerPath
