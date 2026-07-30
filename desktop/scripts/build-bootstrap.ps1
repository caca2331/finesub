[CmdletBinding()]
param(
    [string]$VenvPath = "",
    [string]$OutputDirectory = "",
    [string]$Version = "0.2.7",
    [string]$LauncherConfigPath = "",
    [string]$TrustedKeysPath = "",
    [switch]$AllowExampleUpdateConfig,
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "native-command.ps1")
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$IconPath = Join-Path $RepoRoot "desktop\assets\finesub-desktop.ico"
$LauncherVersionTemplate = Join-Path $RepoRoot "desktop\assets\finesub-desktop-version.txt"
foreach ($BrandAsset in @($IconPath, $LauncherVersionTemplate)) {
    if (-not (Test-Path -LiteralPath $BrandAsset -PathType Leaf)) {
        throw "FineSub Desktop brand asset not found: $BrandAsset"
    }
}

function Copy-PythonTree {
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
                $Item.Name.StartsWith(
                    ".tmp",
                    [System.StringComparison]::OrdinalIgnoreCase
                )
            ) {
                continue
            }
            Copy-PythonTree `
                -Source $Item.FullName `
                -Destination (Join-Path $Destination $Item.Name)
        }
        elseif ($Item.Extension -eq ".py") {
            Copy-Item `
                -LiteralPath $Item.FullName `
                -Destination $Destination `
                -Force
        }
    }
}

function Remove-BuildChild {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $ResolvedOutput = $OutputDirectory.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    $ExpectedPrefix = $ResolvedOutput + [System.IO.Path]::DirectorySeparatorChar
    if (
        -not $FullPath.StartsWith(
            $ExpectedPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "Refusing to remove a path outside the build directory: $FullPath"
    }
    if (Test-Path -LiteralPath $FullPath) {
        Remove-Item -LiteralPath $FullPath -Recurse -Force
    }
}

function New-VersionResource {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TemplatePath,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath,
        [Parameter(Mandatory = $true)]
        [string]$Version
    )

    if ($Version -notmatch "^(?<core>\d+(?:\.\d+){0,3})(?:-[0-9A-Za-z.-]+)?$") {
        throw "Invalid FineSub Desktop release version: $Version"
    }
    $Parts = @($Matches.core.Split(".") | ForEach-Object { [int]$_ })
    while ($Parts.Count -lt 4) {
        $Parts += 0
    }
    if ($Parts | Where-Object { $_ -gt 65535 }) {
        throw "Version components must fit the Windows version resource: $Version"
    }

    $Quad = $Parts -join ", "
    $FileVersion = $Parts -join "."
    $Content = Get-Content -LiteralPath $TemplatePath -Raw -Encoding UTF8
    $Content = [regex]::Replace(
        $Content,
        "filevers=\([^)]+\)",
        "filevers=($Quad)"
    )
    $Content = [regex]::Replace(
        $Content,
        "prodvers=\([^)]+\)",
        "prodvers=($Quad)"
    )
    $Content = [regex]::Replace(
        $Content,
        "StringStruct\('FileVersion', '[^']*'\)",
        "StringStruct('FileVersion', '$FileVersion')"
    )
    $Content = [regex]::Replace(
        $Content,
        "StringStruct\('ProductVersion', '[^']*'\)",
        "StringStruct('ProductVersion', '$Version')"
    )
    [System.IO.File]::WriteAllText(
        $DestinationPath,
        $Content,
        (New-Object System.Text.UTF8Encoding($false))
    )
}

if (-not $VenvPath) {
    if ($env:FINESUB_DESKTOP_VENV) {
        $VenvPath = $env:FINESUB_DESKTOP_VENV
    }
    else {
        $VenvPath = Join-Path $RepoRoot ".venv-desktop"
    }
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $RepoRoot "dist\bootstrap"
}
if (-not $LauncherConfigPath) {
    $LauncherConfigPath = Join-Path $RepoRoot "desktop\resources\launcher.json"
}
if (-not $TrustedKeysPath) {
    $TrustedKeysPath = Join-Path $RepoRoot "desktop\resources\trusted-update-keys.json"
}
if ($AllowExampleUpdateConfig) {
    if (-not (Test-Path -LiteralPath $LauncherConfigPath -PathType Leaf)) {
        $LauncherConfigPath = Join-Path $RepoRoot "desktop\resources\launcher.example.json"
    }
    if (-not (Test-Path -LiteralPath $TrustedKeysPath -PathType Leaf)) {
        $TrustedKeysPath = Join-Path $RepoRoot "desktop\resources\trusted-update-keys.example.json"
    }
}
if (-not (Test-Path -LiteralPath $LauncherConfigPath -PathType Leaf)) {
    throw "Launcher update config not found: $LauncherConfigPath"
}
if (-not (Test-Path -LiteralPath $TrustedKeysPath -PathType Leaf)) {
    throw "Trusted update keys not found: $TrustedKeysPath"
}

$VenvPath = [System.IO.Path]::GetFullPath($VenvPath)
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$OutputRoot = [System.IO.Path]::GetPathRoot($OutputDirectory)
if (
    $OutputDirectory.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    ) -eq $OutputRoot.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
) {
    throw "The build output directory cannot be a drive root."
}
if ($OutputDirectory -match "[^\u0000-\u007F]") {
    throw @"
PyInstaller build paths must contain ASCII characters only on Windows.
Use -OutputDirectory with a path such as G:\finesub-build\0.2.0.
The source repository may remain in its current Unicode path.
"@
}

# Prefer env-root python.exe (conda) over Scripts\ (venv). A Scripts hardlink
# into a conda env breaks prefix detection and resolves to the base env.
$Python = $null
foreach ($Candidate in @(
        (Join-Path $VenvPath "python.exe"),
        (Join-Path $VenvPath "Scripts\python.exe")
    )) {
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
        $Python = $Candidate
        break
    }
}
if (-not $Python) {
    throw "Desktop Python environment not found under: $VenvPath"
}
if (-not $SkipFrontend) {
    & (Join-Path $PSScriptRoot "build-frontend.ps1") -PythonPath $Python
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$StageDirectory = Join-Path $OutputDirectory ".pyinstaller-stage"
$DistDirectory = Join-Path $OutputDirectory ".pyinstaller-dist"
$WorkDirectory = Join-Path $OutputDirectory ".pyinstaller-work"
$SpecDirectory = Join-Path $OutputDirectory ".pyinstaller-spec"
$VersionResourceDirectory = Join-Path $OutputDirectory ".version-resources"
$LauncherDist = Join-Path $OutputDirectory "FineSub Desktop.dist"
foreach (
    $BuildChild in @(
        $StageDirectory,
        $DistDirectory,
        $WorkDirectory,
        $SpecDirectory,
        $VersionResourceDirectory,
        $LauncherDist
    )
) {
    Remove-BuildChild -Path $BuildChild
}

New-Item -ItemType Directory -Force -Path $VersionResourceDirectory | Out-Null
$LauncherVersionFile = Join-Path $VersionResourceDirectory "FineSub Desktop.txt"
New-VersionResource `
    -TemplatePath $LauncherVersionTemplate `
    -DestinationPath $LauncherVersionFile `
    -Version $Version
$StageDesktop = Join-Path $StageDirectory "desktop"
New-Item -ItemType Directory -Force -Path $StageDesktop | Out-Null
foreach ($EntryPoint in @("__init__.py", "FineSub.py")) {
    Copy-Item `
        -LiteralPath (Join-Path $RepoRoot "desktop\$EntryPoint") `
        -Destination $StageDesktop `
        -Force
}
Copy-PythonTree `
    -Source (Join-Path $RepoRoot "desktop\backend") `
    -Destination (Join-Path $StageDesktop "backend")

$PreviousPyInstallerConfig = $env:PYINSTALLER_CONFIG_DIR
$env:PYINSTALLER_CONFIG_DIR = Join-Path $OutputDirectory ".pyinstaller-cache"
try {
    $CommonArgs = @(
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--distpath=$DistDirectory",
        "--specpath=$SpecDirectory",
        "--paths=$StageDirectory"
    )

    $LauncherStdout = Join-Path $OutputDirectory "FineSub.pyinstaller.log"
    $LauncherStderr = Join-Path $OutputDirectory "FineSub.pyinstaller.err.log"
    $LauncherArgs = $CommonArgs + @(
        "--name=FineSub Desktop",
        "--workpath=$(Join-Path $WorkDirectory 'FineSub Desktop')",
        "--icon=$IconPath",
        "--version-file=$LauncherVersionFile",
        "--hidden-import=webview.platforms.winforms",
        "--hidden-import=webview.platforms.edgechromium",
        "--hidden-import=webview.platforms.win32",
        (Join-Path $StageDesktop "FineSub.py")
    )
    $LauncherExitCode = Invoke-NativeCommand `
        -FilePath $Python `
        -ArgumentList $LauncherArgs `
        -StdoutPath $LauncherStdout `
        -StderrPath $LauncherStderr `
        -RedactSensitiveEnvironment
    if ($LauncherExitCode -ne 0) {
        Get-Content -LiteralPath $LauncherStderr -Tail 80
        throw "FineSub launcher build failed."
    }

}
finally {
    if ($null -eq $PreviousPyInstallerConfig) {
        Remove-Item Env:PYINSTALLER_CONFIG_DIR -ErrorAction SilentlyContinue
    }
    else {
        $env:PYINSTALLER_CONFIG_DIR = $PreviousPyInstallerConfig
    }
}

Move-Item `
    -LiteralPath (Join-Path $DistDirectory "FineSub Desktop") `
    -Destination $LauncherDist

& (Join-Path $PSScriptRoot "package-bootstrap.ps1") `
    -RepoRoot $RepoRoot `
    -OutputDirectory $OutputDirectory `
    -Version $Version `
    -LauncherConfigPath $LauncherConfigPath `
    -TrustedKeysPath $TrustedKeysPath

foreach (
    $BuildChild in @(
        $StageDirectory,
        $DistDirectory,
        $WorkDirectory,
        $SpecDirectory,
        $VersionResourceDirectory,
        (Join-Path $OutputDirectory ".pyinstaller-cache")
    )
) {
    Remove-BuildChild -Path $BuildChild
}

Write-Host "FineSub Windows build completed: $LauncherDist"
