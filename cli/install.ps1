# FineSub CLI installer.
#
# Installs uv when missing, then installs (or upgrades: safe to re-run) the
# `finesub` tool from PyPI. Only needed on machines without uv -- with uv on
# PATH this whole script is just `uv tool install finesub`.
#
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/caca2331/finesub/main/cli/install.ps1 | iex"

$ErrorActionPreference = "Stop"

function Resolve-Uv {
    $existing = Get-Command uv -ErrorAction SilentlyContinue
    if ($existing) {
        return $existing.Source
    }
    Write-Host "uv 未安装，先安装 uv（https://docs.astral.sh/uv/）..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $candidates = @()
    if ($env:UV_INSTALL_DIR) {
        $candidates += (Join-Path $env:UV_INSTALL_DIR "uv.exe")
    }
    if ($env:XDG_BIN_HOME) {
        $candidates += (Join-Path $env:XDG_BIN_HOME "uv.exe")
    }
    $candidates += (Join-Path $env:USERPROFILE ".local\bin\uv.exe")
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    throw "uv 安装完成但未找到 uv.exe；请开一个新终端重跑本脚本。"
}

$Uv = Resolve-Uv
Write-Host "安装 FineSub CLI ..."

# This runs before FineSub exists, so the automatic region detection -- which
# lives in the Python it is about to install -- cannot help here. Only the
# explicit override applies; from the first `finesub` command onwards the
# route is resolved normally.
$InstallArgs = @("tool", "install", "--force")
if ($env:FINESUB_PYPI_INDEX) {
    # Host only. An index may carry credentials (https://user:token@host/simple)
    # and this line lands in terminal scrollback and CI logs.
    $IndexHost = $env:FINESUB_PYPI_INDEX
    try { $IndexHost = ([System.Uri]$env:FINESUB_PYPI_INDEX).Host } catch {}
    Write-Host "使用指定的 PyPI 源：$IndexHost"
    $InstallArgs += @("--default-index", $env:FINESUB_PYPI_INDEX)
}
& $Uv @InstallArgs finesub
if ($LASTEXITCODE -ne 0) {
    throw "uv tool install 失败，退出码 $LASTEXITCODE。"
}

# Settle where the big files go while the user is still here, but download
# nothing: a full `finesub setup` fetches several GB, which would turn this
# one-line install into a multi-minute one. The runtime is built on first use.
#
# By absolute path, never by name. uv puts its tool directory on the *user*
# PATH, which the already-running process does not see -- so on a machine that
# just got uv, `finesub` is not a command yet. With ErrorActionPreference
# "Stop" that is a terminating error, and the install would abort here without
# ever printing how to get started. Resolve-Uv solves the same problem for uv
# itself a few lines up.
Write-Host ""
$FineSub = $null
try {
    $BinDir = (& $Uv tool dir --bin 2>$null | Select-Object -First 1)
    if ($BinDir) {
        $Candidate = Join-Path $BinDir.Trim() "finesub.exe"
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) { $FineSub = $Candidate }
    }
} catch {}
if (-not $FineSub) {
    $FineSub = (Get-Command finesub -ErrorAction SilentlyContinue).Source
}
if ($FineSub) {
    & $FineSub setup --dirs-only
    if ($LASTEXITCODE -ne 0) {
        Write-Host "（未能记录数据目录，首次运行时会再问一次。）"
    }
} else {
    Write-Host "（找不到刚安装的 finesub，数据目录会在首次运行时询问。）"
}
Write-Host ""
Write-Host "完成。运行 ``finesub --help`` 开始使用（提示找不到命令的话，开一个新终端，"
Write-Host "或先跑 ``uv tool update-shell`` 把 uv 的 bin 目录加入 PATH）。"
