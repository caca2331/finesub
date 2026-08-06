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
& $Uv tool install --force finesub
if ($LASTEXITCODE -ne 0) {
    throw "uv tool install 失败，退出码 $LASTEXITCODE。"
}
Write-Host ""
Write-Host "完成。运行 ``finesub --help`` 开始使用（提示找不到命令的话，开一个新终端，"
Write-Host "或先跑 ``uv tool update-shell`` 把 uv 的 bin 目录加入 PATH）。"
