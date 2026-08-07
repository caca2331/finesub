<#
.SYNOPSIS
Publish dev's tree to the public orphan `main`, gated on CI.

.DESCRIPTION
The snapshot commit is pushed to a throwaway `ci-gate` branch first and only
fast-forwarded onto `main` once every workflow it triggered is green. Pushing
straight to `main` puts the first CI run *after* publication, so a failure can
only be repaired by force-pushing over a branch the public already fetched.

Nothing is rewritten and nothing is merged: the commit that lands on `main` is
the exact commit CI approved, and its parent is the previous `main`.
#>
[CmdletBinding()]
param(
    [string]$Message = "chore: snapshot dev into main",
    [string]$Source = "dev",
    [string]$GateBranch = "ci-gate",
    [int]$TimeoutMinutes = 45,
    # Leave the gate branch in place after a successful publish (debugging).
    [switch]$KeepGate
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    # git reports ordinary progress on stderr, so merging the streams must not
    # trip $ErrorActionPreference by itself: the exit code is the verdict.
    $ErrorActionPreference = "Continue"
    $PSNativeCommandUseErrorActionPreference = $false
    $output = @(& git @Arguments 2>&1 | ForEach-Object { [string]$_ })
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed:`n$($output -join "`n")"
    }
    return $output
}

function Invoke-GitLine {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    # Indexing the raw result would slice a single-line string into characters.
    return (@(Invoke-Git -Arguments $Arguments)[0]).Trim()
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "The GitHub CLI (gh) is required to read the gate's CI result."
}

$RepoRoot = Invoke-GitLine @("rev-parse", "--show-toplevel")
Set-Location -LiteralPath $RepoRoot

if (Invoke-Git @("status", "--porcelain")) {
    throw "The working tree is dirty; commit or stash before publishing."
}

Invoke-Git @("fetch", "origin") | Out-Null
$localMain = Invoke-GitLine @("rev-parse", "main")
$remoteMain = Invoke-GitLine @("rev-parse", "origin/main")
if ($localMain -ne $remoteMain) {
    throw @"
Local main ($localMain) and origin/main ($remoteMain) disagree. Reset the local
ref to the remote one before publishing:
    git update-ref refs/heads/main $remoteMain
"@
}

# The snapshot: dev's tree, the published main as parent. Never a merge --
# merging the orphan line back into dev is what this layout exists to avoid.
$tree = Invoke-GitLine @("rev-parse", "$Source^{tree}")
if ($tree -eq (Invoke-GitLine @("rev-parse", "main^{tree}"))) {
    Write-Host "main already carries $Source's tree; nothing to publish."
    return
}
$snapshot = Invoke-GitLine @("commit-tree", $tree, "-p", $localMain, "-m", $Message)

Write-Host "Snapshot $snapshot -> $GateBranch (gating on CI)"
Invoke-Git @("push", "--force", "origin", "${snapshot}:refs/heads/$GateBranch") | Out-Null

function Get-GateRuns {
    param([Parameter(Mandatory = $true)][string]$Sha)

    $body = & gh run list --branch $GateBranch --limit 20 --json databaseId,name,headSha,status,conclusion,url
    if ($LASTEXITCODE -ne 0) {
        # A transient API failure is not a verdict; the caller polls again.
        return $null
    }
    $runs = $body | ConvertFrom-Json
    return @($runs | Where-Object { $_.headSha -eq $Sha })
}

$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
# Both workflows are registered within seconds of the push, but not at the same
# instant: judging completeness before that window closes could pass a snapshot
# on the strength of whichever workflow happened to register first.
$settleUntil = (Get-Date).AddSeconds(60)
$runs = @()
while ($true) {
    if ((Get-Date) -gt $deadline) {
        throw "CI did not finish within $TimeoutMinutes minutes; gate branch $GateBranch is left in place."
    }
    Start-Sleep -Seconds 15
    $current = Get-GateRuns -Sha $snapshot
    if ($null -eq $current -or $current.Count -eq 0) {
        continue
    }
    $runs = $current
    $pending = @($runs | Where-Object { $_.status -ne "completed" })
    if ($pending.Count -gt 0) {
        Write-Host ("  waiting: " + (($pending | ForEach-Object { "$($_.name) [$($_.status)]" }) -join ", "))
        continue
    }
    if ((Get-Date) -lt $settleUntil) {
        continue
    }
    break
}

$failed = @($runs | Where-Object { $_.conclusion -ne "success" })
foreach ($run in $runs) {
    Write-Host ("  {0}: {1}  {2}" -f $run.name, $run.conclusion, $run.url)
}
if ($failed.Count -gt 0) {
    throw @"
CI is red on the snapshot; main was not moved. Fix it on $Source, commit, and
run this script again -- the gate branch is rewritten, main never was.
"@
}

# Fast-forward: the approved commit itself, whose parent is the published main.
Invoke-Git @("push", "origin", "${snapshot}:refs/heads/main") | Out-Null
Invoke-Git @("update-ref", "refs/heads/main", $snapshot) | Out-Null
if (-not $KeepGate) {
    Invoke-Git @("push", "origin", "--delete", $GateBranch) | Out-Null
}

Write-Host "main is now $snapshot ($Message)"
