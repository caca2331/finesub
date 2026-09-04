<#
.SYNOPSIS
Publish dev's tree to the public orphan `main`, gated on CI.

.DESCRIPTION
The snapshot commit is pushed to a throwaway `ci-gate` branch first and only
lands on `main` once every workflow it triggered is green. Pushing straight to
`main` puts the first CI run *after* publication, so a failure can only be
repaired by rewriting a branch the public already fetched.

Nothing is merged, and the commit that lands on `main` is the exact commit CI
approved. `main` holds releases and checkpoints only (owner 2026-09-04): when
its tip is a plain checkpoint -- the default message, and no tag pointing at
it -- the new snapshot takes that checkpoint's *parent* as its own and replaces
it with `--force-with-lease` on exactly that commit, so at most one untagged
commit ever sits above the last tag. A tip that is tagged, or that carries any
other message (a release waiting for its tag), is never rewritten: the snapshot
stacks on it. -KeepPrevious forces stacking. The tag check is best-effort
(remote tags read at the start and again just before the push): a tag created
in the instant between that last read and the push is not caught. This is a
one-maintainer repository; if that ever changes, the guarantee needs a branch
protection rule or a publish lock, not more checks here.

The published tree is `dev`'s minus $PrivatePaths -- local notes and Claude
Code configuration that every worktree should have but the public should not.
CI runs on that filtered tree, so anything the public tree needs and no longer
has turns the gate red before `main` moves.
#>
[CmdletBinding()]
param(
    [string]$Message = "chore: snapshot dev into main",
    [string]$Source = "dev",
    [string]$GateBranch = "ci-gate",
    [int]$TimeoutMinutes = 45,
    # Leave the gate branch in place after a successful publish (debugging).
    [switch]$KeepGate,
    # Stack on an untagged checkpoint instead of replacing it.
    [switch]$KeepPrevious,
    # Every workflow that must have produced a run before main may move.
    # Names are the `name:` of each file in .github/workflows.
    [string[]]$RequiredWorkflows = @("CI"),
    # Tracked on `dev` (so worktrees and fresh clones get them) but stripped
    # from every public snapshot. gitignore cannot express this: it only
    # governs untracked files, so a tracked file is tracked on every branch.
    # Deny by default -- `.claude` is where local configuration accretes, and
    # naming its members one by one would leak whatever is added next.
    [string[]]$PrivatePaths = @(
        ".claude", "docs/archive", "docs/report",
        # The maintainer-only task document. Verified 2026-09-01 against
        # `main`'s whole history: it has never been published, and moving it
        # out of `.claude/` must not be what publishes it.
        "agent-tasks/release",
        # The audit task itself ships (CLAUDE.md cites it), but its evals
        # name real material: BV ids, a streamer, and a local path under
        # `data/`. They also cannot run in a public tree, which ships
        # neither `data/` nor `out/` -- exposure without a use.
        "agent-tasks/run-audit/evals"
    ),
    # Nothing is carved back out today: the agent task documents moved to
    # `agent-tasks/` (2026-09-01), which is a public path in its own right
    # rather than a hole punched in a private one.
    [string[]]$PublicExceptions = @()
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

# Refuse to publish from a linked worktree. Refs are shared across worktrees,
# so `$Source^{tree}` below reads the *main checkout's* dev while the clean-tree
# check above validates the worktree you are standing in: a clean worktree
# would quietly publish work that is not in front of you, and the output would
# look entirely normal.
$GitDir = Invoke-GitLine @("rev-parse", "--git-dir")
$CommonDir = Invoke-GitLine @("rev-parse", "--git-common-dir")
if ((Resolve-Path -LiteralPath $GitDir).Path -ne (Resolve-Path -LiteralPath $CommonDir).Path) {
    throw @"
This is a linked worktree. Refs are shared, so publishing here would snapshot
the main checkout's $Source rather than this tree. Run the script from the
main checkout.
"@
}

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

# Which commit the snapshot stands on: main itself, or -- when main's tip is
# an untagged checkpoint -- its parent, so the checkpoint is replaced rather
# than stacked on (see .DESCRIPTION). Decided before anything is built or
# pushed, and re-checked at the final push with --force-with-lease, so a tag
# or a second publisher arriving in between turns into a refusal, not a loss.
$CheckpointMessage = "chore: snapshot dev into main"

function Get-RemoteTagsAt {
    param([Parameter(Mandatory = $true)][string]$Sha)

    # The remote's tags, not this clone's: a tag pushed from another machine
    # is what makes a commit a release, and this clone may never have fetched
    # it. Annotated tags list twice (the tag object and its `^{}` target); the
    # target is the line that matters.
    $tags = @()
    foreach ($line in Invoke-Git @("ls-remote", "--tags", "origin")) {
        $parts = $line -split "`t"
        if ($parts.Count -eq 2 -and $parts[0] -eq $Sha) {
            $tags += $parts[1] -replace '^refs/tags/', '' -replace '\^\{\}$', ''
        }
    }
    return @($tags | Sort-Object -Unique)
}

$parent = $localMain
$replacing = $null
if (-not $KeepPrevious) {
    $tipSubject = Invoke-GitLine @("log", "-1", "--format=%s", $localMain)
    $tipTags = Get-RemoteTagsAt $localMain
    if ($tipSubject -eq $CheckpointMessage -and $tipTags.Count -eq 0) {
        $parent = Invoke-GitLine @("rev-parse", "$localMain^")
        $replacing = $localMain
    }
}

# The snapshot: dev's tree minus $PrivatePaths, on the parent chosen above.
# Never a merge -- merging the orphan line back into dev is what this layout
# exists to avoid.
#
# The filtering runs in a throwaway index, so neither the real index nor the
# working tree is touched: read dev's tree into it, drop the private paths,
# read the public exceptions back, and write the result out as a new tree.
$IndexFile = Join-Path ([IO.Path]::GetTempPath()) ("finesub-publish-" + [Guid]::NewGuid().ToString("N") + ".index")
$PreviousIndex = $env:GIT_INDEX_FILE
try {
    $env:GIT_INDEX_FILE = $IndexFile
    Invoke-Git @("read-tree", "$Source^{tree}") | Out-Null
    Invoke-Git (@("rm", "-r", "--cached", "-f", "--ignore-unmatch", "-q", "--") + $PrivatePaths) | Out-Null
    foreach ($kept in $PublicExceptions) {
        # An exception equal to the path it carves from puts everything back,
        # and the leak check below cannot object: it reads $PublicExceptions as
        # the statement of intent. Nullifying an entry this way rather than
        # deleting it leaves a list that reads as protection and is not.
        if ($PrivatePaths -contains $kept) {
            throw "`$PublicExceptions entry '$kept' carves out the whole of the private path by the same name, which publishes all of it. Drop it from `$PrivatePaths instead; nothing was pushed."
        }
        # Resolved first so a renamed exception says so. Left to read-tree the
        # failure is `fatal: Needed a single revision`, which names neither the
        # path nor the parameter it came from. A file rather than a directory
        # still fails inside read-tree, loudly and without publishing.
        #
        # Through Invoke-Git rather than a bare `& git` plus an exit-code test:
        # the caller's profile may set $PSNativeCommandUseErrorActionPreference,
        # and then a failing native command throws before any such test runs --
        # which is what that function neutralises for every other call here.
        try { Invoke-Git @("rev-parse", "--verify", "${Source}^{tree}:$kept") | Out-Null }
        catch { throw "`$PublicExceptions entry '$kept' does not exist in $Source; nothing was pushed." }
        Invoke-Git @("read-tree", "--prefix=$kept/", "${Source}^{tree}:$kept") | Out-Null
    }
    $tree = Invoke-GitLine @("write-tree")
} finally {
    if ($null -eq $PreviousIndex) { Remove-Item Env:GIT_INDEX_FILE -ErrorAction SilentlyContinue }
    else { $env:GIT_INDEX_FILE = $PreviousIndex }
    Remove-Item -LiteralPath $IndexFile -ErrorAction SilentlyContinue
}

# A stale entry -- a directory since renamed, or a typo -- removed nothing
# above and guards nothing below, because both read this same list: they would
# go blind together and read as clean. Nothing here
# can tell a renamed path from one that simply does not exist yet, so this
# warns rather than throws, at the moment someone is watching the output.
foreach ($path in $PrivatePaths) {
    if (-not @(Invoke-Git @("ls-tree", "-r", "--name-only", "$Source^{tree}", "--", $path))) {
        Write-Warning "`$PrivatePaths entry '$path' matches nothing in ${Source}: renamed, or misspelled? That name is protecting nothing."
    }
}

# The list is only as good as what it actually removed, and the failure mode
# here is publishing private content to a branch that is never force-pushed.
# So verify the tree itself rather than trusting the parameters.
$published = @(Invoke-Git @("ls-tree", "-r", "--name-only", $tree))
# Ordinal: `StartsWith(string)` compares by the current culture, which folds
# characters a path separator never should.
$leaked = @($published | Where-Object {
    $name = $_
    $under = { param($prefix) $name -eq $prefix -or $name.StartsWith("$prefix/", [StringComparison]::Ordinal) }
    if (-not @($PrivatePaths | Where-Object { & $under $_ })) { return $false }
    return -not @($PublicExceptions | Where-Object { & $under $_ })
})
if ($leaked.Count -gt 0) {
    throw @"
The snapshot still carries private paths, so nothing was pushed:
$($leaked -join "`n")
"@
}

# `diff-tree`, not `diff`: the porcelain honours whatever diff config the
# publishing machine happens to carry, and this line is a receipt.
$stripped = @(Invoke-Git @("diff-tree", "-r", "--name-only", $tree, "$Source^{tree}"))
if ($stripped.Count -gt 0) {
    Write-Host "Stripped from the public snapshot ($($stripped.Count) files):"
    $groups = $stripped | ForEach-Object {
        $parts = $_ -split '/'
        if ($parts.Count -gt 1) { $parts[0..1] -join '/' } else { $parts[0] }
    }
    foreach ($group in ($groups | Sort-Object -Unique)) {
        Write-Host "  $group"
    }
}

if ($tree -eq (Invoke-GitLine @("rev-parse", "main^{tree}"))) {
    # Compared after filtering: a $Source commit that only touched private
    # paths produces the tree main already has, and publishing it would spend
    # a CI run to add a commit that changes nothing.
    Write-Host "main already carries $Source's public tree; nothing to publish."
    return
}
$snapshot = Invoke-GitLine @("commit-tree", $tree, "-p", $parent, "-m", $Message)
if ($replacing) {
    Write-Host "Replacing checkpoint $replacing (untagged, default message); parent is $parent"
}

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

# Green is not enough: the gate has to know *which* workflows ran. It only
# ever asked that every run matching the sha had succeeded, so a workflow that
# produced no run at all -- renamed, `ci-gate` dropped from its `on.push`
# branches, or simply broken YAML -- was indistinguishable from one that
# passed. The root suite is the only place a Linux-specific failure shows up,
# and it could have silently stopped being part of the gate.
$missing = @($RequiredWorkflows | Where-Object { $_ -notin @($runs | ForEach-Object { $_.name }) })
if ($missing.Count -gt 0) {
    throw @"
Expected CI workflows did not run on the snapshot: $($missing -join ', ').
Green here would mean nothing -- a workflow that produces no run looks exactly
like one that passed. Check .github/workflows (name, and `ci-gate` in
on.push.branches); main was not moved.
"@
}

# The approved commit itself. A fast-forward when it stands on main; when it
# replaces a checkpoint, a lease on exactly that checkpoint -- if main moved
# meanwhile (a tag cannot move it, but another publisher can) the push is
# refused and nothing is lost.
if ($replacing) {
    # The gate can take most of an hour, and a tag pushed meanwhile does not
    # move main -- so the lease alone would still let this rewrite a commit
    # that has since become a release. Ask the remote again first.
    $taggedMeanwhile = Get-RemoteTagsAt $replacing
    if ($taggedMeanwhile.Count -gt 0) {
        throw @"
$replacing was tagged ($($taggedMeanwhile -join ', ')) while the gate ran, so it
is a release now and is not replaced. main was not moved; the approved snapshot
is $snapshot on $GateBranch. Run the script again: it will stack on the tag.
"@
    }
    Invoke-Git @("push", "--force-with-lease=refs/heads/main:$replacing", "origin", "${snapshot}:refs/heads/main") | Out-Null
} else {
    Invoke-Git @("push", "origin", "${snapshot}:refs/heads/main") | Out-Null
}
Invoke-Git @("update-ref", "refs/heads/main", $snapshot) | Out-Null
if (-not $KeepGate) {
    Invoke-Git @("push", "origin", "--delete", $GateBranch) | Out-Null
}

if ($replacing) {
    Write-Host "main is now $snapshot ($Message), replacing checkpoint $replacing"
} else {
    Write-Host "main is now $snapshot ($Message)"
}
