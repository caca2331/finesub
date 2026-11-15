<#
.SYNOPSIS
Publish dev's tree to the public orphan `main`, gated on CI.

.DESCRIPTION
The snapshot commit is pushed to a throwaway `ci-gate` branch first and only
lands on `main` once every workflow it triggered is green. Pushing straight to
`main` puts the first CI run *after* publication, so a failure can only be
repaired by rewriting a branch the public already fetched.

Nothing is merged, and the commit that lands on `main` is the exact commit CI
approved. `main` holds releases, checkpoints and merged external PRs only
(owner 2026-09-04; PRs 2026-09-11): when its tip is a plain checkpoint -- the
default message, and no tag pointing at it -- the new snapshot takes that
checkpoint's *parent* as its own and replaces it with `--force-with-lease` on
exactly that commit, so at most one untagged checkpoint ever sits above the last
tag. A tip that is tagged, or that carries any other message (a release waiting
for its tag, a squash-merged PR), is never rewritten: the snapshot stacks on it.
A merged PR must already be cherry-picked into $Source (`git cherry-pick -x`),
or the snapshot -- which is $Source's tree -- reverts it on `main`. So every
commit on main's first-parent line above the last snapshot this script made
(snapshots carry a `Published-From:` line) must be named by a
`cherry picked from commit` line in $Source, or nothing is published.
-KeepPrevious forces stacking. -DryRun builds and checks the snapshot tree and
stops before creating or pushing anything (it skips only the `gh` check, which
is needed for the CI gate alone). The tag check is best-effort
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
    # The public GitHub remote. `origin` is the private one, which carries the
    # unstripped `dev`; nothing here ever touches it.
    [string]$PublicRemote = "public",
    [int]$TimeoutMinutes = 45,
    # Leave the gate branch in place after a successful publish (debugging).
    [switch]$KeepGate,
    # Stack on an untagged checkpoint instead of replacing it.
    [switch]$KeepPrevious,
    # Build the snapshot tree and run every refusal check, then stop: nothing
    # is committed, pushed or moved.
    [switch]$DryRun,
    # Every workflow that must have produced a run before main may move.
    # Names are the `name:` of each file in .github/workflows.
    [string[]]$RequiredWorkflows = @("CI"),
    # Tracked on `dev` (so worktrees and fresh clones get them) but stripped
    # from every public snapshot. gitignore cannot express this: it only
    # governs untracked files, so a tracked file is tracked on every branch.
    # Deny by default -- `.claude` is where local configuration accretes, and
    # naming its members one by one would leak whatever is added next.
    [string[]]$PrivatePaths = @(
        ".claude", "docs/archive", "docs/reports",
        # Plan drafts are the maintainer's own ledger -- read from a worktree or
        # a dev clone, never published -- owner 2026-09-11. What a plan still
        # owes the public side was distilled into the owning docs first; see
        # the private-notes section of docs/README.md. test_doc_links' LOCAL_ONLY
        # must carry the same prefix or the gate branch turns red on links dev
        # resolves fine; test_publish_filter checks that lockstep.
        "docs/plans",
        # The maintainer-only task document. Verified 2026-09-01 against
        # `main`'s whole history: it has never been published, and moving it
        # out of `.claude/` must not be what publishes it.
        "agent-tasks/release",
        # The prompt cold-read task: its ledger holds rendered prompts and
        # review findings about unreleased prompt work, and its renderer reads
        # private fixtures -- maintainer tooling with no public use.
        "agent-tasks/prompt-coldread",
        # How the maintainer runs quality comparisons: built from their own
        # rulings on private experiments, and names private material.
        "agent-tasks/quality-compare",
        # The audit task itself ships (AGENTS.md cites it), but its evals
        # name real material: BV ids, a streamer, and a local path under
        # `data/`. They also cannot run in a public tree, which ships
        # neither `data/` nor `out/` -- exposure without a use.
        "agent-tasks/run-audit/evals",
        # One-off experiments (owner 2026-10-23, the personal project
        # conventions' `explore/`): tracked so worktrees and clones carry
        # them, stripped because their scripts hard-code local paths and
        # name real material. Public docs name the directory, not entries.
        "explore",
        # The maintainer's private supplement to AGENTS.md (test authorisation,
        # this publishing procedure, the plan index).
        "AGENTS.local.md"
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

if (-not $DryRun -and -not (Get-Command gh -ErrorAction SilentlyContinue)) {
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

# The pre-push hook is what stops a stray `git push public dev`, and a clone on
# another machine starts without it: hooks are not cloned. This script is the
# one place that runs on every publishing clone, so it refuses until the hook
# is in place -- under -DryRun too, which is how a new machine finds out.
$PublicRepo = "caca2331/finesub"
$publicUrl = (Invoke-Git @("remote", "-v")) | Where-Object { $_ -match "^$([regex]::Escape($PublicRemote))\s+(\S+)\s+\(push\)$" } | ForEach-Object { $Matches[1] }
if (-not $publicUrl) {
    throw "No remote named '$PublicRemote'. Add it: git remote add $PublicRemote https://github.com/$PublicRepo.git"
}
if ($publicUrl -notmatch 'github\.com[:/]([^/]+/[^/]+?)(\.git)?/?$' -or $Matches[1] -ne $PublicRepo) {
    throw "Remote '$PublicRemote' points at $publicUrl, not github.com/$PublicRepo; nothing was published."
}
# Read from the full listing (keys come out lower-cased): `config --get` exits 1
# when the key is unset, which Invoke-Git would treat as a failure.
if (Invoke-Git @("config", "--list") | Where-Object { $_ -match '^core\.hookspath=.' }) {
    throw "core.hooksPath is set, so the pre-push hook in the git directory never runs. Unset it before publishing."
}
$HookPath = Join-Path $CommonDir "hooks/pre-push"
# $Source's copy, not the working tree's: the checkout may be on another branch.
$expectedHook = (Invoke-Git @("show", "${Source}:scripts/pre-push")) -join "`n"
if (-not (Test-Path -LiteralPath $HookPath) -or
        ((Get-Content -LiteralPath $HookPath -Raw) -replace "`r`n", "`n").TrimEnd("`n") -cne $expectedHook.TrimEnd("`n")) {
    throw @"
The pre-push hook is missing or differs from $Source's scripts/pre-push. With
$Source checked out, install it (a copy, not a PowerShell redirect: that would
re-encode the file or give it CRLF endings, and sh would then fail on it):
    Copy-Item scripts/pre-push "`$(git rev-parse --git-common-dir)/hooks/pre-push"
"@
}
# git skips a hook without the execute bit, saying at most a hint. Not
# $IsWindows: Windows PowerShell 5.1 has no such variable, and $null would
# send it down this branch on every run.
if ([Environment]::OSVersion.Platform -ne 'Win32NT' -and (Get-Item -LiteralPath $HookPath).UnixMode -notmatch '^-..x') {
    throw "The pre-push hook is not executable, so git skips it: chmod +x `"$HookPath`""
}

Invoke-Git @("fetch", $PublicRemote) | Out-Null
$localMain = Invoke-GitLine @("rev-parse", "main")
$remoteMain = Invoke-GitLine @("rev-parse", "$PublicRemote/main")
if ($localMain -ne $remoteMain) {
    throw @"
Local main ($localMain) and $PublicRemote/main ($remoteMain) disagree. Reset the local
ref to the remote one before publishing:
    git update-ref refs/heads/main $remoteMain
"@
}

# Every external PR merged on GitHub must already be in $Source, or this
# snapshot -- $Source's tree -- silently reverts it on main. main and $Source
# share no history, so the only evidence is the `cherry picked from commit`
# line `git cherry-pick -x` leaves. What counts as an external PR: every commit
# on main's first-parent line above the last snapshot this script made (its
# message carries `Published-From:`). First parent, so a PR merged with a merge
# commit counts once, as the commit `git cherry-pick -x -m 1` records.
#
# $LegacyBaseline stands in for that last snapshot until one carries the line:
# snapshots made before 2026-09-24 do not. It is main's tip on that day (the
# 0.5.1 release); the PRs below it (#1 #5 #7 #8) were published over by later
# snapshots long ago, so checking them now could only refuse, never rescue.
# Delete it once a snapshot with `Published-From:` is on main.
$LegacyBaseline = "b9b2f10c80abd58ddff739b60461624f51c0789c"
$unreconciled = @()
$reachedBase = $false
foreach ($commit in Invoke-Git @("rev-list", "--first-parent", $remoteMain)) {
    if ($commit -eq $LegacyBaseline) { $reachedBase = $true; break }
    $body = (Invoke-Git @("log", "-1", "--format=%B", $commit)) -join "`n"
    if ($body -match '(?m)^Published-From: ') { $reachedBase = $true; break }
    $picked = @(Invoke-Git @("log", $Source, "--format=%H", "-1", "--fixed-strings", "--grep=cherry picked from commit $commit"))
    if (-not $picked) { $unreconciled += $commit }
}
if (-not $reachedBase) {
    throw "Walked all of main without meeting a snapshot marked Published-From: or the legacy baseline $LegacyBaseline; refusing rather than treating every commit as an unreconciled PR."
}
if ($unreconciled.Count -gt 0) {
    $picks = ($unreconciled | ForEach-Object { "    git cherry-pick -x $_" }) -join "`n"
    throw @"
main carries commits that $Source does not, so this snapshot would revert them
(external PRs merged on GitHub). Bring each into $Source first -- add -m 1 for a
merge commit -- then run this again:
$picks
If a change is already in $Source by some other route (its commits picked one
by one, or rewritten), record that instead with an empty commit:
    git commit --allow-empty -m "Reconcile external PR (cherry picked from commit <sha>)"
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
    foreach ($line in Invoke-Git @("ls-remote", "--tags", $PublicRemote)) {
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
# go blind together and read as clean. It used to warn; a warning scrolls past,
# and the one case it exists for is a rename whose new name is now published
# whole (owner 2026-09-24: refuse). An entry for something that does not exist
# yet has no use, so there is no legitimate empty match.
foreach ($path in $PrivatePaths) {
    if (-not @(Invoke-Git @("ls-tree", "-r", "--name-only", "$Source^{tree}", "--", $path))) {
        throw "`$PrivatePaths entry '$path' matches nothing in ${Source}: renamed, or misspelled? Its new name, if it has one, would be published. Fix the list; nothing was pushed."
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
if ($DryRun) {
    Write-Host "Dry run: the snapshot tree is $tree on parent $parent; every check passed. Nothing was committed or pushed."
    return
}
# The trailer is how the next run tells this script's snapshots from external
# PRs merged on GitHub (see the check after the fetch).
$sourceHead = Invoke-GitLine @("rev-parse", $Source)
$snapshot = Invoke-GitLine @("commit-tree", $tree, "-p", $parent, "-m", $Message, "-m", "Published-From: $sourceHead")
if ($replacing) {
    Write-Host "Replacing checkpoint $replacing (untagged, default message); parent is $parent"
}

Write-Host "Snapshot $snapshot -> $GateBranch (gating on CI)"
Invoke-Git @("push", "--force", $PublicRemote, "${snapshot}:refs/heads/$GateBranch") | Out-Null

function Get-GateRuns {
    param([Parameter(Mandatory = $true)][string]$Sha)

    $body = & gh run list --repo $PublicRepo --branch $GateBranch --limit 20 --json databaseId,name,headSha,status,conclusion,url
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
    Invoke-Git @("push", "--force-with-lease=refs/heads/main:$replacing", $PublicRemote, "${snapshot}:refs/heads/main") | Out-Null
} else {
    Invoke-Git @("push", $PublicRemote, "${snapshot}:refs/heads/main") | Out-Null
}
Invoke-Git @("update-ref", "refs/heads/main", $snapshot) | Out-Null
if (-not $KeepGate) {
    Invoke-Git @("push", $PublicRemote, "--delete", $GateBranch) | Out-Null
}

if ($replacing) {
    Write-Host "main is now $snapshot ($Message), replacing checkpoint $replacing"
} else {
    Write-Host "main is now $snapshot ($Message)"
}
