<#
.SYNOPSIS
  Release gate: every hand-pinned external binary URL still resolves.

.DESCRIPTION
  The dependency resolvers keep pythonhosted / HuggingFace / mirror URLs honest --
  those are generated and do not silently vanish. What does vanish is anything a
  human pinned by hand:

    * assets under our own github.com/caca2331/finesub releases (the CT2 wheel,
      tokcount). 0.4.0 shipped pinning a CT2 wheel whose release was never
      created, so every fresh install 404'd at the ASR step.
    * hand-picked third-party pins (uv / ffmpeg / MinGit / yt-dlp in
      runtime-manifest.json, the separator + Whisper + Qwen files in
      model-manifest.json). Upstream retags and deletes these.

  No CI job fetches any of them: ci.yml installs [harness,dev] and skips [asr]
  on purpose, and its Windows jobs never download models or tools. So this is
  the only thing standing between a forgotten upload and a broken release.

  For our own release assets it also cross-checks GitHub's recorded asset digest
  against the sha256 the lock files pin, which catches a re-upload of a different
  build under the same tag without downloading the file.

  One manifest entry pins no digest on purpose: ffmpeg carries `digest_from`,
  because BtbN replaces the bytes behind its `latest` tag on every build (see
  docs/cli-bootstrap-logging-download-plan.md 5.6). A reachable URL proves
  nothing there, so for those the resolution itself is exercised -- if the
  release API stops reporting a digest, provisioning breaks at install time and
  this is where that shows up.

.PARAMETER SkipDigest
  Skip the `gh`-based digest cross-check (URL reachability only).
#>
[CmdletBinding()]
param(
  [switch] $SkipDigest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$ownRepoPrefix = 'https://github.com/caca2331/finesub/releases/download/'

# Files that carry hand-pinned URLs. Anything resolver-generated in the pylocks
# is deliberately out of scope except our own-repo asset, which is matched by
# prefix below.
$pinnedFiles = @(
  'pyproject.toml',
  'src/finesub_bootstrap/pylock.win-py312.toml',
  'src/finesub_bootstrap/pylock.win-py312.cn.toml',
  'src/finesub_bootstrap/runtime-manifest.json',
  'src/finesub_bootstrap/model-manifest.json'
)

# A concrete file to fetch, not a mirror base or a geo-lookup endpoint.
$assetPattern = '(?<url>https?://[^"''\s\\]+\.(?:whl|zip|exe|tar\.gz|ckpt|yaml|json|bin|safetensors))'

$targets = @{}
foreach ($relative in $pinnedFiles) {
  $path = Join-Path $repoRoot $relative
  if (-not (Test-Path -LiteralPath $path)) {
    throw "pinned-URL source is missing: $relative"
  }
  $lines = @(Get-Content -LiteralPath $path)
  for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = $lines[$i]
    foreach ($match in [regex]::Matches($line, $assetPattern)) {
      $url = $match.Groups['url'].Value
      $isOwn = $url.StartsWith($ownRepoPrefix)
      # Resolver-generated entries (pythonhosted, mirrors, HF inside a pylock)
      # are out of scope; our own asset inside a pylock is not.
      if ($relative -like '*pylock*' -and -not $isOwn) { continue }

      if (-not $targets.ContainsKey($url)) {
        $targets[$url] = [pscustomobject]@{
          Url       = $url
          Own       = $isOwn
          Source    = New-Object System.Collections.ArrayList
          Sha256    = $null
          Resolvable = $false
        }
      }
      if ($targets[$url].Source -notcontains $relative) {
        [void]$targets[$url].Source.Add($relative)
      }
      # Find the digest that belongs to this URL. The pylocks put it on the same
      # line (`archive = { url = "...", hashes = { sha256 = "..." } }`); the JSON
      # manifests put it one or two lines below, inside the same asset object.
      # Stop at the next url so a digest is never borrowed from the next entry.
      for ($j = $i; $j -lt [math]::Min($i + 4, $lines.Count); $j++) {
        if ($j -gt $i -and $lines[$j] -match '"?url"?\s*[:=]') { break }
        if (-not $targets[$url].Sha256) {
          $sha = [regex]::Match($lines[$j], 'sha256"?\s*[:=]\s*"(?<h>[0-9a-f]{64})"')
          if ($sha.Success) { $targets[$url].Sha256 = $sha.Groups['h'].Value }
        }
        # `digest_from` means the manifest deliberately pins no digest and asks
        # the release API for it at install time. Reachability alone does not
        # prove that entry works, so the resolution gets exercised below.
        if ($lines[$j] -match '"digest_from"\s*:\s*"github-release-api"') {
          $targets[$url].Resolvable = $true
        }
      }
    }
  }
}

if ($targets.Count -eq 0) {
  throw 'no pinned URLs found -- the extraction pattern or the file list has drifted'
}

function Test-Url {
  param([string] $Url)
  # HEAD first; some hosts answer 405, so fall back to a one-byte ranged GET.
  $code = & curl.exe -sIL --max-time 30 -o NUL -w '%{http_code}' $Url
  if ($code -notin @('200', '206')) {
    $code = & curl.exe -sL --max-time 30 -r 0-0 -o NUL -w '%{http_code}' $Url
  }
  return $code
}

function Get-ReleaseAssetRef {
  # https://github.com/<owner>/<repo>/releases/download/<tag>/<name>
  param([string] $Url)
  $pattern = '^https://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/([^/]+)$'
  $match = [regex]::Match($Url, $pattern)
  if (-not $match.Success) { return $null }
  return [pscustomobject]@{
    Owner = $match.Groups[1].Value
    Repo  = $match.Groups[2].Value
    # The download URL percent-encodes `+` as %2B; the API wants the real name.
    Tag   = [uri]::UnescapeDataString($match.Groups[3].Value)
    Name  = [uri]::UnescapeDataString($match.Groups[4].Value)
  }
}


$failures = New-Object System.Collections.ArrayList
$ownFailed = $false
$results = New-Object System.Collections.ArrayList

foreach ($target in $targets.Values | Sort-Object -Property @{Expression = { -not $_.Own } }, Url) {
  $code = Test-Url -Url $target.Url
  $ok = $code -in @('200', '206')
  if (-not $ok) {
    [void]$failures.Add("$code  $($target.Url)  [$($target.Source -join ', ')]")
    if ($target.Own) { $ownFailed = $true }
  }
  [void]$results.Add([pscustomobject]@{
      Status = if ($ok) { "ok $code" } else { "FAIL $code" }
      Scope  = if ($target.Own) { 'own-release' } else { 'third-party' }
      Pin    = if ($target.Resolvable) { 'api-digest' } else { 'pinned' }
      Asset  = Split-Path -Leaf ($target.Url -replace '\?.*$', '')
    })
}

$results | Format-Table -AutoSize | Out-String -Width 200 | Write-Host

# Ask the release API about the assets whose digest matters. Two questions, one
# API call each, neither downloading a byte:
#   * pinned own-release assets -- does the tag still hold the bytes the locks
#     pin? (catches a re-upload of a different build under the same tag)
#   * `digest_from` assets -- can the digest still be resolved at all? For those
#     the manifest pins nothing, so a reachable URL proves nothing on its own;
#     if the API stops reporting a digest, provisioning fails at install time.
if (-not $SkipDigest) {
  foreach ($target in $targets.Values | Where-Object { $_.Resolvable -or ($_.Own -and $_.Sha256) }) {
    $reference = Get-ReleaseAssetRef -Url $target.Url
    if (-not $reference) {
      [void]$failures.Add("not a GitHub release asset URL: $($target.Url)")
      continue
    }
    # Parse the JSON here rather than with `--jq`: a jq filter needs embedded
    # quotes, and Windows PowerShell 5.1 mangles those on the way to a native exe.
    $payload = & gh release view $reference.Tag --repo "$($reference.Owner)/$($reference.Repo)" --json assets 2>$null | Out-String
    $asset = $null
    if ($payload.Trim()) {
      $asset = (ConvertFrom-Json $payload).assets |
        Where-Object { $_.name -eq $reference.Name } |
        Select-Object -First 1
    }
    if (-not $asset -or -not $asset.digest) {
      [void]$failures.Add(
        "no asset digest for $($reference.Name) under tag $($reference.Tag)"
      )
      if ($target.Own) { $ownFailed = $true }
      continue
    }
    $digest = ($asset.digest -replace '^sha256:', '').Trim()
    if ($target.Resolvable) {
      # Nothing to compare against by design -- being answerable is the check.
      Write-Host "resolves ok  $($reference.Name) ($($reference.Tag)) -> $($digest.Substring(0, 12))..."
      continue
    }
    if ($digest -ne $target.Sha256) {
      [void]$failures.Add(
        "digest mismatch for $($reference.Name)`n    release: $digest`n    locked:  $($target.Sha256)"
      )
    }
    else {
      Write-Host "digest ok    $($reference.Name) ($($reference.Tag))"
    }
  }
}

if ($failures.Count -gt 0) {
  Write-Host ''
  Write-Host "$($failures.Count) pinned URL problem(s):" -ForegroundColor Red
  foreach ($failure in $failures) { Write-Host "  $failure" -ForegroundColor Red }
  Write-Host ''
  Write-Host 'Every one of these breaks a fresh install at the step that fetches it.'
  if ($ownFailed) {
    Write-Host 'Own-release asset missing: it was never uploaded. Publish it before tagging'
    Write-Host '(docs/ct2-distribution.md for the CT2 wheel). Already-shipped versions'
    Write-Host 'recover on their own -- these URLs resolve at install time.'
  }
  else {
    Write-Host 'Third-party pin: upstream deleted or retagged it. Pick a new version and'
    Write-Host 'update url + size + sha256 together (the manifest verifies the digest).'
  }
  exit 1
}

Write-Host ''
Write-Host "all $($targets.Count) pinned URLs resolve" -ForegroundColor Green
