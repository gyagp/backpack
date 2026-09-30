param(
    [string]$BackupRoot = 'D:\workspace\project\backpack\gitignore\evolution\backups\llamacpp'
)
$ErrorActionPreference = 'Stop'
if ([Environment]::MachineName -ine 'webgfx-104') {
    throw 'This goal only permits execution on webgfx-104'
}
# GitHub's /latest may point to a source-only semantic-version release.
# Select the newest published release that actually supplies the required binary.
$releases = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30' -Headers @{'User-Agent'='Backpack'}
$release = $releases | Where-Object {
    -not $_.draft -and
    ($_.assets | Where-Object { $_.name -match 'win-vulkan-x64\.zip$' })
} | Sort-Object published_at -Descending | Select-Object -First 1
if (-not $release) { throw 'No published Windows Vulkan x64 release found' }
$asset = $release.assets | Where-Object { $_.name -match 'win-vulkan-x64\.zip$' } | Select-Object -First 1
$destination = Join-Path (Join-Path $BackupRoot $release.tag_name) 'vulkan'
if (-not (Test-Path (Join-Path $destination 'llama-bench.exe'))) {
    $temporary = 'D:\workspace\project\backpack\gitignore\evolution\llamacpp-download'
    $archive = Join-Path $temporary $asset.name
    $expanded = Join-Path $temporary $release.tag_name
    New-Item -ItemType Directory -Force $temporary,$expanded,$destination | Out-Null
    Invoke-WebRequest $asset.browser_download_url -OutFile $archive
    Expand-Archive $archive $expanded -Force
    $bench = Get-ChildItem $expanded -Recurse -Filter llama-bench.exe | Select-Object -First 1
    if (-not $bench) { throw 'Downloaded llama.cpp archive does not contain llama-bench.exe' }
    Copy-Item (Join-Path $bench.Directory.FullName '*') $destination -Recurse -Force
}
$manifest = [ordered]@{release=$release.tag_name; published_at=$release.published_at; asset=$asset.name; architecture='x64'; backend='vulkan'}
$manifestPath = Join-Path $destination 'build-manifest.json'
[IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))
$benchmarkAdapter = Join-Path $PSScriptRoot 'benchmark_llamacpp.py'
Copy-Item -LiteralPath $benchmarkAdapter -Destination (Join-Path $BackupRoot 'benchmark_llamacpp.py') -Force
Write-Output "llama.cpp $($release.tag_name) ready locally on webgfx-104"
