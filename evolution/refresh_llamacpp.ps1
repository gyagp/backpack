param(
    [string]$BackupRoot = 'D:\backup\x64\llamacpp'
)
$ErrorActionPreference = 'Stop'
$release = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest' -Headers @{'User-Agent'='Backpack'}
$asset = $release.assets | Where-Object { $_.name -match 'win-vulkan-x64\.zip$' } | Select-Object -First 1
if (-not $asset) { throw "No Windows Vulkan x64 asset found in $($release.tag_name)" }
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
$manifest | ConvertTo-Json | Set-Content (Join-Path $destination 'build-manifest.json') -Encoding utf8
$targets = @('10.172.21.15','10.95.136.157')
foreach ($target in $targets) {
    $session = New-PSSession -ComputerName $target -Authentication Negotiate
    try {
        $remote = "D:\backup\x64\llamacpp\$($release.tag_name)\vulkan"
        Invoke-Command -Session $session -ScriptBlock { param($path) New-Item -ItemType Directory -Force $path | Out-Null } -ArgumentList $remote
        Copy-Item (Join-Path $destination '*') $remote -ToSession $session -Recurse -Force
    } finally {
        Remove-PSSession $session
    }
}
Write-Output "llama.cpp $($release.tag_name) downloaded once and distributed to $($targets.Count) x64 devices"
