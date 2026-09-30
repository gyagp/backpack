param(
    [string]$OrtRepository = 'D:\workspace\project\onnxruntime',
    [string]$GenaiRepository = 'D:\workspace\project\onnxruntime-genai',
    [string]$BuildScript = (Join-Path $PSScriptRoot 'build_native_reference.py'),
    [string]$BackupRoot = 'D:\workspace\project\backpack\gitignore\evolution\backups\ort'
)

$ErrorActionPreference = 'Stop'
if ([Environment]::MachineName -ine 'webgfx-104') {
    throw 'This goal only permits execution on webgfx-104'
}
$workspace = Split-Path -Parent $PSScriptRoot
$temporaryRoot = Join-Path $workspace 'gitignore\o'
$resolvedTemporaryRoot = [IO.Path]::GetFullPath($temporaryRoot)
$requiredPrefix = [IO.Path]::GetFullPath((Join-Path $workspace 'gitignore')) + [IO.Path]::DirectorySeparatorChar
if (-not $resolvedTemporaryRoot.StartsWith($requiredPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "ORT build root must remain below the Backpack gitignore directory: $resolvedTemporaryRoot"
}
$BackupRoot = [IO.Path]::GetFullPath($BackupRoot)
if (-not $BackupRoot.StartsWith($requiredPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Native reference backups must remain below the workspace gitignore directory: $BackupRoot"
}
$temporaryFiles = Join-Path $workspace 'gitignore\tmp'
New-Item -ItemType Directory -Force $temporaryFiles | Out-Null
$env:TMP = $temporaryFiles
$env:TEMP = $temporaryFiles

foreach ($repository in @($OrtRepository, $GenaiRepository)) {
    if (-not (Test-Path (Join-Path $repository '.git'))) {
        # A linked worktree uses a .git file; the primary source checkout used
        # for fetching is expected to be a normal repository directory.
        if (-not (Test-Path $repository)) { throw "Repository not found: $repository" }
    }
    & git -C $repository fetch origin main
    if ($LASTEXITCODE -ne 0) { throw "Failed to fetch origin/main for $repository" }
}

$ortRevision = (& git -C $OrtRepository rev-parse origin/main).Trim()
$genaiRevision = (& git -C $GenaiRepository rev-parse origin/main).Trim()
$revisionKey = "ort-$($ortRevision.Substring(0,10))-genai-$($genaiRevision.Substring(0,10))"
$pathKey = "$($ortRevision.Substring(0,7))-$($genaiRevision.Substring(0,7))"
$buildRoot = Join-Path $temporaryRoot $pathKey
$ortWorktree = Join-Path $buildRoot 'onnxruntime'
$genaiWorktree = Join-Path $buildRoot 'onnxruntime-genai'
$buildCwd = Join-Path $buildRoot 'agents'
New-Item -ItemType Directory -Force $buildRoot,$buildCwd | Out-Null

if (-not (Test-Path $ortWorktree)) {
    & git -C $OrtRepository worktree add --detach $ortWorktree $ortRevision
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the isolated ONNX Runtime worktree' }
}
if (-not (Test-Path $genaiWorktree)) {
    & git -C $GenaiRepository worktree add --detach $genaiWorktree $genaiRevision
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the isolated ONNX Runtime GenAI worktree' }
}

# Windows WebGPU policy forbids subgroup-matrix operations. Upstream native
# ORT requests every optional adapter feature it sees, so remove only that
# request in the isolated build worktree. The source checkout remains intact,
# and a changed upstream spelling fails closed instead of silently enabling it.
$webgpuContext = Join-Path $ortWorktree 'onnxruntime\core\providers\webgpu\webgpu_context.cc'
$webgpuSource = Get-Content -LiteralPath $webgpuContext -Raw
$subgroupRequest = '      wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix,'
$subgroupMarker = '      // Backpack Windows policy: subgroup-matrix feature intentionally not requested.'
if ($webgpuSource.Contains($subgroupRequest)) {
    $webgpuSource = $webgpuSource.Replace($subgroupRequest, $subgroupMarker)
    $webgpuSource | Set-Content -LiteralPath $webgpuContext -Encoding utf8 -NoNewline
} elseif (-not $webgpuSource.Contains($subgroupMarker)) {
    throw 'Could not enforce the Windows subgroup-matrix exclusion in upstream ORT'
}

Push-Location $buildCwd
try {
    if (-not (Test-Path -LiteralPath $BuildScript -PathType Leaf)) {
        throw "Native build driver not found: $BuildScript"
    }
    & python -B $BuildScript --workspace $workspace --ort-source $ortWorktree --genai-source $genaiWorktree `
        --log-root (Join-Path $buildRoot 'logs')
    if ($LASTEXITCODE -ne 0) { throw "Native reference build failed with exit code $LASTEXITCODE" }
    $nativeBuild = Get-Content -LiteralPath (Join-Path $buildRoot 'logs/build-layout.json') -Raw | ConvertFrom-Json

    # Build the native conformance example against the same library pair. Deterministic
    # conformance needs model_chat.exe from the identical GenAI revision.
    # GenAI's build.py example wrapper assumes a downloaded ./ort staging
    # directory even when --ort_home is supplied, so configure model_chat
    # directly against the exact ORT install and GenAI output instead.
    $ortHome = $nativeBuild.ort_home
    $exampleSource = Join-Path $genaiWorktree 'examples\c'
    $exampleBuild = Join-Path $exampleSource 'build-backpack'
    $genaiLibrary = $nativeBuild.genai_library
    & cmake -S $exampleSource -B $exampleBuild -G 'Visual Studio 17 2022' -A x64 `
        -DMODEL_CHAT=ON "-DORT_INCLUDE_DIR=$(Join-Path $ortHome 'include')" `
        "-DORT_LIB_DIR=$(Join-Path $ortHome 'lib')" "-DOGA_INCLUDE_DIR=$(Join-Path $genaiWorktree 'src')" `
        "-DOGA_LIB_DIR=$genaiLibrary"
    if ($LASTEXITCODE -ne 0) { throw "GenAI model_chat configure failed with exit code $LASTEXITCODE" }
    & cmake --build $exampleBuild --config Release --target model_chat --parallel
    if ($LASTEXITCODE -ne 0) { throw "GenAI model_chat build failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

function Find-One([string]$root, [string]$name) {
    $item = Get-ChildItem -LiteralPath $root -Recurse -File -Filter $name |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $item) { throw "Built artifact $name was not found below $root" }
    return $item.FullName
}

$date = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
$destination = [IO.Path]::GetFullPath((Join-Path $BackupRoot "$date-$revisionKey"))
$staging = [IO.Path]::GetFullPath((Join-Path $buildRoot ('backup-staging-' + [Guid]::NewGuid().ToString('N'))))
New-Item -ItemType Directory -Force $staging | Out-Null
$artifacts = [ordered]@{
    'onnxruntime.dll' = Find-One $nativeBuild.ort_home 'onnxruntime.dll'
    'dxcompiler.dll' = Find-One $nativeBuild.ort_build 'dxcompiler.dll'
    'onnxruntime-genai.dll' = Find-One $nativeBuild.genai_library 'onnxruntime-genai.dll'
    'model_benchmark.exe' = Find-One $nativeBuild.genai_build 'model_benchmark.exe'
    'model_chat.exe' = Find-One $exampleBuild 'model_chat.exe'
    'genai_state_reference.exe' = $nativeBuild.reference_helper
}
# These DLLs are optional for a monolithic WebGPU build or integrated DXIL validator.
foreach ($optional in @('onnxruntime_providers_shared.dll', 'dxil.dll')) {
    $found = Get-ChildItem -LiteralPath $nativeBuild.ort_build -Recurse -File -Filter $optional |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($found) { $artifacts[$optional] = $found.FullName }
}
foreach ($entry in $artifacts.GetEnumerator()) {
    Copy-Item -LiteralPath $entry.Value -Destination (Join-Path $staging $entry.Key) -Force
}

$hashes = [ordered]@{}
foreach ($name in $artifacts.Keys) {
    $hashes[$name] = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $staging $name)).Hash.ToLowerInvariant()
}
$manifest = [ordered]@{
    date = (Get-Date).ToUniversalTime().ToString('o')
    architecture = 'x64'
    configuration = 'Release'
    backend = 'webgpu'
    onnxruntime_revision = $ortRevision
    onnxruntime_genai_revision = $genaiRevision
    artifacts = @($artifacts.Keys)
    artifact_hashes = $hashes
    source = 'isolated worktrees built once on webgfx-104'
    windows_subgroup_matrix = 'disabled: optional ChromiumExperimentalSubgroupMatrix feature is not requested'
}
$manifestPath = Join-Path $staging 'build-manifest.json'
[IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 4), (New-Object Text.UTF8Encoding($false)))
# Keep partial builds outside discovery until the whole backup is ready.
if (-not $staging.StartsWith($requiredPrefix, [StringComparison]::OrdinalIgnoreCase) -or
    -not $destination.StartsWith(($BackupRoot + [IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Backup publication paths escaped their verified workspace directories'
}
if (Test-Path -LiteralPath $destination) { throw "Backup destination already exists: $destination" }
New-Item -ItemType Directory -Force $BackupRoot | Out-Null
Move-Item -LiteralPath $staging -Destination $destination
$benchmarkAdapter = Join-Path $PSScriptRoot 'benchmark_ort.py'
Copy-Item -LiteralPath $benchmarkAdapter -Destination (Join-Path $BackupRoot 'benchmark_ort.py') -Force

Write-Output "ORT $ortRevision / GenAI $genaiRevision ready locally on webgfx-104"
