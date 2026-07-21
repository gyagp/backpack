param(
    [string]$Server = "http://10.172.21.28:8787",
    [string]$InstallDir = "D:\workspace\project\agents\backpack-evolution",
    [string]$Repo = "D:\workspace\project\backpack",
    [string]$ModelsDir = "D:\workspace\project\agents\ai-models"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Invoke-WebRequest "$Server/api/agent.py" -UseBasicParsing -OutFile "$InstallDir\agent.py"
if (-not (Test-Path "$Repo\.git")) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Repo) | Out-Null
    git clone https://github.com/gyagp/backpack $Repo
}
$python = (Get-Command python -ErrorAction Stop).Source
$name = $env:COMPUTERNAME.ToLowerInvariant()
& $python "$InstallDir\agent.py" --server $Server --name $name --label role=required register
& $python "$InstallDir\agent.py" --server $Server --name $name sync-models --models-dir $ModelsDir
$watchScript = @"
`$ErrorActionPreference = 'Continue'
& '$python' '$InstallDir\agent.py' --server '$Server' --name '$name' --label 'role=required' watch --repo '$Repo' --worktrees '$Repo\gitignore\evolution\worktrees' *>> '$InstallDir\watch.log'
"@
[IO.File]::WriteAllText("$InstallDir\watch.ps1", $watchScript)
if (Get-ScheduledTask -TaskName backpack-evolution-agent -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName backpack-evolution-agent -ErrorAction SilentlyContinue
}
$taskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$InstallDir\watch.ps1`""
schtasks.exe /Create /TN backpack-evolution-agent /TR $taskCommand /SC ONLOGON /F | Out-Null
schtasks.exe /Run /TN backpack-evolution-agent | Out-Null
Write-Host "Backpack evolution agent installed for $name and connected to $Server"
