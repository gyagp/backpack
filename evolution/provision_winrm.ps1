param(
    [Parameter(Mandatory=$true)][string]$ComputerName,
    [Parameter(Mandatory=$true)][string]$ServerUrl
)
$ErrorActionPreference = "Stop"
# Match webgfx-agents Provisioner.invokeRemote: Negotiate plus an encoded nested
# PowerShell payload. The server must have WinRM and TrustedHosts configured.
$remoteScript = @"
`$ErrorActionPreference = 'Stop'
`$install = 'D:\workspace\project\agents\backpack-evolution'
New-Item -ItemType Directory -Force -Path `$install | Out-Null
`$bootstrap = Join-Path `$install 'bootstrap-agent.ps1'
Invoke-WebRequest '$ServerUrl/api/bootstrap.ps1' -UseBasicParsing -OutFile `$bootstrap
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File `$bootstrap -Server '$ServerUrl'
if (`$LASTEXITCODE -ne 0) { throw "Worker bootstrap exited with code `$LASTEXITCODE" }
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteScript))
Invoke-Command -ComputerName $ComputerName -Authentication Negotiate -ScriptBlock {
    powershell -NoProfile -EncodedCommand $using:encoded
}
