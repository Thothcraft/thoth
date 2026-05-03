# ============================================================================
# Thoth Windows — Uninstaller
# ============================================================================

$TaskName = "ThothStartup"

Write-Host "Uninstalling Thoth Windows ..."

# Signal the Brain server that this device is going offline before killing it
$ThothRoot     = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ConfigFile    = Join-Path $ThothRoot "data\config\device_config.json"
if (Test-Path $ConfigFile) {
    try {
        $cfg       = Get-Content $ConfigFile -Raw | ConvertFrom-Json
        $DeviceId  = $cfg.device_id
        $ServerUrl = $cfg.brain_server_url
        if ($DeviceId -and $ServerUrl) {
            $OfflineUrl = "$ServerUrl/device/$DeviceId/offline"
            Invoke-RestMethod -Uri $OfflineUrl -Method Post -ContentType "application/json" `
                              -Body "{}" -TimeoutSec 3 -ErrorAction SilentlyContinue | Out-Null
            Write-Host "  Sent offline signal to Brain server."
        }
    } catch {
        Write-Host "  (Could not send offline signal — continuing uninstall)"
    }
}

# Stop the process
Get-Process -Name "python*" -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*thoth_win*"
} | Stop-Process -Force -ErrorAction SilentlyContinue

# Remove scheduled task
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# Remove registry Run key fallback (if install used that method instead)
$RegPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
Remove-ItemProperty -Path $RegPath -Name $TaskName -ErrorAction SilentlyContinue

Write-Host "  Thoth has been uninstalled."
Write-Host ""

$yn = Read-Host "Remove virtual environment? (y/N)"
if ($yn -eq "y" -or $yn -eq "Y") {
    $VenvDir = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) ".venv"
    Remove-Item -Recurse -Force $VenvDir -ErrorAction SilentlyContinue
    Write-Host "  Virtual environment removed."
}

Read-Host "Press Enter to close"
