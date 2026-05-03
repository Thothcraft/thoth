# ============================================================================
# Thoth Windows — Uninstaller
# ============================================================================

$TaskName = "ThothStartup"

Write-Host "Uninstalling Thoth Windows ..."

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
