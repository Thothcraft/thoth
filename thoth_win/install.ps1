# ============================================================================
# Thoth Windows — One-Click Installer (PowerShell)
#
# Usage:  Right-click → Run with PowerShell   (or:  powershell -ExecutionPolicy Bypass -File install.ps1)
#
# What it does:
#   1. Creates a Python venv in thoth_win\.venv
#   2. Installs core + Windows dependencies
#   3. Creates a Scheduled Task so Thoth starts on every login
# ============================================================================

$ErrorActionPreference = "Stop"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ThothRoot  = Split-Path -Parent $ScriptDir
$VenvDir    = Join-Path $ScriptDir ".venv"
$PythonBin  = Join-Path $VenvDir "Scripts\python.exe"
$AppScript  = Join-Path $ScriptDir "app.py"
$TaskName   = "ThothStartup"

Write-Host "============================================"
Write-Host "        Thoth Windows Installer"
Write-Host "============================================"

# --- 1. Python venv ---
Write-Host "`n> Creating virtual environment ..."
python -m venv $VenvDir
& "$VenvDir\Scripts\Activate.ps1"

Write-Host "> Upgrading pip ..."
& $PythonBin -m pip install --upgrade pip -q

Write-Host "> Installing core dependencies ..."
& $PythonBin -m pip install -r "$ThothRoot\thoth_core\requirements.txt" -q

Write-Host "> Installing Windows dependencies ..."
& $PythonBin -m pip install -r "$ScriptDir\requirements.txt" -q

# --- 2. Create Scheduled Task (run at logon) ---
Write-Host "> Registering startup task ..."

# Remove existing task if present
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

$Action  = New-ScheduledTaskAction -Execute $PythonBin -Argument "`"$AppScript`"" -WorkingDirectory $ScriptDir
$Trigger = New-ScheduledTaskTrigger -AtLogon
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Thoth Smart Home Sensor Platform" | Out-Null

# --- 3. Create logs dir ---
New-Item -ItemType Directory -Force -Path "$ThothRoot\logs" | Out-Null

# --- 4. Start Thoth now ---
Write-Host "> Starting Thoth ..."
Start-Process -FilePath $PythonBin -ArgumentList "`"$AppScript`"" -WindowStyle Hidden

Write-Host "`n  Thoth installed and running!"
Write-Host "  Dashboard:  http://localhost:8000"
Write-Host "  System tray: look for the Thoth icon"
Write-Host ""
Write-Host "To uninstall, run:  .\uninstall.ps1"
Read-Host "Press Enter to close"
