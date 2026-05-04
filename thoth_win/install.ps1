# ============================================================================
# Thoth Windows — One-Click Installer (PowerShell)
#
# Usage:  Right-click → Run with PowerShell   (or:  powershell -ExecutionPolicy Bypass -File install.ps1)
#
# What it does:
#   1. Creates a Python venv in thoth_win\.venv
#   2. Installs core + Windows dependencies
#   3. Registers Thoth to start on login (Scheduled Task if admin, registry Run key otherwise)
# ============================================================================

# --- Self-elevate to Administrator if not already running elevated ---
$IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $IsAdmin) {
    Write-Host "Requesting administrator privileges ..."
    $Args = "-ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`""
    Start-Process powershell -Verb RunAs -ArgumentList $Args
    exit
}

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

# --- 2. Register startup (Scheduled Task preferred; registry Run key as fallback) ---
Write-Host "> Registering startup task ..."

$StartupRegistered = $false

# Try Scheduled Task first (requires admin — we should have it, but guard anyway)
try {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

    $Action   = New-ScheduledTaskAction -Execute $PythonBin -Argument "`"$AppScript`"" -WorkingDirectory $ScriptDir
    $Trigger  = New-ScheduledTaskTrigger -AtLogon
    $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    $Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
        -Principal $Principal -Description "Thoth Smart Home Sensor Platform" | Out-Null

    Write-Host "  [OK] Scheduled Task registered (runs at login, elevated)."
    $StartupRegistered = $true
} catch {
    Write-Host "  [WARN] Could not register Scheduled Task: $($_.Exception.Message)"
}

# Fallback: HKCU Run registry key (no admin needed, current user only)
if (-not $StartupRegistered) {
    try {
        $RegPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
        Set-ItemProperty -Path $RegPath -Name $TaskName -Value "`"$PythonBin`" `"$AppScript`"" -Force
        Write-Host "  [OK] Startup registered via registry Run key (current user)."
        $StartupRegistered = $true
    } catch {
        Write-Host "  [WARN] Could not register startup entry: $($_.Exception.Message)"
        Write-Host "  Thoth will NOT start automatically at login."
        Write-Host "  You can start it manually: & '$PythonBin' '$AppScript'"
    }
}

# --- 3. Create logs dir ---
New-Item -ItemType Directory -Force -Path "$ThothRoot\logs" | Out-Null

# --- 4. Register in Programs & Features ---
Write-Host "> Registering in Programs & Features..."
& "$ScriptDir\install_registry.ps1"

# --- 5. Start Thoth now ---
Write-Host "> Starting Thoth ..."
Start-Process -FilePath $PythonBin -ArgumentList "`"$AppScript`"" -WindowStyle Hidden

Write-Host "`n  Thoth installed and running!"
Write-Host "  Dashboard:  http://localhost:8000"
Write-Host "  System tray: look for the Thoth icon"
Write-Host ""
Write-Host "To uninstall, you can:"
Write-Host "  1. Right-click the Thoth system tray icon and select 'Uninstall Thoth'"
Write-Host "  2. Go to Settings > Apps > Apps & features and uninstall 'Thoth'"
Write-Host "  3. Run: .\uninstall.ps1"
Read-Host "Press Enter to close"
