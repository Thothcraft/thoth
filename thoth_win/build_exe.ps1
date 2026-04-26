# ============================================================================
# Thoth Windows — Build .exe Installer
#
# This script:
#   1. Converts icon.png → icon.ico
#   2. Builds Thoth.exe via PyInstaller
#   3. Creates an Inno Setup installer (.exe with wizard)
#
# Prerequisites:
#   pip install pyinstaller Pillow
#   Optional: Inno Setup 6 (https://jrsoftware.org/isinfo.php) for the installer wizard
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File build_exe.ps1
# ============================================================================

$ErrorActionPreference = "Stop"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ThothRoot  = Split-Path -Parent $ScriptDir
$VenvDir    = Join-Path $ScriptDir ".venv"
$PythonBin  = Join-Path $VenvDir "Scripts\python.exe"
$DistDir    = Join-Path $ScriptDir "dist"
$Version    = "1.0.0"

Write-Host "============================================"
Write-Host "    Thoth Windows — Build EXE Installer"
Write-Host "============================================"

# --- 0. Ensure venv + deps ---
if (-not (Test-Path $VenvDir)) {
    Write-Host "> Creating build venv ..."
    python -m venv $VenvDir
}
& "$VenvDir\Scripts\Activate.ps1"
& $PythonBin -m pip install --upgrade pip -q
& $PythonBin -m pip install -r "$ThothRoot\thoth_core\requirements.txt" -q
& $PythonBin -m pip install -r "$ScriptDir\requirements.txt" -q

# --- 1. Convert PNG → ICO ---
Write-Host "> Converting icon.png to icon.ico ..."
& $PythonBin -c @"
from PIL import Image
img = Image.open('$ScriptDir/icon.png')
img.save('$ScriptDir/icon.ico', format='ICO', sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
print('icon.ico created')
"@

# --- 2. Build with PyInstaller ---
Write-Host "> Building Thoth.exe with PyInstaller ..."
& $PythonBin -m PyInstaller `
    --name "Thoth" `
    --icon "$ScriptDir\icon.ico" `
    --noconsole `
    --add-data "$ThothRoot\thoth_core\backend\templates;thoth_core\backend\templates" `
    --add-data "$ThothRoot\thoth_core\backend\static;thoth_core\backend\static" `
    --add-data "$ThothRoot\.env;." `
    --add-data "$ScriptDir\icon.png;." `
    --paths "$ThothRoot" `
    --hidden-import "thoth_core.backend.app" `
    --hidden-import "thoth_core.backend.routes.files" `
    --hidden-import "thoth_core.sensors" `
    --hidden-import "thoth_win.sensors" `
    --hidden-import "engineio.async_drivers.threading" `
    --distpath "$DistDir" `
    --workpath "$ScriptDir\build" `
    --specpath "$ScriptDir" `
    "$ScriptDir\app.py"

if (-not (Test-Path "$DistDir\Thoth\Thoth.exe")) {
    Write-Host "ERROR: Build failed"
    exit 1
}

Write-Host ""
Write-Host "  Thoth.exe built at: $DistDir\Thoth\Thoth.exe"
Write-Host ""

# --- 3. Optionally build Inno Setup installer ---
$InnoCompiler = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
$IssFile      = Join-Path $ScriptDir "thoth_installer.iss"

if (Test-Path $InnoCompiler) {
    if (Test-Path $IssFile) {
        Write-Host "> Building Inno Setup installer ..."
        & $InnoCompiler $IssFile
        Write-Host "  Installer created at: $DistDir\Thoth-Setup-$Version.exe"
    } else {
        Write-Host "  Skipping Inno Setup (no .iss file found)"
    }
} else {
    Write-Host "  Skipping Inno Setup (not installed — download from jrsoftware.org)"
}

Write-Host ""
Write-Host "============================================"
Write-Host "  BUILD COMPLETE"
Write-Host "============================================"
Read-Host "Press Enter to close"
