@echo off
REM ====================================================================
REM Thoth Windows Uninstaller (Batch file for Programs & Features)
REM ====================================================================

setlocal enabledelayedexpansion

echo Uninstalling Thoth...
echo.

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
set THOTH_ROOT=%SCRIPT_DIR%..

REM Stop Thoth processes
echo Stopping Thoth processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Thoth*" 2>nul
taskkill /F /IM Thoth.exe 2>nul

REM Remove scheduled task
echo Removing scheduled task...
schtasks /Delete /TN "ThothStartup" /F 2>nul

REM Remove from registry
echo Removing from startup registry...
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v "ThothStartup" /f 2>nul

REM Remove desktop shortcut
echo Removing desktop shortcut...
del "%USERPROFILE%\Desktop\Thoth.lnk" 2>nul

REM Remove Start Menu shortcuts
echo removing Start Menu shortcuts...
rmdir /S /Q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Thoth" 2>nul

echo.
echo Thoth has been uninstalled.
echo.
set /p choice="Remove virtual environment? (y/N) "
if /i "!choice!"=="y" (
    echo Removing virtual environment...
    rmdir /S /Q "%SCRIPT_DIR%.venv" 2>nul
    echo Virtual environment removed.
)

echo.
echo Uninstallation complete!
pause
