# ====================================================================
# Thoth Windows - Add to Programs & Features Registry
# ====================================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ThothRoot = Split-Path -Parent $ScriptDir
$Version = "1.0.0"

# Create registry entries for Programs & Features
$RegPath = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Thoth"

# Check if running as admin
$IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

if ($IsAdmin) {
    # Add to system-wide registry (requires admin)
    Write-Host "Adding to Programs & Features (system-wide)..."
    
    # Remove old entry if exists
    Remove-Item -Path $RegPath -Recurse -ErrorAction SilentlyContinue
    
    # Create new entry
    New-Item -Path $RegPath -Force | Out-Null
    Set-ItemProperty -Path $RegPath -Name "DisplayName" -Value "Thoth - Smart Home Sensor Platform"
    Set-ItemProperty -Path $RegPath -Name "DisplayVersion" -Value $Version
    Set-ItemProperty -Path $RegPath -Name "Publisher" -Value "Thothcraft"
    Set-ItemProperty -Path $RegPath -Name "DisplayIcon" -Value "$ScriptDir\icon.ico"
    Set-ItemProperty -Path $RegPath -Name "InstallLocation" -Value $ThothRoot
    Set-ItemProperty -Path $RegPath -Name "NoModify" -Value 1
    Set-ItemProperty -Path $RegPath -Name "NoRepair" -Value 1
    Set-ItemProperty -Path $RegPath -Name "UninstallString" -Value "powershell -ExecutionPolicy Bypass -File `"$ScriptDir\uninstall.ps1`""
    Set-ItemProperty -Path $RegPath -Name "QuietUninstallString" -Value "powershell -ExecutionPolicy Bypass -File `"$ScriptDir\uninstall.ps1`" -Force"
    
    Write-Host "Added to Programs & Features successfully!"
} else {
    # Add to user-specific registry
    Write-Host "Adding to Programs & Features (current user)..."
    
    $RegPath = "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Thoth"
    
    # Remove old entry if exists
    Remove-Item -Path $RegPath -Recurse -ErrorAction SilentlyContinue
    
    # Create new entry
    New-Item -Path $RegPath -Force | Out-Null
    Set-ItemProperty -Path $RegPath -Name "DisplayName" -Value "Thoth - Smart Home Sensor Platform"
    Set-ItemProperty -Path $RegPath -Name "DisplayVersion" -Value $Version
    Set-ItemProperty -Path $RegPath -Name "Publisher" -Value "Thothcraft"
    Set-ItemProperty -Path $RegPath -Name "DisplayIcon" -Value "$ScriptDir\icon.ico"
    Set-ItemProperty -Path $RegPath -Name "InstallLocation" -Value $ThothRoot
    Set-ItemProperty -Path $RegPath -Name "NoModify" -Value 1
    Set-ItemProperty -Path $RegPath -Name "NoRepair" -Value 1
    Set-ItemProperty -Path $RegPath -Name "UninstallString" -Value "powershell -ExecutionPolicy Bypass -File `"$ScriptDir\uninstall.ps1`""
    Set-ItemProperty -Path $RegPath -Name "QuietUninstallString" -Value "powershell -ExecutionPolicy Bypass -File `"$ScriptDir\uninstall.ps1`" -Force"
    
    Write-Host "Added to Programs & Features (current user) successfully!"
}

Write-Host ""
Write-Host "To uninstall Thoth, you can now:"
Write-Host "  1. Right-click the Thoth system tray icon and select 'Uninstall Thoth'"
Write-Host "  2. Go to Settings > Apps > Apps & features and uninstall 'Thoth'"
Write-Host "  3. Run: .\uninstall.ps1"
Write-Host ""
