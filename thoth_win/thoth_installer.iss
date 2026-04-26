; ============================================================================
; Thoth Windows — Inno Setup Installer Script
;
; Creates a professional installer (.exe) with:
;   - Welcome wizard with Thoth logo
;   - License agreement
;   - Install location picker
;   - Start Menu + Desktop shortcuts
;   - Run at login (optional)
;   - Uninstaller
;
; Requires: Inno Setup 6 (https://jrsoftware.org/isinfo.php)
; Build:    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" thoth_installer.iss
; ============================================================================

#define MyAppName "Thoth"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Thothcraft"
#define MyAppURL "https://thothcraft.com"
#define MyAppExeName "Thoth.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=..\LICENSE
OutputDir=dist
OutputBaseFilename=Thoth-Setup-{#MyAppVersion}
SetupIconFile=icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startup"; Description: "Start Thoth when Windows starts"; GroupDescription: "Startup:"

[Files]
Source: "dist\Thoth\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "Thoth"; ValueData: """{app}\{#MyAppExeName}"""; Flags: uninsdeletevalue; Tasks: startup

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
