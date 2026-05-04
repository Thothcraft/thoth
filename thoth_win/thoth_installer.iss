; ============================================================================
; Thoth Windows - Inno Setup Installer Script
;
; Creates a professional GUI installer (.exe) with:
;   - Welcome wizard with Thoth branding
;   - License agreement page
;   - Install location picker
;   - Desktop icon (checked by default)
;   - Start Menu shortcuts with Uninstall entry
;   - Optional run-at-login
;   - Proper Programs & Features uninstall registration
;   - Stops running Thoth process before uninstalling
;
; Requires: Inno Setup 6 (https://jrsoftware.org/isinfo.php)
; Build:    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" thoth_installer.iss
;
; In CI the version is injected via:
;   ISCC.exe /DMyAppVersion=1.2.3 thoth_installer.iss
; ============================================================================

#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif

#define MyAppName      "Thoth"
#define MyAppPublisher "Thothcraft"
#define MyAppURL       "https://thothcraft.com"
#define MyAppExeName   "Thoth.exe"
#define MyAppID        "A1B2C3D4-E5F6-7890-ABCD-EF1234567890"

; ---- [Setup] ----------------------------------------------------------------
[Setup]
AppId={{{#MyAppID}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/support
AppUpdatesURL={#MyAppURL}/releases

; Install into per-user Program Files so no admin is required
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes

; Appearance
WizardStyle=modern
WizardSizePercent=120
SetupIconFile=icon.ico
; WizardImageFile and WizardSmallImageFile can point to custom bitmaps if added
; WizardImageFile=wizard_banner.bmp
; WizardSmallImageFile=wizard_small.bmp

; Output
OutputDir=dist
OutputBaseFilename=Thoth-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
InternalCompressLevel=ultra

; Privileges - runs without admin; uses HKCU for registry
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Uninstall
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName} {#MyAppVersion}

; Misc
AllowNoIcons=yes
ShowLanguageDialog=no
LicenseFile=..\LICENSE
ChangesAssociations=no
ArchitecturesInstallIn64BitMode=x64compatible

; ---- [Languages] ------------------------------------------------------------
[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

; ---- [Tasks] ----------------------------------------------------------------
[Tasks]
; Desktop icon is checked by default
Name: "desktopicon"; \
  Description: "Create a &desktop shortcut"; \
  GroupDescription: "Additional shortcuts:"; \
  Flags: checkedonce

; Run at Windows login
Name: "startup"; \
  Description: "Start {#MyAppName} automatically when Windows starts"; \
  GroupDescription: "Startup:"

; ---- [Files] ----------------------------------------------------------------
[Files]
; Main application bundle produced by PyInstaller
Source: "dist\Thoth\*"; \
  DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs

; ---- [Icons] ----------------------------------------------------------------
[Icons]
; Start Menu
Name: "{group}\{#MyAppName}";                        Filename: "{app}\{#MyAppExeName}"; Comment: "Open the Thoth sensor dashboard"
Name: "{group}\{#MyAppName} - Open Dashboard";       Filename: "{app}\{#MyAppExeName}"; Parameters: "--open-dashboard"; Comment: "Open dashboard in browser"
Name: "{group}\Uninstall {#MyAppName}";              Filename: "{uninstallexe}"

; Desktop shortcut (only when task is selected)
Name: "{autodesktop}\{#MyAppName}";                  Filename: "{app}\{#MyAppExeName}"; Comment: "Thoth - Smart Home Sensor Platform"; Tasks: desktopicon

; ---- [Registry] -------------------------------------------------------------
[Registry]
; Run-at-login entry (added only when startup task is selected; removed on uninstall)
Root: HKCU; \
  Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
  ValueType: string; \
  ValueName: "{#MyAppName}"; \
  ValueData: """{app}\{#MyAppExeName}"""; \
  Flags: uninsdeletevalue; \
  Tasks: startup

; ---- [UninstallRun] ---------------------------------------------------------
[UninstallRun]
; Gracefully stop the running Thoth process before files are removed
Filename: "taskkill"; Parameters: "/F /IM {#MyAppExeName}"; \
  Flags: runhidden skipifdoesntexist; RunOnceId: "KillThoth"

; ---- [Run] ------------------------------------------------------------------
[Run]
; Offer to launch Thoth after setup completes
Filename: "{app}\{#MyAppExeName}"; \
  Description: "Launch {#MyAppName} now"; \
  Flags: nowait postinstall skipifsilent

; ---- [Code] -----------------------------------------------------------------
[Code]
// Kill any running Thoth instance before upgrading/reinstalling
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssInstall then begin
    Exec('taskkill', '/F /IM {#MyAppExeName}', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;
