; Thoth-Setup — web installer for the Thoth node on Windows.
; Built by CI: makensis /DVERSION=x.y.z thoth.nsi
!ifndef VERSION
!define VERSION "0.0.0"
!endif

Name "Thoth ${VERSION}"
OutFile "Thoth-Setup-${VERSION}.exe"
Unicode True
InstallDir "$TEMP\thoth-setup"
RequestExecutionLevel admin
SetCompressor /SOLID lzma

!include "MUI2.nsh"
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Section "Install"
    DetailPrint "Fetching https://thothcraft.com/install.ps1 ..."
    nsExec::ExecToLog 'powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$$f = Join-Path $$env:TEMP thoth-install.ps1; irm https://thothcraft.com/install.ps1 -OutFile $$f; & $$f; exit $$LASTEXITCODE"'
    Pop $0
    DetailPrint "Installer exit code: $0"
    IntCmp $0 0 done
        SetErrorLevel 1
        MessageBox MB_ICONEXCLAMATION|MB_OK "Thoth installer reported an error ($0). You can retry manually: download https://thothcraft.com/install.ps1 and run it with powershell -File"
    done:
SectionEnd
