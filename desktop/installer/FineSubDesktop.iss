#ifndef AppSource
  #error AppSource must point to the packaged FineSub Desktop application directory.
#endif

#ifndef AppVersion
  #define AppVersion "0.3.2"
#endif

#ifndef OutputDir
  #define OutputDir "."
#endif

#ifndef SetupIcon
  #error SetupIcon must point to the FineSub Desktop .ico file.
#endif

#define AppPublisher "FineSub"
#define AppExeName "FineSub Desktop.exe"

[Setup]
AppId={{D4C7C84D-3037-4CF5-B9CA-9EA30265414F}
AppName=FineSub Desktop
AppVersion={#AppVersion}
AppVerName=FineSub Desktop {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\FineSub Desktop
DefaultGroupName=FineSub Desktop
DisableDirPage=no
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=FineSub-Desktop-{#AppVersion}-Setup
SetupIconFile={#SetupIcon}
UninstallDisplayIcon={app}\FineSub Desktop.exe
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
SetupLogging=yes
UsePreviousAppDir=yes
UsePreviousTasks=yes
CloseApplications=yes
RestartApplications=no

[Languages]
#ifdef IncludeChineseLanguage
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
#endif
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce

[Files]
Source: "{#AppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\FineSub Desktop"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#AppExeName}"
Name: "{autodesktop}\FineSub Desktop"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,FineSub Desktop}"; Flags: nowait postinstall skipifsilent

[Code]
{ The marker separates an installed copy (personal data in
  %LOCALAPPDATA%\FineSub) from a portable one (everything beside the exe).
  Only this installer writes it; update payloads never contain it and the
  in-app updater preserves it. }
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    SaveStringToFile(ExpandConstant('{app}\installed.marker'), '', False);
end;

{ Inno only removes files it installed, so the runtime state FineSub creates
  beside the exe (managed Python, models, caches - all rebuildable) would
  survive an uninstall. Personal data lives outside the install dir and is
  only removed when the user agrees. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  PersonalData: String;
begin
  if CurUninstallStep <> usPostUninstall then
    exit;
  DelTree(ExpandConstant('{app}\runtime'), True, True, True);
  DelTree(ExpandConstant('{app}\models'), True, True, True);
  DelTree(ExpandConstant('{app}\cache'), True, True, True);
  DelTree(ExpandConstant('{app}\app'), True, True, True);
  DelTree(ExpandConstant('{app}\.update'), True, True, True);
  DeleteFile(ExpandConstant('{app}\installed.marker'));
  RemoveDir(ExpandConstant('{app}'));
  PersonalData := ExpandConstant('{localappdata}\FineSub');
  if DirExists(PersonalData) then
  begin
    if MsgBox(
      'Also delete the FineSub data folder (settings, API keys, task '
        + 'history, and any FineSub CLI downloads)?'
        + #13#10 + PersonalData,
      mbConfirmation, MB_YESNO
    ) = IDYES then
      DelTree(PersonalData, True, True, True);
  end;
end;
