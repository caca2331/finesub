#ifndef AppSource
  #error AppSource must point to the packaged FineSub Desktop application directory.
#endif

#ifndef AppVersion
  #define AppVersion "0.2.8"
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
