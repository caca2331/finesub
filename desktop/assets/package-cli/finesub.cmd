@echo off
rem `finesub` for this installation. The app's executables are windowed and
rem cannot write to a console, so the command line runs on the managed
rem interpreter the app already installed.
rem ASCII only, CRLF only: cmd.exe reads this file in the console code page,
rem and LF-only line endings make it garble every second line.
setlocal
set "FINESUB_PACKAGE=%~dp0"
set "FINESUB_PYTHON=%FINESUB_PACKAGE%runtime\python\Scripts\python.exe"
if not exist "%FINESUB_PYTHON%" (
    echo FineSub is not set up yet: the managed Python runtime is missing.>&2
    echo Open FineSub Desktop and let it finish installing resources,>&2
    echo or use the pip-installed "finesub" command instead.>&2
    exit /b 1
)
"%FINESUB_PYTHON%" -X utf8 "%FINESUB_PACKAGE%finesub.py" %*
exit /b %ERRORLEVEL%
