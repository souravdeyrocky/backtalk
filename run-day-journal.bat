@echo off
rem Jarvis Day Journal -- launcher for the "Jarvis Day Journal"
rem Windows Scheduled Task (see day_journal.py for the exact
rem registration command Captain must approve) and for manual runs
rem while testing. Everything here stays on D: -- the interpreter,
rem the repo, and the failure log. This file is the ONLY thing the
rem scheduled task is ever pointed at; it never invokes python.exe
rem directly.
setlocal

set "REPO=%~dp0"
cd /d "%REPO%"

rem The verified Backtalk venv interpreter -- the SAME one `uv run`
rem already uses for every other backtalk entry point. No fallback to
rem a different python: day_journal must run with the interpreter
rem Captain named, or fail loudly and say why, never silently pick
rem something else.
set "PYEXE=%REPO%.venv\Scripts\python.exe"
set "LOGDIR=%REPO%logs\day_journal"
set "LOGFILE=%LOGDIR%\launcher.log"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

if not exist "%PYEXE%" (
  echo %DATE% %TIME% ERROR: venv interpreter not found at %PYEXE% >> "%LOGFILE%"
  exit /b 1
)

"%PYEXE%" -m backtalk.day_journal --write >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo %DATE% %TIME% ERROR: day_journal --write exited with code %errorlevel% >> "%LOGFILE%"
  exit /b 1
)

echo %DATE% %TIME% OK >> "%LOGFILE%"
endlocal
