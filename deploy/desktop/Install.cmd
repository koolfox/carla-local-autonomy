@echo off
setlocal
title CARLA Vision - Setup
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "SetupExit=%errorlevel%"
if not "%SetupExit%"=="0" (
  echo.
  echo Setup did not finish. Read the message above before retrying.
  pause
)
exit /b %SetupExit%
