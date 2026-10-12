@echo off
setlocal
title CARLA Vision
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" %*
set "LauncherExit=%errorlevel%"
if not "%LauncherExit%"=="0" pause
exit /b %LauncherExit%
