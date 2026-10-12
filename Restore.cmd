@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\restore\restore.ps1" %*
exit /b %errorlevel%
