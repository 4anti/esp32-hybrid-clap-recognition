@echo off
cd /d "%~dp0"
echo Clap Lab
if not defined PORT set PORT=8788
echo PC:     http://127.0.0.1:%PORT%/
echo Default: private PC-only recording server.
echo Optional trusted LAN access: set CLAP_LAB_HOST=0.0.0.0 before starting.
echo LAN mode allows reachable devices to read, create, and delete recordings.
node server.js
pause
