@echo off & setlocal EnableDelayedExpansion
for /f "delims=" %%i in ('"dir /a/s/b/on *.jpg"') do (
set file=%%~fi
set file=!file:/=/!
echo !file! >> path.txt
)