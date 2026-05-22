@echo off
setlocal

cd /d "%~dp0"

if not exist "desktop\package.json" (
    echo desktop\package.json not found.
    pause
    exit /b 1
)

where npm >nul 2>nul
if not %errorlevel%==0 (
    echo Could not find npm. Please install Node.js first.
    pause
    exit /b 1
)

if not exist "desktop\node_modules" (
    echo Installing Electron desktop dependencies...
    cd /d "%~dp0desktop"
    set ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/
    npm install
    if not %errorlevel%==0 (
        echo npm install failed.
        pause
        exit /b 1
    )
) else (
    cd /d "%~dp0desktop"
)

npm run start
if not %errorlevel%==0 (
    echo Electron desktop failed to start.
    pause
    exit /b 1
)

