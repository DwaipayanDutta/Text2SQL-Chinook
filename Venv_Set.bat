@echo off

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app_streamlit2.py

pause
@echo off
setlocal enabledelayedexpansion

echo ================================
echo   Smart Streamlit Launcher
echo ================================
echo.

REM ---- Config ----
set "VENV_DIR=.venv"
set "REQ_FILE=requirements.txt"
set "APP_FILE=app_streamlit2.py"
set "STAMP_FILE=%VENV_DIR%\req_timestamp.txt"

REM ---- Flags ----
set "FORCE_INSTALL=0"
if "%1"=="--reinstall" set "FORCE_INSTALL=1"

REM ---- Check Python ----
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    pause
    exit /b 1
)

REM ---- Create venv if missing ----
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

REM ---- Activate venv ----
call "%VENV_DIR%\Scripts\activate.bat"

REM ---- Get current requirements timestamp ----
if exist "%REQ_FILE%" (
    for %%A in ("%REQ_FILE%") do set CURR_TS=%%~tA
) else (
    set CURR_TS=NONE
)

REM ---- Read stored timestamp ----
set OLD_TS=NONE
if exist "%STAMP_FILE%" (
    set /p OLD_TS=<"%STAMP_FILE%"
)

REM ---- Decide install ----
set NEED_INSTALL=0

if %FORCE_INSTALL%==1 (
    set NEED_INSTALL=1
) else if not "%CURR_TS%"=="%OLD_TS%" (
    set NEED_INSTALL=1
)

REM ---- Install if needed ----
if %NEED_INSTALL%==1 (
    if exist "%REQ_FILE%" (
        echo Installing/updating dependencies...
        pip install -r "%REQ_FILE%"
        if errorlevel 1 (
            echo [ERROR] Dependency install failed.
            pause
            exit /b 1
        )
        echo %CURR_TS% > "%STAMP_FILE%"
    ) else (
        echo [WARNING] requirements.txt not found — skipping install.
    )
) else (
    echo Dependencies unchanged — fast start enabled.
)

REM ---- Run app ----
echo.
echo Starting Streamlit...
streamlit run "%APP_FILE%"

echo.
pause
endlocal
