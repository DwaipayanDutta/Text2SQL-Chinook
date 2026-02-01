@echo off
setlocal enabledelayedexpansion

echo ================================
echo   Smart Streamlit Launcher
echo ================================
echo.

REM ---------- Config ----------
set "VENV_DIR=.venv"
set "REQ_FILE=requirements.txt"
set "APP_FILE=app_streamlit2.py"
set "STAMP_FILE=%VENV_DIR%\req_timestamp.txt"

REM ---------- Detect Python ----------
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON=py -3.12"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python not found in PATH.
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

echo Using interpreter: %PYTHON%
echo.

REM ---------- Create venv ----------
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment...
    %PYTHON% -m venv "%VENV_DIR%"

    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM ---------- Activate ----------
call "%VENV_DIR%\Scripts\activate.bat"

REM ---------- Upgrade pip (VERY important for 3.12) ----------
python -m pip install --upgrade pip setuptools wheel >nul

REM ---------- Requirements timestamp ----------
set CURR_TS=NONE
if exist "%REQ_FILE%" (
    for %%A in ("%REQ_FILE%") do set CURR_TS=%%~tA
)

set OLD_TS=NONE
if exist "%STAMP_FILE%" (
    set /p OLD_TS=<"%STAMP_FILE%"
)

REM ---------- Install dependencies if changed ----------
if not "%CURR_TS%"=="%OLD_TS%" (
    if exist "%REQ_FILE%" (
        echo Installing dependencies...
        python -m pip install -r "%REQ_FILE%"

        if errorlevel 1 (
            echo [ERROR] Dependency installation failed.
            pause
            exit /b 1
        )

        echo %CURR_TS% > "%STAMP_FILE%"
    ) else (
        echo [WARNING] requirements.txt not found — skipping install.
    )
) else (
    echo Dependencies unchanged — fast startup enabled.
)

REM ---------- Launch Streamlit ----------
echo.
echo Starting Streamlit...
python -m streamlit run "%APP_FILE%"

echo.
pause
endlocal
