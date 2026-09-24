@echo off
rem Relaunch OIModeler App without re-running the full installer (no Python/Git
rem check, no download, no dependency install). This is what the desktop
rem shortcut created by install.ps1 points to; safe to run directly too.
cd /d "%~dp0.."

if not exist env_oim (
    echo No environment found here. Run installer\install.ps1 first.
    exit /b 1
)

call env_oim\Scripts\activate.bat
streamlit run app.py
