@echo off
REM Quick launcher to view OceanEmbed assets

echo.
echo ========================================
echo   OceanEmbed - View Assets
echo ========================================
echo.

if not exist "assets\" (
    echo [ERROR] Assets folder not found!
    echo.
    echo Please generate assets first:
    echo   python scripts\generate_demo_assets.py
    echo.
    pause
    exit /b 1
)

echo Opening assets folder...
explorer assets\

echo.
echo Assets folder opened in File Explorer
echo.
echo TIP: Drag PNG files directly into PowerPoint/Google Slides
echo.
pause
