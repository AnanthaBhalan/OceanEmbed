# OceanEmbed Demo Launcher
# Launches both API and Dashboard for hackathon presentation

param(
    [switch]$API,
    [switch]$Dashboard,
    [switch]$Both,
    [switch]$Assets
)

$ErrorActionPreference = "Stop"

# Banner
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   🌊 OceanEmbed Demo Launcher - Hackathon Edition" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

function Show-Help {
    Write-Host "Usage:" -ForegroundColor Green
    Write-Host "  .\launch_demo.ps1 -Assets          # Open assets folder" -ForegroundColor White
    Write-Host "  .\launch_demo.ps1 -API             # Launch API only" -ForegroundColor White
    Write-Host "  .\launch_demo.ps1 -Dashboard       # Launch Dashboard only" -ForegroundColor White
    Write-Host "  .\launch_demo.ps1 -Both            # Launch both (recommended)" -ForegroundColor White
    Write-Host ""
    Write-Host "Quick Commands:" -ForegroundColor Green
    Write-Host "  explorer assets\                   # View generated plots" -ForegroundColor White
    Write-Host "  start assets\01_spatial_depth_slices.png  # Open specific plot" -ForegroundColor White
    Write-Host ""
}

function Open-Assets {
    Write-Host "📊 Opening assets folder..." -ForegroundColor Green
    if (Test-Path "assets") {
        explorer assets\
        Write-Host "✓ Assets folder opened in File Explorer" -ForegroundColor Green
    } else {
        Write-Host "✗ Assets folder not found. Generate first with:" -ForegroundColor Red
        Write-Host "  python scripts\generate_demo_assets.py" -ForegroundColor Yellow
    }
}

function Start-API {
    Write-Host ""
    Write-Host "🚀 Launching FastAPI Backend..." -ForegroundColor Green
    Write-Host "   URL: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "   Docs: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the API server" -ForegroundColor Yellow
    Write-Host ""
    
    python src\app\api.py
}

function Start-Dashboard {
    Write-Host ""
    Write-Host "🎨 Launching Streamlit Dashboard..." -ForegroundColor Green
    Write-Host "   URL: http://localhost:8501" -ForegroundColor Cyan
    Write-Host "   Theme: Tactical Dark (Emerald/Gold)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "The dashboard will open automatically in your browser" -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Yellow
    Write-Host ""
    
    streamlit run src\app\dashboard.py
}

function Start-Both {
    Write-Host ""
    Write-Host "🚀 Launching Full Demo Stack..." -ForegroundColor Green
    Write-Host ""
    Write-Host "This will open TWO terminal windows:" -ForegroundColor Yellow
    Write-Host "  1. API Backend (http://localhost:8000)" -ForegroundColor Cyan
    Write-Host "  2. Streamlit Dashboard (http://localhost:8501)" -ForegroundColor Cyan
    Write-Host ""
    
    # Launch API in new window
    Write-Host "Starting API server in new window..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host '🚀 OceanEmbed API Server' -ForegroundColor Cyan; python src\app\api.py"
    
    Start-Sleep -Seconds 2
    
    # Launch Dashboard in new window
    Write-Host "Starting Dashboard in new window..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host '🎨 OceanEmbed Dashboard' -ForegroundColor Cyan; streamlit run src\app\dashboard.py"
    
    Write-Host ""
    Write-Host "✓ Both services launched!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access Points:" -ForegroundColor Yellow
    Write-Host "  • Dashboard: http://localhost:8501" -ForegroundColor Cyan
    Write-Host "  • API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  • API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To stop: Close the two PowerShell windows" -ForegroundColor Yellow
    Write-Host ""
}

# Main execution
if (-not ($API -or $Dashboard -or $Both -or $Assets)) {
    Show-Help
    exit 0
}

if ($Assets) {
    Open-Assets
}

if ($API) {
    Start-API
}

if ($Dashboard) {
    Start-Dashboard
}

if ($Both) {
    Start-Both
}
