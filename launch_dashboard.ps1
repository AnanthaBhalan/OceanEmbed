# Quick Dashboard Launcher for OceanEmbed Demo
# Uses lightweight demo version (no torch/model dependencies required)

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   🌊 OceanEmbed Dashboard - Demo Version" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting Streamlit dashboard..." -ForegroundColor Green
Write-Host ""
Write-Host "URL: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Theme: Tactical Dark (Emerald/Gold)" -ForegroundColor Cyan
Write-Host ""
Write-Host "The dashboard will open automatically in your browser" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

streamlit run src\app\dashboard_demo.py
