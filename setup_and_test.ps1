# OceanEmbed Setup and Test Script
# Automated setup, data generation, and validation

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host "   OCEANEMBED: Automated Setup & Test Script" -ForegroundColor Yellow
Write-Host "   Problem Statement ID26066 (INCOIS/MoES)" -ForegroundColor Yellow
Write-Host ("="*80) -ForegroundColor Cyan

# Step 1: Check Python installation
Write-Host "`n[1/6] Checking Python installation..." -ForegroundColor Green
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python not found. Please install Python 3.10 or higher." -ForegroundColor Red
    exit 1
}

# Step 2: Create virtual environment
Write-Host "`n[2/6] Setting up virtual environment..." -ForegroundColor Green
if (Test-Path "venv") {
    Write-Host "  → Virtual environment already exists" -ForegroundColor Yellow
} else {
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Step 3: Activate virtual environment and install dependencies
Write-Host "`n[3/6] Installing dependencies..." -ForegroundColor Green
Write-Host "  → This may take 5-10 minutes..." -ForegroundColor Yellow

& "venv\Scripts\Activate.ps1"

# Upgrade pip
python -m pip install --upgrade pip --quiet

# Install requirements
pip install -r requirements.txt --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to install dependencies" -ForegroundColor Red
    Write-Host "  → Try manually: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Step 4: Validate structure
Write-Host "`n[4/6] Validating project structure..." -ForegroundColor Green
python test_structure.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Structure validation passed" -ForegroundColor Green
} else {
    Write-Host "  ✗ Structure validation failed" -ForegroundColor Red
    exit 1
}

# Step 5: Generate synthetic data
Write-Host "`n[5/6] Generating synthetic ocean data..." -ForegroundColor Green
Write-Host "  → Creating 50 samples of realistic ocean data..." -ForegroundColor Yellow

python -c "from src.data.mock_generator import MockOceanDataGenerator; MockOceanDataGenerator('config.yaml').generate_dataset(50, 'mock_data')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Data generation complete" -ForegroundColor Green
    Write-Host "  → Data saved to: mock_data/" -ForegroundColor Cyan
} else {
    Write-Host "  ✗ Data generation failed" -ForegroundColor Red
    exit 1
}

# Step 6: Run integration test
Write-Host "`n[6/6] Running integration test suite..." -ForegroundColor Green
Write-Host "  → This may take 5-10 minutes..." -ForegroundColor Yellow

python tests/test_pipeline.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n" -NoNewline
    Write-Host ("="*80) -ForegroundColor Green
    Write-Host "   ✓ SETUP COMPLETE - ALL TESTS PASSED!" -ForegroundColor Green
    Write-Host ("="*80) -ForegroundColor Green
    
    Write-Host "`n📚 Next Steps:" -ForegroundColor Yellow
    Write-Host "  1. Train the model:" -ForegroundColor Cyan
    Write-Host "     python src/training/train.py`n" -ForegroundColor White
    
    Write-Host "  2. Launch the API server:" -ForegroundColor Cyan
    Write-Host "     python src/app/api.py`n" -ForegroundColor White
    
    Write-Host "  3. Launch the dashboard:" -ForegroundColor Cyan
    Write-Host "     streamlit run src/app/dashboard.py`n" -ForegroundColor White
    
    Write-Host "  4. Monitor training with TensorBoard:" -ForegroundColor Cyan
    Write-Host "     tensorboard --logdir=logs`n" -ForegroundColor White
    
    Write-Host "📖 Documentation:" -ForegroundColor Yellow
    Write-Host "  • README.md - Main documentation" -ForegroundColor White
    Write-Host "  • QUICKSTART.md - Quick start guide" -ForegroundColor White
    Write-Host "  • PROJECT_SUMMARY.md - Technical specifications" -ForegroundColor White
    
} else {
    Write-Host "`n" -NoNewline
    Write-Host ("="*80) -ForegroundColor Red
    Write-Host "   ✗ TESTS FAILED - Please check errors above" -ForegroundColor Red
    Write-Host ("="*80) -ForegroundColor Red
    exit 1
}

Write-Host "`n🌊 OceanEmbed is ready for subsurface temperature reconstruction!`n" -ForegroundColor Cyan
