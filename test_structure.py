"""
Quick structure and syntax validation test (no dependencies required).
"""

import sys
from pathlib import Path
import ast

def check_syntax(filepath):
    """Check if Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def test_project_structure():
    """Test that all required files exist."""
    print("="*70)
    print("TESTING PROJECT STRUCTURE")
    print("="*70)
    
    required_files = [
        "config.yaml",
        "requirements.txt",
        "README.md",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/data/mock_generator.py",
        "src/data/preprocessor.py",
        "src/data/dataset.py",
        "src/models/__init__.py",
        "src/models/encoder.py",
        "src/models/decoder.py",
        "src/models/loss.py",
        "src/models/ocean_embed_net.py",
        "src/evaluation/__init__.py",
        "src/evaluation/metrics.py",
        "src/evaluation/argo_validator.py",
        "src/training/__init__.py",
        "src/training/train.py",
        "src/app/__init__.py",
        "src/app/api.py",
        "src/app/dashboard.py",
        "tests/test_pipeline.py"
    ]
    
    missing_files = []
    for filepath in required_files:
        if not Path(filepath).exists():
            missing_files.append(filepath)
            print(f"  ✗ Missing: {filepath}")
        else:
            print(f"  ✓ Found: {filepath}")
    
    if missing_files:
        print(f"\n✗ Missing {len(missing_files)} files")
        return False
    else:
        print(f"\n✓ All {len(required_files)} required files present")
        return True

def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print("\n" + "="*70)
    print("TESTING PYTHON SYNTAX")
    print("="*70)
    
    python_files = list(Path("src").rglob("*.py")) + list(Path("tests").rglob("*.py"))
    
    errors = []
    for filepath in python_files:
        valid, error = check_syntax(filepath)
        if valid:
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath}: {error}")
            errors.append((filepath, error))
    
    if errors:
        print(f"\n✗ Found {len(errors)} syntax errors")
        return False
    else:
        print(f"\n✓ All {len(python_files)} Python files have valid syntax")
        return True

def test_config_yaml():
    """Test that config.yaml is valid."""
    print("\n" + "="*70)
    print("TESTING CONFIGURATION")
    print("="*70)
    
    try:
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required keys
        required_keys = ['domain', 'input_channels', 'depth_levels', 'model', 'training']
        for key in required_keys:
            if key in config:
                print(f"  ✓ Found config key: {key}")
            else:
                print(f"  ✗ Missing config key: {key}")
                return False
        
        # Check depth levels
        depth_levels = config['depth_levels']
        expected_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
        if depth_levels == expected_depths:
            print(f"  ✓ Correct depth levels: {len(depth_levels)} levels")
        else:
            print(f"  ✗ Incorrect depth levels")
            return False
        
        # Check input channels
        if len(config['input_channels']) == 8:
            print(f"  ✓ Correct number of input channels: 8")
        else:
            print(f"  ✗ Incorrect number of input channels: {len(config['input_channels'])}")
            return False
        
        print("\n✓ Configuration file is valid")
        return True
    
    except Exception as e:
        print(f"\n✗ Configuration error: {e}")
        return False

def test_requirements():
    """Test that requirements.txt exists and has key packages."""
    print("\n" + "="*70)
    print("TESTING REQUIREMENTS")
    print("="*70)
    
    try:
        with open("requirements.txt", 'r') as f:
            requirements = f.read()
        
        key_packages = ['torch', 'xarray', 'fastapi', 'streamlit', 'numpy']
        
        for package in key_packages:
            if package in requirements:
                print(f"  ✓ Found package: {package}")
            else:
                print(f"  ✗ Missing package: {package}")
                return False
        
        print(f"\n✓ All key packages listed in requirements.txt")
        return True
    
    except Exception as e:
        print(f"\n✗ Requirements error: {e}")
        return False

def main():
    """Run all structure tests."""
    print("\n" + "="*80)
    print(" "*20 + "OCEANEMBED STRUCTURE VALIDATION")
    print("="*80 + "\n")
    
    results = []
    
    results.append(("Project Structure", test_project_structure()))
    results.append(("Python Syntax", test_python_syntax()))
    results.append(("Configuration", test_config_yaml()))
    results.append(("Requirements", test_requirements()))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:30s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "="*80)
        print(" "*25 + "✓ ALL CHECKS PASSED")
        print("="*80)
        print("\nProject structure is valid and ready for deployment!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Generate data: python -m src.data.mock_generator")
        print("  3. Run full test: python tests/test_pipeline.py")
        print("  4. Train model: python src/training/train.py")
        print("  5. Launch API: python src/app/api.py")
        print("  6. Launch dashboard: streamlit run src/app/dashboard.py")
        return 0
    else:
        print("\n" + "="*80)
        print(" "*25 + "✗ SOME CHECKS FAILED")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
