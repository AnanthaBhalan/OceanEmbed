"""
Integration Test for OceanEmbed Pipeline
Tests complete workflow from data generation to inference.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import shutil
import yaml

from src.data.mock_generator import MockOceanDataGenerator
from src.data.preprocessor import OceanDataPreprocessor
from src.data.dataset import OceanDataset, create_dataloaders
from src.models.ocean_embed_net import OceanEmbedNet
from src.training.train import OceanTrainer
from src.evaluation.metrics import OceanMetrics, compute_depth_wise_metrics
from src.evaluation.argo_validator import ArgoValidator


def test_data_generation():
    """Test mock data generation."""
    print("\n" + "="*70)
    print("TEST 1: Data Generation")
    print("="*70)
    
    generator = MockOceanDataGenerator("config.yaml")
    
    # Generate sample
    surface_ds, subsurface_ds = generator.generate_sample(time_idx=0)
    
    # Check shapes
    assert surface_ds.SST.shape == (101, 241), "SST shape mismatch"
    assert subsurface_ds.temperature.shape == (15, 101, 241), "Temperature shape mismatch"
    
    # Check channels
    assert 'SST' in surface_ds, "Missing SST"
    assert 'SSS' in surface_ds, "Missing SSS"
    assert 'SSH' in surface_ds, "Missing SSH"
    
    # Generate dataset
    test_data_dir = "test_mock_data"
    generator.generate_dataset(num_samples=10, output_dir=test_data_dir)
    
    print("✓ Data generation test passed")
    return test_data_dir


def test_preprocessing(data_dir):
    """Test data preprocessing."""
    print("\n" + "="*70)
    print("TEST 2: Data Preprocessing")
    print("="*70)
    
    preprocessor = OceanDataPreprocessor("config.yaml")
    preprocessor.compute_statistics(data_dir)
    
    # Check statistics
    assert 'SST' in preprocessor.stats, "Missing SST statistics"
    assert 'temperature' in preprocessor.stats, "Missing temperature statistics"
    
    # Test normalization
    test_data = np.array([20, 25, 30])
    normalized = preprocessor.normalize(test_data, 'SST')
    denormalized = preprocessor.denormalize(normalized, 'SST')
    
    assert np.allclose(test_data, denormalized, rtol=1e-5), "Normalization roundtrip failed"
    
    # Save and load statistics
    preprocessor.save_statistics("test_stats.pkl")
    
    preprocessor2 = OceanDataPreprocessor("config.yaml")
    preprocessor2.load_statistics("test_stats.pkl")
    
    assert preprocessor.stats['SST']['mean'] == preprocessor2.stats['SST']['mean'], \
        "Statistics save/load failed"
    
    print("✓ Preprocessing test passed")
    return preprocessor


def test_dataset(data_dir, preprocessor):
    """Test PyTorch dataset."""
    print("\n" + "="*70)
    print("TEST 3: Dataset Loading")
    print("="*70)
    
    dataset = OceanDataset(data_dir, preprocessor, split='train')
    
    # Check dataset size
    assert len(dataset) > 0, "Empty dataset"
    
    # Get sample
    surface, target, metadata = dataset[0]
    
    # Check shapes
    assert surface.shape == (8, 101, 241), f"Surface shape mismatch: {surface.shape}"
    assert target.shape == (15, 101, 241), f"Target shape mismatch: {target.shape}"
    
    # Check types
    assert isinstance(surface, torch.Tensor), "Surface not a tensor"
    assert isinstance(target, torch.Tensor), "Target not a tensor"
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir,
        preprocessor,
        batch_size=2,
        num_workers=0
    )
    
    # Test batch
    batch = next(iter(train_loader))
    batch_surface, batch_target, batch_metadata = batch
    
    assert batch_surface.shape == (2, 8, 101, 241), "Batch surface shape mismatch"
    assert batch_target.shape == (2, 15, 101, 241), "Batch target shape mismatch"
    
    print("✓ Dataset test passed")
    return train_loader, val_loader


def test_model():
    """Test model architecture."""
    print("\n" + "="*70)
    print("TEST 4: Model Architecture")
    print("="*70)
    
    model = OceanEmbedNet("config.yaml")
    
    # Print parameters
    params = model.count_parameters()
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    dummy_input = torch.randn(2, 8, 101, 241).to(device)
    output = model(dummy_input)
    
    assert output.shape == (2, 15, 101, 241), f"Output shape mismatch: {output.shape}"
    
    # Test loss computation
    dummy_target = torch.randn(2, 15, 101, 241).to(device)
    loss_dict = model.compute_loss(output, dummy_target, dummy_input)
    
    assert 'loss' in loss_dict, "Missing loss"
    assert 'mse_loss' in loss_dict, "Missing MSE loss"
    assert 'stratification_loss' in loss_dict, "Missing stratification loss"
    
    print("✓ Model test passed")
    return model


def test_training(model, train_loader, val_loader):
    """Test training loop."""
    print("\n" + "="*70)
    print("TEST 5: Training (1 Epoch)")
    print("="*70)
    
    trainer = OceanTrainer(model, "config.yaml")
    
    # Train for 1 epoch
    train_metrics = trainer.train_epoch(train_loader)
    
    print(f"  Train loss: {train_metrics['loss']:.4f}")
    print(f"  MSE loss: {train_metrics['mse_loss']:.4f}")
    
    # Validate
    val_metrics = trainer.validate(val_loader)
    
    print(f"  Val loss: {val_metrics['loss']:.4f}")
    print(f"  Val RMSE: {val_metrics['avg_rmse']:.4f}")
    
    # Save checkpoint
    trainer.save_checkpoint("test_checkpoint.pth")
    
    # Load checkpoint
    trainer2 = OceanTrainer(OceanEmbedNet("config.yaml"), "config.yaml")
    trainer2.load_checkpoint("test_checkpoint.pth")
    
    print("✓ Training test passed")
    return trainer


def test_inference(model, data_dir, preprocessor):
    """Test inference."""
    print("\n" + "="*70)
    print("TEST 6: Inference")
    print("="*70)
    
    import xarray as xr
    from pathlib import Path
    
    # Load test sample
    surface_files = sorted(Path(data_dir).glob("surface/*.nc"))
    surface_ds = xr.open_dataset(surface_files[0])
    
    # Preprocess
    surface_tensor = preprocessor.preprocess_surface(surface_ds)
    surface_tensor, _ = preprocessor.handle_nan_mask(
        surface_tensor,
        np.zeros((15, 101, 241))
    )
    
    # Predict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    surface_torch = torch.from_numpy(surface_tensor).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        pred = model(surface_torch)
    
    assert pred.shape == (1, 15, 101, 241), f"Prediction shape mismatch: {pred.shape}"
    
    # Denormalize
    pred_np = pred.cpu().numpy()[0]
    pred_denorm = np.zeros_like(pred_np)
    
    for d in range(pred_np.shape[0]):
        pred_denorm[d] = preprocessor.denormalize(pred_np[d], 'temperature')
    
    # Check reasonable values
    assert np.nanmean(pred_denorm) > 0, "Temperature should be positive"
    assert np.nanmean(pred_denorm) < 40, "Temperature too high"
    
    print(f"  Mean predicted temperature: {np.nanmean(pred_denorm):.2f} °C")
    print(f"  Temperature range: [{np.nanmin(pred_denorm):.2f}, {np.nanmax(pred_denorm):.2f}] °C")
    
    print("✓ Inference test passed")
    return pred_denorm


def test_evaluation(pred, target):
    """Test evaluation metrics."""
    print("\n" + "="*70)
    print("TEST 7: Evaluation Metrics")
    print("="*70)
    
    # Create tensors
    pred_tensor = torch.from_numpy(pred).unsqueeze(0)
    target_tensor = torch.from_numpy(target).unsqueeze(0)
    mask_tensor = torch.ones(1, 1, 101, 241)
    
    # Compute metrics
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    metrics = compute_depth_wise_metrics(
        pred_tensor,
        target_tensor,
        mask_tensor,
        config['depth_levels']
    )
    
    # Check metrics
    assert 'rmse' in metrics, "Missing RMSE"
    assert 'correlation' in metrics, "Missing correlation"
    assert len(metrics['rmse']) == 15, "Wrong number of depth levels"
    
    # Print summary
    metrics_calc = OceanMetrics("config.yaml")
    metrics_calc.print_metrics_summary(metrics, "Test Metrics")
    
    print("✓ Evaluation test passed")


def test_api_imports():
    """Test API module imports."""
    print("\n" + "="*70)
    print("TEST 8: API Imports")
    print("="*70)
    
    try:
        from src.app import api
        print("  ✓ API module imported successfully")
    except Exception as e:
        print(f"  ✗ API import failed: {e}")
    
    try:
        from src.app import dashboard
        print("  ✓ Dashboard module imported successfully")
    except Exception as e:
        print(f"  ✗ Dashboard import failed: {e}")
    
    print("✓ API imports test passed")


def cleanup(data_dir):
    """Clean up test files."""
    print("\n" + "="*70)
    print("Cleanup")
    print("="*70)
    
    # Remove test data
    if Path(data_dir).exists():
        shutil.rmtree(data_dir)
        print(f"  ✓ Removed {data_dir}")
    
    # Remove test checkpoint
    if Path("test_checkpoint.pth").exists():
        Path("test_checkpoint.pth").unlink()
        print("  ✓ Removed test checkpoint")
    
    # Remove test statistics
    if Path("test_stats.pkl").exists():
        Path("test_stats.pkl").unlink()
        print("  ✓ Removed test statistics")
    
    if Path("test_argo_report.txt").exists():
        Path("test_argo_report.txt").unlink()
        print("  ✓ Removed test ARGO report")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*80)
    print(" " * 20 + "OCEANEMBED INTEGRATION TEST")
    print("="*80)
    
    try:
        # Test 1: Data generation
        data_dir = test_data_generation()
        
        # Test 2: Preprocessing
        preprocessor = test_preprocessing(data_dir)
        
        # Test 3: Dataset
        train_loader, val_loader = test_dataset(data_dir, preprocessor)
        
        # Test 4: Model
        model = test_model()
        
        # Test 5: Training
        trainer = test_training(model, train_loader, val_loader)
        
        # Test 6: Inference
        pred = test_inference(model, data_dir, preprocessor)
        
        # Test 7: Evaluation
        # Load ground truth for comparison
        import xarray as xr
        subsurface_files = sorted(Path(data_dir).glob("subsurface/*.nc"))
        subsrf_ds = xr.open_dataset(subsurface_files[0])
        target = preprocessor.preprocess_subsurface(subsrf_ds)
        target_denorm = np.zeros_like(target)
        for d in range(target.shape[0]):
            target_denorm[d] = preprocessor.denormalize(target[d], 'temperature')
        
        test_evaluation(pred, target_denorm)
        
        # Test 8: API imports
        test_api_imports()
        
        # Success
        print("\n" + "="*80)
        print(" " * 25 + "✓ ALL TESTS PASSED")
        print("="*80 + "\n")
        
        return True
    
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        cleanup(data_dir)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
