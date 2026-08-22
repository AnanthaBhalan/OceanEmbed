"""
Integration Test for Spatiotemporal OceanEmbed Pipeline
Tests 7-day ConvLSTM framework from data generation to inference.
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
from src.models.ocean_embed_net import OceanSpatiotemporalNet
from src.models.convlstm import ConvLSTMCell, SpatiotemporalEncoder
from src.evaluation.metrics import OceanMetrics, compute_depth_wise_metrics


def test_convlstm_components():
    """Test ConvLSTM cell and encoder."""
    print("\n" + "="*70)
    print("TEST 1: ConvLSTM Components")
    print("="*70)
    
    # Test ConvLSTMCell
    print("  Testing ConvLSTMCell...")
    cell = ConvLSTMCell(input_channels=8, hidden_channels=64, kernel_size=3)
    
    B, C, H, W = 2, 8, 101, 241
    x = torch.randn(B, C, H, W)
    h_prev, c_prev = cell.init_hidden(B, (H, W), x.device)
    
    h_next, c_next = cell(x, h_prev, c_prev)
    
    assert h_next.shape == (B, 64, H, W), f"Hidden shape mismatch: {h_next.shape}"
    assert c_next.shape == (B, 64, H, W), f"Cell shape mismatch: {c_next.shape}"
    print(f"    ✓ ConvLSTMCell output shapes correct")
    
    # Test SpatiotemporalEncoder
    print("  Testing SpatiotemporalEncoder...")
    encoder = SpatiotemporalEncoder(
        input_channels=8,
        hidden_channels=[64, 128, 256],
        num_layers=3
    )
    
    B, T, C, H, W = 2, 7, 8, 101, 241
    x_seq = torch.randn(B, T, C, H, W)
    
    layer_output, last_states = encoder(x_seq)
    
    assert layer_output.shape == (B, T, 256, H, W), f"Output shape mismatch: {layer_output.shape}"
    assert len(last_states) == 3, f"Wrong number of layers: {len(last_states)}"
    
    for i, (h, c) in enumerate(last_states):
        expected_channels = [64, 128, 256][i]
        assert h.shape == (B, expected_channels, H, W), f"Layer {i} hidden shape mismatch"
        assert c.shape == (B, expected_channels, H, W), f"Layer {i} cell shape mismatch"
    
    print(f"    ✓ SpatiotemporalEncoder output shapes correct")
    
    # Test final state extraction
    final_h, final_c = encoder.get_final_states(x_seq)
    assert final_h.shape == (B, 256, H, W), "Final hidden shape mismatch"
    assert final_c.shape == (B, 256, H, W), "Final cell shape mismatch"
    print(f"    ✓ Final state extraction working")
    
    print("✓ ConvLSTM components test passed")


def test_temporal_data_generation():
    """Test temporal sequence data generation."""
    print("\n" + "="*70)
    print("TEST 2: Temporal Data Generation (14+ Days)")
    print("="*70)
    
    generator = MockOceanDataGenerator("config.yaml")
    
    # Generate temporal dataset (at least 14 days)
    test_data_dir = "test_temporal_data"
    num_days = 20  # Generate 20 days for robust testing
    
    print(f"  Generating {num_days} days of mock data...")
    generator.generate_dataset(num_samples=num_days, output_dir=test_data_dir)
    
    # Verify files
    surface_dir = Path(test_data_dir) / "surface"
    subsurface_dir = Path(test_data_dir) / "subsurface"
    
    surface_files = sorted(surface_dir.glob("*.nc"))
    subsurface_files = sorted(subsurface_dir.glob("*.nc"))
    
    assert len(surface_files) >= 14, f"Need at least 14 days, got {len(surface_files)}"
    assert len(surface_files) == len(subsurface_files), "Surface/subsurface count mismatch"
    
    print(f"  ✓ Generated {len(surface_files)} days of data")
    print("✓ Temporal data generation test passed")
    
    return test_data_dir


def test_temporal_dataset(data_dir, preprocessor, sequence_length=7):
    """Test dataset with temporal sequences."""
    print("\n" + "="*70)
    print(f"TEST 3: Temporal Dataset Loading (sequence_length={sequence_length})")
    print("="*70)
    
    # Create dataset with temporal sequences
    dataset = OceanDataset(
        data_dir,
        preprocessor,
        split='train',
        sequence_length=sequence_length
    )
    
    print(f"  Dataset size: {len(dataset)} sequences")
    assert len(dataset) > 0, "Empty dataset"
    
    # Get sample
    surface_seq, target, metadata = dataset[0]
    
    # Check shapes
    expected_surface_shape = (sequence_length, 8, 101, 241)
    expected_target_shape = (15, 101, 241)
    
    assert surface_seq.shape == expected_surface_shape, \
        f"Surface shape mismatch: expected {expected_surface_shape}, got {surface_seq.shape}"
    assert target.shape == expected_target_shape, \
        f"Target shape mismatch: expected {expected_target_shape}, got {target.shape}"
    
    print(f"  ✓ Surface sequence shape: {surface_seq.shape}")
    print(f"  ✓ Target shape: {target.shape}")
    
    # Check metadata
    assert 'sequence_start' in metadata, "Missing sequence_start in metadata"
    assert 'sequence_end' in metadata, "Missing sequence_end in metadata"
    assert metadata['sequence_length'] == sequence_length, "Wrong sequence length in metadata"
    
    print(f"  ✓ Sequence: {metadata['sequence_start']} → {metadata['sequence_end']}")
    
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
    
    expected_batch_surface = (2, sequence_length, 8, 101, 241)
    expected_batch_target = (2, 15, 101, 241)
    
    assert batch_surface.shape == expected_batch_surface, \
        f"Batch surface shape mismatch: expected {expected_batch_surface}, got {batch_surface.shape}"
    assert batch_target.shape == expected_batch_target, \
        f"Batch target shape mismatch: expected {expected_batch_target}, got {batch_target.shape}"
    
    print(f"  ✓ Batch surface shape: {batch_surface.shape}")
    print(f"  ✓ Batch target shape: {batch_target.shape}")
    
    print("✓ Temporal dataset test passed")
    return train_loader, val_loader


def test_spatiotemporal_model():
    """Test spatiotemporal model architecture."""
    print("\n" + "="*70)
    print("TEST 4: Spatiotemporal Model Architecture")
    print("="*70)
    
    # Test full encoder
    print("  Testing full encoder (ConvNeXt + ConvLSTM)...")
    model_full = OceanSpatiotemporalNet("config.yaml", encoder_type="full")
    
    params = model_full.count_parameters()
    print(f"    Total parameters: {params['total']:,}")
    print(f"    Trainable parameters: {params['trainable']:,}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_full.to(device)
    
    B, T, C, H, W = 2, 7, 8, 101, 241
    dummy_input = torch.randn(B, T, C, H, W).to(device)
    
    output = model_full(dummy_input)
    
    expected_output_shape = (B, 15, H, W)
    assert output.shape == expected_output_shape, \
        f"Output shape mismatch: expected {expected_output_shape}, got {output.shape}"
    
    print(f"    ✓ Forward pass successful: {dummy_input.shape} → {output.shape}")
    
    # Test lightweight encoder
    print("  Testing lightweight encoder (ConvLSTM only)...")
    model_light = OceanSpatiotemporalNet("config.yaml", encoder_type="lightweight")
    model_light.to(device)
    
    params_light = model_light.count_parameters()
    print(f"    Total parameters: {params_light['total']:,}")
    
    output_light = model_light(dummy_input)
    assert output_light.shape == expected_output_shape, "Lightweight output shape mismatch"
    
    print(f"    ✓ Lightweight encoder successful")
    
    # Test loss computation
    print("  Testing physics-informed loss...")
    dummy_target = torch.randn(B, 15, H, W).to(device)
    loss_dict = model_full.compute_loss(output, dummy_target, dummy_input)
    
    assert 'loss' in loss_dict, "Missing loss"
    assert 'mse_loss' in loss_dict, "Missing MSE loss"
    assert 'stratification_loss' in loss_dict, "Missing stratification loss"
    assert 'gradient_smoothness_loss' in loss_dict, "Missing gradient smoothness loss"
    
    print(f"    ✓ Loss computation successful")
    print(f"      Total loss: {loss_dict['loss'].item():.4f}")
    print(f"      MSE loss: {loss_dict['mse_loss'].item():.4f}")
    print(f"      Stratification loss: {loss_dict['stratification_loss'].item():.4f}")
    
    print("✓ Spatiotemporal model test passed")
    return model_full, model_light


def test_temporal_training(model, train_loader):
    """Test training with temporal sequences."""
    print("\n" + "="*70)
    print("TEST 5: Temporal Training (Forward & Backward Pass)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Get one batch
    batch = next(iter(train_loader))
    surface_seq, target, metadata = batch
    
    surface_seq = surface_seq.to(device)
    target = target.to(device)
    
    print(f"  Input shape: {surface_seq.shape}")
    print(f"  Target shape: {target.shape}")
    
    # Forward pass
    print("  Running forward pass...")
    pred = model(surface_seq)
    
    assert pred.shape == target.shape, f"Prediction shape mismatch: {pred.shape} vs {target.shape}"
    print(f"    ✓ Prediction shape: {pred.shape}")
    
    # Compute loss
    print("  Computing loss...")
    loss_dict = model.compute_loss(pred, target, surface_seq)
    loss = loss_dict['loss']
    
    print(f"    ✓ Loss: {loss.item():.4f}")
    
    # Backward pass
    print("  Running backward pass...")
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients computed"
    print(f"    ✓ Gradients computed successfully")
    
    # Optimizer step
    print("  Running optimizer step...")
    optimizer.step()
    print(f"    ✓ Optimizer step successful")
    
    print("✓ Temporal training test passed")


def test_temporal_inference(model, data_dir, preprocessor):
    """Test inference on temporal sequences."""
    print("\n" + "="*70)
    print("TEST 6: Temporal Inference")
    print("="*70)
    
    import xarray as xr
    
    # Load 7 consecutive days
    surface_files = sorted(Path(data_dir).glob("surface/*.nc"))[:7]
    
    print(f"  Loading {len(surface_files)} days for inference...")
    
    # Preprocess sequence
    surface_sequence = []
    for f in surface_files:
        surface_ds = xr.open_dataset(f)
        surface_tensor = preprocessor.preprocess_surface(surface_ds)
        surface_tensor, _ = preprocessor.handle_nan_mask(
            surface_tensor,
            np.zeros((15, 101, 241))
        )
        surface_sequence.append(surface_tensor)
        surface_ds.close()
    
    # Stack into (T, C, H, W)
    surface_sequence = np.stack(surface_sequence, axis=0)
    
    # Predict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Add batch dimension: (1, T, C, H, W)
    surface_torch = torch.from_numpy(surface_sequence).unsqueeze(0).float().to(device)
    
    print(f"  Input shape: {surface_torch.shape}")
    
    with torch.no_grad():
        pred = model(surface_torch)
    
    expected_shape = (1, 15, 101, 241)
    assert pred.shape == expected_shape, f"Prediction shape mismatch: {pred.shape}"
    
    print(f"  ✓ Prediction shape: {pred.shape}")
    
    # Denormalize
    pred_np = pred.cpu().numpy()[0]
    pred_denorm = np.zeros_like(pred_np)
    
    for d in range(pred_np.shape[0]):
        pred_denorm[d] = preprocessor.denormalize(pred_np[d], 'temperature')
    
    # Check reasonable values
    mean_temp = np.nanmean(pred_denorm)
    min_temp = np.nanmin(pred_denorm)
    max_temp = np.nanmax(pred_denorm)
    
    assert mean_temp > 0, "Mean temperature should be positive"
    assert mean_temp < 40, "Mean temperature too high"
    
    print(f"  ✓ Mean predicted temperature: {mean_temp:.2f} °C")
    print(f"  ✓ Temperature range: [{min_temp:.2f}, {max_temp:.2f}] °C")
    
    print("✓ Temporal inference test passed")
    return pred_denorm


def test_shape_validation():
    """Test tensor shape validation throughout pipeline."""
    print("\n" + "="*70)
    print("TEST 7: Tensor Shape Validation")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    shapes_to_test = [
        ("Single sample", (1, 7, 8, 101, 241)),
        ("Small batch", (2, 7, 8, 101, 241)),
        ("Medium batch", (4, 7, 8, 101, 241)),
        ("Large batch", (8, 7, 8, 101, 241))
    ]
    
    model = OceanSpatiotemporalNet("config.yaml", encoder_type="lightweight")
    model.to(device)
    model.eval()
    
    for name, shape in shapes_to_test:
        print(f"  Testing {name}: {shape}")
        x = torch.randn(*shape).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        expected_output = (shape[0], 15, 101, 241)
        assert output.shape == expected_output, \
            f"Shape mismatch for {name}: expected {expected_output}, got {output.shape}"
        
        print(f"    ✓ {name} passed: {shape} → {output.shape}")
    
    print("✓ Shape validation test passed")


def cleanup(data_dir):
    """Clean up test files."""
    print("\n" + "="*70)
    print("Cleanup")
    print("="*70)
    
    if Path(data_dir).exists():
        shutil.rmtree(data_dir)
        print(f"  ✓ Removed {data_dir}")
    
    if Path("test_stats.pkl").exists():
        Path("test_stats.pkl").unlink()
        print("  ✓ Removed test statistics")


def run_all_spatiotemporal_tests():
    """Run all spatiotemporal integration tests."""
    print("\n" + "="*80)
    print(" " * 15 + "SPATIOTEMPORAL OCEANEMBED TEST SUITE")
    print(" " * 20 + "7-Day ConvLSTM Framework")
    print("="*80)
    
    try:
        # Test 1: ConvLSTM components
        test_convlstm_components()
        
        # Test 2: Temporal data generation
        data_dir = test_temporal_data_generation()
        
        # Preprocessing
        print("\n  [Setup] Computing preprocessing statistics...")
        preprocessor = OceanDataPreprocessor("config.yaml")
        preprocessor.compute_statistics(data_dir)
        preprocessor.save_statistics("test_stats.pkl")
        
        # Test 3: Temporal dataset
        train_loader, val_loader = test_temporal_dataset(data_dir, preprocessor)
        
        # Test 4: Spatiotemporal model
        model_full, model_light = test_spatiotemporal_model()
        
        # Test 5: Temporal training
        test_temporal_training(model_full, train_loader)
        
        # Test 6: Temporal inference
        pred = test_temporal_inference(model_light, data_dir, preprocessor)
        
        # Test 7: Shape validation
        test_shape_validation()
        
        # Success
        print("\n" + "="*80)
        print(" " * 25 + "✓ ALL TESTS PASSED")
        print(" " * 15 + "Spatiotemporal Framework Validated")
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
    success = run_all_spatiotemporal_tests()
    sys.exit(0 if success else 1)
