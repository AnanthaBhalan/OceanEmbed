"""
Training Module for OceanEmbed
Complete training loop with checkpointing, early stopping, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import yaml
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ocean_embed_net import OceanEmbedNet
from src.data.dataset import create_dataloaders
from src.data.preprocessor import OceanDataPreprocessor
from src.evaluation.metrics import compute_depth_wise_metrics


class OceanTrainer:
    """Trainer for OceanEmbed model."""
    
    def __init__(
        self,
        model: OceanEmbedNet,
        config_path: str = "config.yaml",
        device: str = None
    ):
        """Initialize trainer.
        
        Args:
            model: OceanEmbedNet model
            config_path: Path to configuration file
            device: Device to use for training
        """
        self.model = model
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.train_config = self.config['training']
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        print(f"Training on device: {self.device}")
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config['weight_decay']
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_config['num_epochs']
        )
        
        # Setup directories
        self.checkpoint_dir = Path(self.train_config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.log_dir = Path(self.train_config['log_dir'])
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        
        epoch_losses = {
            'loss': [],
            'mse_loss': [],
            'stratification_loss': [],
            'smoothness_loss': []
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, (surface, target, metadata) in enumerate(pbar):
            # Move to device
            surface = surface.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(surface)
            
            # Compute loss
            loss_dict = self.model.compute_loss(pred, target, surface)
            loss = loss_dict['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Record losses
            for key in epoch_losses.keys():
                epoch_losses[key].append(loss_dict[key].item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        epoch_losses = {
            'loss': [],
            'mse_loss': [],
            'stratification_loss': [],
            'smoothness_loss': []
        }
        
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for surface, target, metadata in tqdm(val_loader, desc="Validation"):
                # Move to device
                surface = surface.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                pred = self.model(surface)
                
                # Compute loss
                loss_dict = self.model.compute_loss(pred, target, surface)
                
                # Record losses
                for key in epoch_losses.keys():
                    epoch_losses[key].append(loss_dict[key].item())
                
                # Collect for metrics calculation
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
                all_masks.append(surface[:, -1:, :, :].cpu())
        
        # Average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        # Compute depth-wise metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        metrics = compute_depth_wise_metrics(
            all_preds,
            all_targets,
            all_masks,
            self.config['depth_levels']
        )
        
        # Add average RMSE to losses
        avg_losses['avg_rmse'] = np.nanmean(metrics['rmse'])
        avg_losses['avg_correlation'] = np.nanmean(metrics['correlation'])
        
        return avg_losses
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved to {best_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            print(f"Checkpoint {filepath} not found")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        print(f"✓ Checkpoint loaded from {filepath}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ):
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (overrides config if provided)
        """
        if num_epochs is None:
            num_epochs = self.train_config['num_epochs']
        
        print(f"\n{'='*70}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            # Log to TensorBoard
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val RMSE: {val_metrics['avg_rmse']:.4f} | Val Corr: {val_metrics['avg_correlation']:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
            
            # Check for best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth", is_best=True)
                self.patience_counter = 0
                print(f"  ✓ New best model (val_loss: {self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.train_config['early_stopping_patience']:
                print(f"\n✓ Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final training history
        history_file = self.checkpoint_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history
            }, f, indent=2)
        
        self.writer.close()
        
        print(f"\n{'='*70}")
        print(f"Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")


def train_model(
    data_dir: str = "mock_data",
    config_path: str = "config.yaml",
    resume_from: Optional[str] = None
):
    """Convenience function to train OceanEmbed model.
    
    Args:
        data_dir: Directory containing training data
        config_path: Path to configuration file
        resume_from: Checkpoint to resume from (optional)
    """
    print("="*70)
    print("OceanEmbed Training Pipeline")
    print("="*70)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize preprocessor
    print("\n1. Loading and preprocessing data...")
    preprocessor = OceanDataPreprocessor(config_path)
    
    # Check if statistics file exists
    stats_file = Path("preprocessor_stats.pkl")
    if stats_file.exists():
        preprocessor.load_statistics(str(stats_file))
    else:
        preprocessor.compute_statistics(data_dir)
        preprocessor.save_statistics(str(stats_file))
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir,
        preprocessor,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\n2. Initializing model...")
    model = OceanEmbedNet(config_path)
    
    params = model.count_parameters()
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    
    # Initialize trainer
    trainer = OceanTrainer(model, config_path)
    
    # Resume from checkpoint if specified
    if resume_from is not None:
        trainer.load_checkpoint(resume_from)
    
    # Train
    print("\n3. Training model...")
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    
    print("\nTest Set Results:")
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Test RMSE: {test_metrics['avg_rmse']:.4f}")
    print(f"  Test Correlation: {test_metrics['avg_correlation']:.4f}")
    
    print("\n✓ Training pipeline complete!")
    
    return model, trainer


if __name__ == "__main__":
    # Train model
    model, trainer = train_model(
        data_dir="mock_data",
        config_path="config.yaml"
    )
