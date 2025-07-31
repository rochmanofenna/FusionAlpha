#!/usr/bin/env python3
"""
Fusion Alpha Training Loop

Complete training pipeline for the integrated BICEP -> ENN -> Fusion Alpha system.
Trains on the synthetic data we generated and validates performance.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project paths
sys.path.append('/home/ryan/trading/mismatch-trading')
sys.path.append('/home/ryan/trading/BICEP/src')
sys.path.append('/home/ryan/trading/ENN')

# Set environment for CPU-only training (no CuPy needed)
os.environ['DISABLE_CUPY'] = '1'

# Import our components
try:
    from fusion_alpha.config.integrated_pipeline_config import get_production_config
    from fusion_alpha.pipelines.end_to_end_integration import IntegratedTradingPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Pipeline components not available: {e}")
    print("Will create standalone training loop")
    PIPELINE_AVAILABLE = False

# Setup directories - works both locally and on Colab
if os.path.exists("/content"):
    # Running on Colab
    TRAINING_DIR = Path("/content/mismatch-trading/training")
    DATA_PATH = "/content/mismatch-trading/training_data/synthetic_training_data.parquet"
else:
    # Running locally
    TRAINING_DIR = Path("/home/ryan/trading/mismatch-trading/training")
    DATA_PATH = "/home/ryan/trading/mismatch-trading/training_data/synthetic_training_data.parquet"

TRAINING_DIR.mkdir(exist_ok=True)
(TRAINING_DIR / "logs").mkdir(exist_ok=True)
(TRAINING_DIR / "checkpoints").mkdir(exist_ok=True)
(TRAINING_DIR / "results").mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TRAINING_DIR / 'logs' / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Data parameters
    batch_size: int = 64
    val_split: float = 0.2
    test_split: float = 0.1
    
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    
    # Model parameters
    tech_input_dim: int = 10
    hidden_dim: int = 256
    output_dim: int = 1
    dropout: float = 0.1
    
    # Training settings
    device: str = 'cpu'  # Use CPU for compatibility
    save_every: int = 10  # Save checkpoint every N epochs
    early_stopping_patience: int = 15
    
    # Loss weights
    prediction_weight: float = 1.0
    contradiction_weight: float = 0.5
    consistency_weight: float = 0.1

class TradingDataset(Dataset):
    """Dataset class for training data"""
    
    def __init__(self, data_path: str, split: str = 'train'):
        """
        Initialize dataset
        
        Args:
            data_path: Path to the training data parquet file
            split: 'train', 'val', or 'test'
        """
        self.data_path = data_path
        self.split = split
        
        # Load data
        self.df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(self.df)} samples from {data_path}")
        
        # Features we'll use for training
        self.tech_features = [
            'rsi', 'macd', 'bb_upper', 'bb_lower', 'volume_ratio',
            'volatility', 'atr', 'returns', 'volume_ratio', 'hour'
        ]
        
        # Prepare features
        self._prepare_features()
        
    def _prepare_features(self):
        """Prepare and normalize features"""
        
        # Technical features
        tech_data = self.df[self.tech_features].fillna(0)
        self.tech_features_tensor = torch.FloatTensor(tech_data.values)
        
        # Normalize technical features
        self.tech_mean = self.tech_features_tensor.mean(dim=0)
        self.tech_std = self.tech_features_tensor.std(dim=0) + 1e-8
        self.tech_features_tensor = (self.tech_features_tensor - self.tech_mean) / self.tech_std
        
        # Price movements for contradiction detection
        self.price_movements = torch.FloatTensor(self.df['returns'].fillna(0).values)
        
        # Sentiment scores
        self.sentiment_scores = torch.FloatTensor(self.df['sentiment_score'].fillna(0).values)
        
        # Mock FinBERT embeddings (768-dim) - in production would be real embeddings
        self.finbert_embeddings = torch.randn(len(self.df), 768)
        
        # Targets - future returns
        self.targets = torch.FloatTensor(self.df['future_return_5m'].fillna(0).values)
        
        # Contradiction labels for auxiliary loss
        contradiction_map = {'none': 0, 'overhype': 1, 'underhype': 2, 'paradox': 3}
        self.contradiction_labels = torch.LongTensor([
            contradiction_map.get(c, 0) for c in self.df['contradiction_type']
        ])
        
        # Metadata
        self.symbols = self.df['symbol'].values
        self.timestamps = self.df['timestamp'].values
        
        logger.info(f"Features prepared: {self.tech_features_tensor.shape[1]} tech features")
        logger.info(f"Contradiction distribution: {np.bincount(self.contradiction_labels.numpy())}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'finbert_embeddings': self.finbert_embeddings[idx],
            'tech_features': self.tech_features_tensor[idx],
            'price_movements': self.price_movements[idx],
            'sentiment_scores': self.sentiment_scores[idx],
            'targets': self.targets[idx],
            'contradiction_labels': self.contradiction_labels[idx],
            'symbol': self.symbols[idx],
            'timestamp': str(self.timestamps[idx])
        }

class SimpleFusionAlphaModel(nn.Module):
    """Simplified Fusion Alpha model for training (if full pipeline not available)"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        self.config = config
        
        # FinBERT embedding processor
        self.finbert_processor = nn.Sequential(
            nn.Linear(768, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Technical features processor
        self.tech_processor = nn.Sequential(
            nn.Linear(config.tech_input_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Sentiment processor
        self.sentiment_processor = nn.Sequential(
            nn.Linear(1, config.hidden_dim // 4),
            nn.ReLU()
        )
        
        # Fusion layer
        fusion_input_dim = config.hidden_dim // 2 + config.hidden_dim // 2 + config.hidden_dim // 4
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Output heads
        self.prediction_head = nn.Linear(config.hidden_dim // 2, 1)
        self.contradiction_head = nn.Linear(config.hidden_dim // 2, 4)  # 4 contradiction types
        
    def forward(self, finbert_emb, tech_features, sentiment_scores):
        # Process each modality
        finbert_processed = self.finbert_processor(finbert_emb)
        tech_processed = self.tech_processor(tech_features)
        sentiment_processed = self.sentiment_processor(sentiment_scores.unsqueeze(-1))
        
        # Fuse modalities
        fused = torch.cat([finbert_processed, tech_processed, sentiment_processed], dim=-1)
        fused = self.fusion_layer(fused)
        
        # Generate outputs
        predictions = self.prediction_head(fused).squeeze(-1)
        contradiction_logits = self.contradiction_head(fused)
        
        return predictions, contradiction_logits

class FusionAlphaTrainer:
    """Main trainer class"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup tensorboard logging
        self.writer = SummaryWriter(TRAINING_DIR / 'logs' / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Load data
        self.setup_data()
        
        # Setup model
        self.setup_model()
        
        # Setup training
        self.setup_training()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        logger.info("Trainer initialized successfully")
        
    def setup_data(self):
        """Setup data loaders"""        
        if not Path(DATA_PATH).exists():
            raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
        
        # Create dataset
        full_dataset = TradingDataset(DATA_PATH)
        
        # Split data
        total_size = len(full_dataset)
        test_size = int(total_size * self.config.test_split)
        val_size = int(total_size * self.config.val_split)
        train_size = total_size - val_size - test_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=0,  # Use 0 for compatibility
            pin_memory=False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"Data splits - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        
    def setup_model(self):
        """Setup model and move to device"""
        if PIPELINE_AVAILABLE:
            try:
                # Use the full integrated pipeline
                pipeline_config = get_production_config()
                self.model = IntegratedTradingPipeline(pipeline_config)
                logger.info("Using full integrated pipeline")
            except Exception as e:
                logger.warning(f"Failed to create integrated pipeline: {e}")
                logger.info("Falling back to simple model")
                self.model = SimpleFusionAlphaModel(self.config)
        else:
            # Use simplified model
            self.model = SimpleFusionAlphaModel(self.config)
            logger.info("Using simplified Fusion Alpha model")
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
    def setup_training(self):
        """Setup optimizer, scheduler, and loss functions"""
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        logger.info("Training setup complete")
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'prediction': 0.0,
            'contradiction': 0.0,
            'consistency': 0.0
        }
        
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            finbert_emb = batch['finbert_embeddings'].to(self.device)
            tech_features = batch['tech_features'].to(self.device)
            sentiment_scores = batch['sentiment_scores'].to(self.device)
            price_movements = batch['price_movements'].to(self.device)
            targets = batch['targets'].to(self.device)
            contradiction_labels = batch['contradiction_labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                if hasattr(self.model, 'forward') and isinstance(self.model, SimpleFusionAlphaModel):
                    # Simple model
                    predictions, contradiction_logits = self.model(finbert_emb, tech_features, sentiment_scores)
                else:
                    # Integrated pipeline
                    results = self.model.forward(finbert_emb, tech_features, price_movements, sentiment_scores)
                    predictions = results['predictions']
                    # Create mock contradiction logits for compatibility
                    contradiction_logits = torch.randn(len(predictions), 4, device=self.device)
            except Exception as e:
                logger.error(f"Forward pass failed: {e}")
                continue
            
            # Calculate losses
            prediction_loss = self.mse_loss(predictions, targets)
            contradiction_loss = self.ce_loss(contradiction_logits, contradiction_labels)
            
            # Consistency loss (predictions should be consistent with price movements)
            consistency_loss = self.mse_loss(
                torch.sign(predictions), 
                torch.sign(price_movements)
            )
            
            # Combined loss
            total_loss = (
                self.config.prediction_weight * prediction_loss +
                self.config.contradiction_weight * contradiction_loss +
                self.config.consistency_weight * consistency_loss
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            epoch_losses['total'] += total_loss.item()
            epoch_losses['prediction'] += prediction_loss.item()
            epoch_losses['contradiction'] += contradiction_loss.item()
            epoch_losses['consistency'] += consistency_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Pred': f"{prediction_loss.item():.4f}",
                'Contra': f"{contradiction_loss.item():.4f}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'prediction': 0.0,
            'contradiction': 0.0,
            'consistency': 0.0
        }
        
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                finbert_emb = batch['finbert_embeddings'].to(self.device)
                tech_features = batch['tech_features'].to(self.device)
                sentiment_scores = batch['sentiment_scores'].to(self.device)
                price_movements = batch['price_movements'].to(self.device)
                targets = batch['targets'].to(self.device)
                contradiction_labels = batch['contradiction_labels'].to(self.device)
                
                # Forward pass
                try:
                    if hasattr(self.model, 'forward') and isinstance(self.model, SimpleFusionAlphaModel):
                        predictions, contradiction_logits = self.model(finbert_emb, tech_features, sentiment_scores)
                    else:
                        results = self.model.forward(finbert_emb, tech_features, price_movements, sentiment_scores)
                        predictions = results['predictions']
                        contradiction_logits = torch.randn(len(predictions), 4, device=self.device)
                except Exception as e:
                    logger.error(f"Validation forward pass failed: {e}")
                    continue
                
                # Calculate losses
                prediction_loss = self.mse_loss(predictions, targets)
                contradiction_loss = self.ce_loss(contradiction_logits, contradiction_labels)
                consistency_loss = self.mse_loss(torch.sign(predictions), torch.sign(price_movements))
                
                total_loss = (
                    self.config.prediction_weight * prediction_loss +
                    self.config.contradiction_weight * contradiction_loss +
                    self.config.consistency_weight * consistency_loss
                )
                
                # Update metrics
                val_losses['total'] += total_loss.item()
                val_losses['prediction'] += prediction_loss.item()
                val_losses['contradiction'] += contradiction_loss.item()
                val_losses['consistency'] += consistency_loss.item()
                num_batches += 1
                
                # Collect predictions for metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # Calculate additional metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Correlation
        correlation = np.corrcoef(all_predictions, all_targets)[0, 1]
        val_losses['correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        # Direction accuracy
        pred_direction = np.sign(all_predictions)
        true_direction = np.sign(all_targets)
        direction_accuracy = np.mean(pred_direction == true_direction)
        val_losses['direction_accuracy'] = direction_accuracy
        
        return val_losses
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = TRAINING_DIR / 'checkpoints' / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = TRAINING_DIR / 'checkpoints' / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {metrics['total']:.4f}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Validation/{key}', value, epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['total'])
            
            # Log epoch results
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['total']:.4f}, "
                       f"Val Loss: {val_metrics['total']:.4f}, "
                       f"Correlation: {val_metrics.get('correlation', 0):.3f}, "
                       f"Direction Acc: {val_metrics.get('direction_accuracy', 0):.3f}")
            
            # Save checkpoint
            is_best = val_metrics['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total']
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            if epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Final test evaluation
        logger.info("Training complete. Evaluating on test set...")
        test_metrics = self.test()
        
        # Save final results
        self.save_results(test_metrics)
        
        logger.info("Training finished!")
        
    def test(self) -> Dict[str, float]:
        """Test the model"""
        self.model.eval()
        
        # Load best model
        best_checkpoint = torch.load(TRAINING_DIR / 'checkpoints' / 'best_model.pt', map_location=self.device)
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        test_losses = {'total': 0.0, 'prediction': 0.0}
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                finbert_emb = batch['finbert_embeddings'].to(self.device)
                tech_features = batch['tech_features'].to(self.device)
                sentiment_scores = batch['sentiment_scores'].to(self.device)
                price_movements = batch['price_movements'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                try:
                    if isinstance(self.model, SimpleFusionAlphaModel):
                        predictions, _ = self.model(finbert_emb, tech_features, sentiment_scores)
                    else:
                        results = self.model.forward(finbert_emb, tech_features, price_movements, sentiment_scores)
                        predictions = results['predictions']
                except Exception as e:
                    logger.error(f"Test forward pass failed: {e}")
                    continue
                
                prediction_loss = self.mse_loss(predictions, targets)
                
                test_losses['total'] += prediction_loss.item()
                test_losses['prediction'] += prediction_loss.item()
                num_batches += 1
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        for key in test_losses:
            test_losses[key] /= num_batches
        
        # Calculate test metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        correlation = np.corrcoef(all_predictions, all_targets)[0, 1]
        test_losses['correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        direction_accuracy = np.mean(np.sign(all_predictions) == np.sign(all_targets))
        test_losses['direction_accuracy'] = direction_accuracy
        
        logger.info(f"Test Results - Loss: {test_losses['total']:.4f}, "
                   f"Correlation: {test_losses['correlation']:.3f}, "
                   f"Direction Accuracy: {test_losses['direction_accuracy']:.3f}")
        
        return test_losses
    
    def save_results(self, test_metrics: Dict[str, float]):
        """Save final training results"""
        results = {
            'training_config': asdict(self.config),
            'training_completed': datetime.now().isoformat(),
            'total_epochs': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'test_metrics': test_metrics,
            'model_path': str(TRAINING_DIR / 'checkpoints' / 'best_model.pt')
        }
        
        results_path = TRAINING_DIR / 'results' / f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")

def main():
    """Main training script"""
    print("Fusion Alpha Training")
    print("="*50)
    
    # Training configuration - auto-detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128 if device == 'cuda' else 32  # Larger batch for GPU
    
    config = TrainingConfig(
        batch_size=batch_size,
        num_epochs=100,        # More epochs for GPU training
        learning_rate=1e-3,
        device=device
    )
    
    print(f"Training configuration:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Create trainer
    trainer = FusionAlphaTrainer(config)
    
    # Start training
    trainer.train()
    
    print("\nTraining complete!")
    print(f"Results saved in: {TRAINING_DIR}")
    print("\nNext steps:")
    print("1. Build backtesting framework")
    print("2. Test trained model on historical data")
    print("3. Set up paper trading")

if __name__ == "__main__":
    main()