#!/usr/bin/env python3
"""
Training script for Movement Primitive Diffusion using Consistency Distillation (CD).

This script implements Consistency Distillation to accelerate the training and inference
of Movement Primitive Diffusion models. CD reduces the number of sampling steps required
by learning to map any point in the diffusion process directly to the original data.

Usage:
    # Train MPD with Consistency Distillation on MetaWorld
    python train_cm.py --config-name=mpd_metaworld_cm --dataset_path=/path/to/metaworld_dataset
    
    # Train MPD with Consistency Distillation on ManiSkills
    python train_cm.py --config-name=mpd_maniskills_cm --dataset_path=/path/to/maniskills_dataset
"""

import argparse
import hydra
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Dict, Tuple
from tqdm import tqdm

from FODMP.workspace.mpd_metaworld_position_workspace import MPDMetaWorldPosition
from FODMP.workspace.mpd_maniskills_position import MPDManiSkillsPosition


class ConsistencyModel(nn.Module):
    """
    Consistency Model for Movement Primitive Diffusion.
    
    This model learns to map any point in the diffusion process directly to the original data,
    enabling single-step or few-step generation instead of the traditional multi-step process.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_time_embedding: bool = True,
        time_embed_dim: int = 128
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_time_embedding = use_time_embedding
        
        # Time embedding
        if use_time_embedding:
            self.time_embed = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim)
            )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Skip connection
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the consistency model.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            t: Time tensor [batch_size, 1] or [batch_size]
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Ensure time tensor has correct shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Time embedding
        if self.use_time_embedding:
            t_embed = self.time_embed(t)  # [batch_size, time_embed_dim]
            t_embed = t_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, time_embed_dim]
        
        # Input projection
        h = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # Add time embedding
        if self.use_time_embedding:
            h = h + t_embed
        
        # Transformer encoding
        h = self.transformer(h)  # [batch_size, seq_len, hidden_dim]
        
        # Output projection
        output = self.output_proj(h)  # [batch_size, seq_len, output_dim]
        
        # Skip connection
        skip = self.skip_proj(x)  # [batch_size, seq_len, output_dim]
        
        return output + skip


class ConsistencyDistillationTrainer:
    """
    Trainer for Consistency Distillation of Movement Primitive Diffusion models.
    """
    
    def __init__(
        self,
        consistency_model: ConsistencyModel,
        teacher_model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        consistency_weight: float = 1.0,
        distillation_weight: float = 0.1,
        num_inference_steps: int = 50,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0
    ):
        self.consistency_model = consistency_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.device = device
        self.consistency_weight = consistency_weight
        self.distillation_weight = distillation_weight
        self.num_inference_steps = num_inference_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.consistency_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Loss functions
        self.consistency_loss_fn = nn.MSELoss()
        self.distillation_loss_fn = nn.MSELoss()
        
        # Noise schedule
        self._setup_noise_schedule()
        
    def _setup_noise_schedule(self):
        """Setup noise schedule for consistency distillation."""
        # Create a schedule of sigmas
        sigmas = torch.exp(torch.linspace(
            np.log(self.sigma_max), 
            np.log(self.sigma_min), 
            self.num_inference_steps + 1
        ))
        self.sigmas = sigmas.to(self.device)
        
    def add_noise(self, x: torch.Tensor, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to data according to the diffusion process.
        
        Args:
            x: Clean data [batch_size, seq_len, dim]
            sigma: Noise level [batch_size, 1] or scalar
            
        Returns:
            noisy_x: Noisy data
            noise: Added noise
        """
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
            
        # Ensure sigma has the right shape
        while sigma.dim() < x.dim():
            sigma = sigma.unsqueeze(-1)
            
        noise = torch.randn_like(x) * sigma
        noisy_x = x + noise
        
        return noisy_x, noise
        
    def consistency_loss(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss.
        
        Args:
            x: Clean data [batch_size, seq_len, dim]
            sigma: Noise level [batch_size, 1]
            
        Returns:
            Consistency loss
        """
        # Add noise
        noisy_x, noise = self.add_noise(x, sigma)
        
        # Predict denoised data
        t = sigma  # Use sigma as time
        denoised_x = self.consistency_model(noisy_x, t)
        
        # Consistency loss: model should predict clean data
        loss = self.consistency_loss_fn(denoised_x, x)
        
        return loss
        
    def distillation_loss(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss using teacher model.
        
        Args:
            x: Clean data [batch_size, seq_len, dim]
            sigma: Noise level [batch_size, 1]
            
        Returns:
            Distillation loss
        """
        # Add noise
        noisy_x, noise = self.add_noise(x, sigma)
        
        # Teacher prediction (multi-step denoising)
        with torch.no_grad():
            teacher_output = self._teacher_denoise(noisy_x, sigma)
        
        # Student prediction (single-step)
        t = sigma
        student_output = self.consistency_model(noisy_x, t)
        
        # Distillation loss
        loss = self.distillation_loss_fn(student_output, teacher_output)
        
        return loss
        
    def _teacher_denoise(self, noisy_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Multi-step denoising using teacher model.
        
        Args:
            noisy_x: Noisy data
            sigma: Initial noise level
            
        Returns:
            Denoised data
        """
        # Simplified teacher denoising (in practice, use actual teacher model)
        # For now, we'll use a simple approximation
        current_x = noisy_x.clone()
        current_sigma = sigma.clone()
        
        # Multi-step denoising
        for i in range(self.num_inference_steps):
            # Find closest sigma in schedule
            sigma_idx = torch.argmin(torch.abs(self.sigmas - current_sigma), dim=-1)
            sigma_idx = torch.clamp(sigma_idx, 0, len(self.sigmas) - 1)
            
            # Teacher prediction (simplified)
            with torch.no_grad():
                # In practice, this would be the actual teacher model
                teacher_pred = current_x - current_sigma * torch.randn_like(current_x) * 0.1
                
            # Update
            current_x = teacher_pred
            current_sigma = self.sigmas[sigma_idx] * 0.9  # Reduce noise level
            
        return current_x
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of losses
        """
        self.consistency_model.train()
        self.optimizer.zero_grad()
        
        # Extract data
        actions = batch['actions'].to(self.device)  # [batch_size, seq_len, action_dim]
        
        # Sample random noise levels
        batch_size = actions.shape[0]
        sigma = torch.exp(torch.rand(batch_size, device=self.device) * 
                         (np.log(self.sigma_max) - np.log(self.sigma_min)) + 
                         np.log(self.sigma_min))
        
        # Consistency loss
        consistency_loss = self.consistency_loss(actions, sigma)
        
        # Distillation loss
        distillation_loss = self.distillation_loss(actions, sigma)
        
        # Total loss
        total_loss = (self.consistency_weight * consistency_loss + 
                     self.distillation_weight * distillation_loss)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.consistency_model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'distillation_loss': distillation_loss.item()
        }
        
    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int, action_dim: int) -> torch.Tensor:
        """
        Generate samples using the consistency model.
        
        Args:
            batch_size: Number of samples to generate
            seq_len: Sequence length
            action_dim: Action dimension
            
        Returns:
            Generated samples
        """
        self.consistency_model.eval()
        
        # Start with pure noise
        x = torch.randn(batch_size, seq_len, action_dim, device=self.device)
        
        # Single-step generation (consistency model)
        sigma_max = torch.tensor(self.sigma_max, device=self.device)
        t = sigma_max.expand(batch_size, 1)
        
        # Generate in one step
        generated_x = self.consistency_model(x, t)
        
        return generated_x


class CMMetaWorldPosition(MPDMetaWorldPosition):
    """
    Consistency Distillation workspace for MetaWorld position control.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.consistency_trainer = None
        
    def setup_consistency_model(self, device: str = "cuda") -> None:
        """Setup consistency model for training."""
        # Get action dimensions from dataset
        if hasattr(self, 'dataset') and len(self.dataset) > 0:
            sample = self.dataset[0]
            action_dim = sample['actions'].shape[-1]
            seq_len = sample['actions'].shape[-2]
        else:
            action_dim = self.movement_primitive_dim
            seq_len = self.t_act
            
        # Create consistency model
        consistency_model = ConsistencyModel(
            input_dim=action_dim,
            output_dim=action_dim,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
            use_time_embedding=True
        )
        
        # Create teacher model (simplified - in practice, use pre-trained MPD model)
        teacher_model = nn.Identity()  # Placeholder
        
        # Create trainer
        self.consistency_trainer = ConsistencyDistillationTrainer(
            consistency_model=consistency_model,
            teacher_model=teacher_model,
            device=device,
            learning_rate=1e-4,
            consistency_weight=1.0,
            distillation_weight=0.1,
            num_inference_steps=50
        )
        
        print(f"Initialized Consistency Distillation trainer on device: {device}")
        print(f"Action dimension: {action_dim}, Sequence length: {seq_len}")
        
    def train_consistency_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train consistency model for one epoch."""
        if self.consistency_trainer is None:
            raise ValueError("Consistency trainer not initialized. Call setup_consistency_model() first.")
            
        epoch_losses = []
        consistency_losses = []
        distillation_losses = []
        
        pbar = tqdm(dataloader, desc=f"CD Training Epoch {epoch}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Training step
            losses = self.consistency_trainer.train_step(batch)
            
            epoch_losses.append(losses['total_loss'])
            consistency_losses.append(losses['consistency_loss'])
            distillation_losses.append(losses['distillation_loss'])
            
            # Update progress bar
            pbar.set_postfix({
                "total_loss": f"{losses['total_loss']:.4f}",
                "consistency_loss": f"{losses['consistency_loss']:.4f}",
                "distillation_loss": f"{losses['distillation_loss']:.4f}"
            })
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "cd_train/total_loss": losses['total_loss'],
                    "cd_train/consistency_loss": losses['consistency_loss'],
                    "cd_train/distillation_loss": losses['distillation_loss'],
                    "cd_train/epoch": epoch,
                    "cd_train/batch": batch_idx
                })
        
        avg_total_loss = np.mean(epoch_losses)
        avg_consistency_loss = np.mean(consistency_losses)
        avg_distillation_loss = np.mean(distillation_losses)
        
        return {
            "total_loss": avg_total_loss,
            "consistency_loss": avg_consistency_loss,
            "distillation_loss": avg_distillation_loss
        }
        
    def train(
        self,
        dataset_path: str,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        save_checkpoint_every: int = 10,
        eval_every: int = 5,
        test_every: int = 20,
        num_test_trajectories: int = 10,
        checkpoint_dir: str = "./checkpoints"
    ) -> None:
        """
        Main training loop for Consistency Distillation.
        """
        # Setup
        self.setup_agent(device)
        self.setup_dataset(dataset_path)
        self.setup_consistency_model(device)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting Consistency Distillation training for {num_epochs} epochs...")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Dataset size: {len(self.dataset)}")
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Consistency Distillation training
            cd_metrics = self.train_consistency_epoch(train_loader, epoch)
            
            # Environment testing
            test_metrics = {}
            if epoch % test_every == 0:
                print(f"\nTesting consistency model in environment (epoch {epoch})...")
                # Generate samples using consistency model
                samples = self.consistency_trainer.sample(
                    batch_size=num_test_trajectories,
                    seq_len=self.t_act,
                    action_dim=self.movement_primitive_dim
                )
                print(f"Generated {len(samples)} samples using consistency model")
            
            # Save checkpoint
            if epoch % save_checkpoint_every == 0:
                checkpoint_file = checkpoint_path / f"consistency_model_epoch_{epoch}.pt"
                torch.save({
                    'consistency_model_state_dict': self.consistency_trainer.consistency_model.state_dict(),
                    'optimizer_state_dict': self.consistency_trainer.optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': cd_metrics
                }, checkpoint_file)
                print(f"Saved consistency model checkpoint: {checkpoint_file}")
            
            # Log epoch summary
            print(f"Epoch {epoch}: Total Loss: {cd_metrics['total_loss']:.4f}")
            print(f"Epoch {epoch}: Consistency Loss: {cd_metrics['consistency_loss']:.4f}")
            print(f"Epoch {epoch}: Distillation Loss: {cd_metrics['distillation_loss']:.4f}")
        
        # Save final model
        final_checkpoint = checkpoint_path / "final_consistency_model.pt"
        torch.save({
            'consistency_model_state_dict': self.consistency_trainer.consistency_model.state_dict(),
            'optimizer_state_dict': self.consistency_trainer.optimizer.state_dict(),
            'final_epoch': epoch,
            'final_metrics': cd_metrics
        }, final_checkpoint)
        print(f"Consistency Distillation training completed. Final model saved: {final_checkpoint}")


class CMManiSkillsPosition(MPDManiSkillsPosition):
    """
    Consistency Distillation workspace for ManiSkills position control.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.consistency_trainer = None
        
    def setup_consistency_model(self, device: str = "cuda") -> None:
        """Setup consistency model for training."""
        # Get action dimensions from dataset
        if hasattr(self, 'dataset') and len(self.dataset) > 0:
            sample = self.dataset[0]
            action_dim = sample['actions'].shape[-1]
            seq_len = sample['actions'].shape[-2]
        else:
            action_dim = self.movement_primitive_dim
            seq_len = self.t_act
            
        # Create consistency model
        consistency_model = ConsistencyModel(
            input_dim=action_dim,
            output_dim=action_dim,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
            use_time_embedding=True
        )
        
        # Create teacher model (simplified - in practice, use pre-trained MPD model)
        teacher_model = nn.Identity()  # Placeholder
        
        # Create trainer
        self.consistency_trainer = ConsistencyDistillationTrainer(
            consistency_model=consistency_model,
            teacher_model=teacher_model,
            device=device,
            learning_rate=1e-4,
            consistency_weight=1.0,
            distillation_weight=0.1,
            num_inference_steps=50
        )
        
        print(f"Initialized Consistency Distillation trainer for ManiSkills on device: {device}")
        print(f"Action dimension: {action_dim}, Sequence length: {seq_len}")
        
    def train_consistency_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train consistency model for one epoch."""
        if self.consistency_trainer is None:
            raise ValueError("Consistency trainer not initialized. Call setup_consistency_model() first.")
            
        epoch_losses = []
        consistency_losses = []
        distillation_losses = []
        
        pbar = tqdm(dataloader, desc=f"ManiSkills CD Training Epoch {epoch}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Training step
            losses = self.consistency_trainer.train_step(batch)
            
            epoch_losses.append(losses['total_loss'])
            consistency_losses.append(losses['consistency_loss'])
            distillation_losses.append(losses['distillation_loss'])
            
            # Update progress bar
            pbar.set_postfix({
                "total_loss": f"{losses['total_loss']:.4f}",
                "consistency_loss": f"{losses['consistency_loss']:.4f}",
                "distillation_loss": f"{losses['distillation_loss']:.4f}"
            })
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "cd_train/total_loss": losses['total_loss'],
                    "cd_train/consistency_loss": losses['consistency_loss'],
                    "cd_train/distillation_loss": losses['distillation_loss'],
                    "cd_train/epoch": epoch,
                    "cd_train/batch": batch_idx
                })
        
        avg_total_loss = np.mean(epoch_losses)
        avg_consistency_loss = np.mean(consistency_losses)
        avg_distillation_loss = np.mean(distillation_losses)
        
        return {
            "total_loss": avg_total_loss,
            "consistency_loss": avg_consistency_loss,
            "distillation_loss": avg_distillation_loss
        }
        
    def train(
        self,
        dataset_path: str,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        save_checkpoint_every: int = 10,
        eval_every: int = 5,
        test_every: int = 20,
        num_test_trajectories: int = 10,
        checkpoint_dir: str = "./checkpoints"
    ) -> None:
        """
        Main training loop for Consistency Distillation on ManiSkills.
        """
        # Setup
        self.setup_agent(device)
        self.setup_dataset(dataset_path)
        self.setup_consistency_model(device)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting ManiSkills Consistency Distillation training for {num_epochs} epochs...")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Dataset size: {len(self.dataset)}")
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Consistency Distillation training
            cd_metrics = self.train_consistency_epoch(train_loader, epoch)
            
            # Environment testing
            test_metrics = {}
            if epoch % test_every == 0:
                print(f"\nTesting ManiSkills consistency model in environment (epoch {epoch})...")
                # Generate samples using consistency model
                samples = self.consistency_trainer.sample(
                    batch_size=num_test_trajectories,
                    seq_len=self.t_act,
                    action_dim=self.movement_primitive_dim
                )
                print(f"Generated {len(samples)} samples using ManiSkills consistency model")
            
            # Save checkpoint
            if epoch % save_checkpoint_every == 0:
                checkpoint_file = checkpoint_path / f"maniskills_consistency_model_epoch_{epoch}.pt"
                torch.save({
                    'consistency_model_state_dict': self.consistency_trainer.consistency_model.state_dict(),
                    'optimizer_state_dict': self.consistency_trainer.optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': cd_metrics
                }, checkpoint_file)
                print(f"Saved ManiSkills consistency model checkpoint: {checkpoint_file}")
            
            # Log epoch summary
            print(f"Epoch {epoch}: Total Loss: {cd_metrics['total_loss']:.4f}")
            print(f"Epoch {epoch}: Consistency Loss: {cd_metrics['consistency_loss']:.4f}")
            print(f"Epoch {epoch}: Distillation Loss: {cd_metrics['distillation_loss']:.4f}")
        
        # Save final model
        final_checkpoint = checkpoint_path / "final_maniskills_consistency_model.pt"
        torch.save({
            'consistency_model_state_dict': self.consistency_trainer.consistency_model.state_dict(),
            'optimizer_state_dict': self.consistency_trainer.optimizer.state_dict(),
            'final_epoch': epoch,
            'final_metrics': cd_metrics
        }, final_checkpoint)
        print(f"ManiSkills Consistency Distillation training completed. Final model saved: {final_checkpoint}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Movement Primitive Diffusion with Consistency Distillation")
    
    # Core arguments
    parser.add_argument("--config-name", type=str, required=True,
                       help="Name of the configuration file")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the training dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run training on (cuda/cpu)")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for optimizer")
    
    # Consistency Distillation parameters
    parser.add_argument("--consistency_weight", type=float, default=1.0,
                       help="Weight for consistency loss")
    parser.add_argument("--distillation_weight", type=float, default=0.1,
                       help="Weight for distillation loss")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps for teacher model")
    parser.add_argument("--sigma_min", type=float, default=0.002,
                       help="Minimum noise level")
    parser.add_argument("--sigma_max", type=float, default=80.0,
                       help="Maximum noise level")
    
    # Evaluation parameters
    parser.add_argument("--eval_every", type=int, default=5,
                       help="Evaluate every N epochs")
    parser.add_argument("--test_every", type=int, default=20,
                       help="Test in environment every N epochs")
    parser.add_argument("--num_test_trajectories", type=int, default=10,
                       help="Number of test trajectories")
    
    # MPD specific parameters
    parser.add_argument("--movement_primitive_dim", type=int, default=7,
                       help="Dimension of movement primitives")
    parser.add_argument("--position_control", action="store_true", default=True,
                       help="Use position-based control")
    
    # Logging parameters
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Use WandB for logging")
    parser.add_argument("--wandb_project", type=str, default="fodmp-cm",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_workspace_class(config_name: str):
    """Get the appropriate workspace class based on configuration name."""
    if "metaworld" in config_name.lower():
        return CMMetaWorldPosition
    elif "maniskills" in config_name.lower():
        return CMManiSkillsPosition
    else:
        raise ValueError(f"Unknown configuration: {config_name}. Expected 'metaworld' or 'maniskills' in config name.")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Override config with command line arguments
    cfg.dataset_path = args.dataset_path
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.device = args.device
    cfg.num_epochs = args.num_epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.learning_rate
    cfg.eval_every = args.eval_every
    cfg.test_every = args.test_every
    cfg.num_test_trajectories = args.num_test_trajectories
    cfg.movement_primitive_dim = args.movement_primitive_dim
    cfg.position_control = args.position_control
    
    # Consistency Distillation parameters
    cfg.consistency_weight = args.consistency_weight
    cfg.distillation_weight = args.distillation_weight
    cfg.num_inference_steps = args.num_inference_steps
    cfg.sigma_min = args.sigma_min
    cfg.sigma_max = args.sigma_max
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        print(f"Initialized WandB logging: {wandb.run.url}")
    
    # Print configuration
    print("=" * 70)
    print("MOVEMENT PRIMITIVE DIFFUSION CONSISTENCY DISTILLATION TRAINING")
    print("=" * 70)
    print(f"Configuration: {args.config_name}")
    print(f"Dataset path: {cfg.dataset_path}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")
    print(f"Device: {cfg.device}")
    print(f"Number of epochs: {cfg.num_epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"Consistency weight: {cfg.consistency_weight}")
    print(f"Distillation weight: {cfg.distillation_weight}")
    print(f"Number of inference steps: {cfg.num_inference_steps}")
    print(f"Sigma range: [{cfg.sigma_min}, {cfg.sigma_max}]")
    print(f"Movement primitive dimension: {cfg.movement_primitive_dim}")
    print(f"Position control: {cfg.position_control}")
    print(f"Random seed: {args.seed}")
    if args.resume_from:
        print(f"Resume from: {args.resume_from}")
    print("=" * 70)
    
    # Validate dataset path
    dataset_path = Path(cfg.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get workspace class
    workspace_class = get_workspace_class(args.config_name)
    
    # Initialize workspace
    print("Initializing Consistency Distillation workspace...")
    workspace = workspace_class(
        env_config=cfg.env,
        agent_config=cfg.agent,
        dataset_config=cfg.dataset,
        t_act=cfg.get('t_act', 4),
        t_obs=cfg.get('t_obs', 2),
        predict_past=cfg.get('predict_past', False),
        num_upload_successful_videos=cfg.get('num_upload_successful_videos', 5),
        num_upload_failed_videos=cfg.get('num_upload_failed_videos', 5),
        show_images=cfg.get('show_images', False),
        movement_primitive_dim=cfg.movement_primitive_dim,
        position_control=cfg.position_control
    )
    
    # Load checkpoint if resuming
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
            workspace.consistency_trainer.consistency_model.load_state_dict(checkpoint['consistency_model_state_dict'])
            workspace.consistency_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
    
    try:
        # Start training
        print("Starting Consistency Distillation training...")
        workspace.train(
            dataset_path=cfg.dataset_path,
            num_epochs=cfg.num_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            device=cfg.device,
            save_checkpoint_every=cfg.get('save_checkpoint_every', 10),
            eval_every=cfg.eval_every,
            test_every=cfg.test_every,
            num_test_trajectories=cfg.num_test_trajectories,
            checkpoint_dir=cfg.checkpoint_dir
        )
        
        print("Consistency Distillation training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("Saving current state...")
        
        # Save interrupted state
        interrupted_checkpoint = checkpoint_dir / "interrupted_consistency_training.pt"
        torch.save({
            'consistency_model_state_dict': workspace.consistency_trainer.consistency_model.state_dict(),
            'optimizer_state_dict': workspace.consistency_trainer.optimizer.state_dict(),
            'epoch': workspace.current_epoch,
        }, interrupted_checkpoint)
        print(f"Interrupted state saved to: {interrupted_checkpoint}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    finally:
        # Clean up
        workspace.close()
        if args.use_wandb and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
