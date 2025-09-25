#!/usr/bin/env python3
"""
Training script for Diffusion Policy on MetaWorld environments.

This script provides a comprehensive training pipeline for diffusion policy
using CNN-based observation encoding for robotic manipulation tasks.

Usage:
    python train_dp.py --config-name=metaworld_dp_cnn --dataset_path=/path/to/dataset
"""

import argparse
import hydra
import wandb
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from FODMP.workspace.dp_cnn_metaworld_workspace import DPCNNMetaWorldWorkspace


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Diffusion Policy on MetaWorld")
    
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
    
    # Evaluation parameters
    parser.add_argument("--eval_every", type=int, default=5,
                       help="Evaluate every N epochs")
    parser.add_argument("--test_every", type=int, default=20,
                       help="Test in environment every N epochs")
    parser.add_argument("--num_test_trajectories", type=int, default=10,
                       help="Number of test trajectories")
    
    # Logging parameters
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="metaworld-dp-cnn",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    
    return parser.parse_args()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    args = parse_args()
    
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
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        print(f"Initialized WandB logging: {wandb.run.url}")
    
    # Print configuration
    print("=" * 50)
    print("DIFFUSION POLICY TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Dataset path: {cfg.dataset_path}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")
    print(f"Device: {cfg.device}")
    print(f"Number of epochs: {cfg.num_epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"Evaluation every: {cfg.eval_every} epochs")
    print(f"Testing every: {cfg.test_every} epochs")
    print(f"Number of test trajectories: {cfg.num_test_trajectories}")
    print("=" * 50)
    
    # Validate dataset path
    dataset_path = Path(cfg.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize workspace
    print("Initializing workspace...")
    workspace = DPCNNMetaWorldWorkspace(
        env_config=cfg.env,
        agent_config=cfg.agent,
        dataset_config=cfg.dataset,
        t_act=cfg.t_act,
        t_obs=cfg.t_obs,
        predict_past=cfg.predict_past,
        num_upload_successful_videos=cfg.num_upload_successful_videos,
        num_upload_failed_videos=cfg.num_upload_failed_videos,
        show_images=cfg.show_images
    )
    
    try:
        # Resume from checkpoint if specified
        if args.resume_from:
            print(f"Resuming training from: {args.resume_from}")
            workspace.load_checkpoint(args.resume_from)
        
        # Start training
        print("Starting training...")
        workspace.train(
            dataset_path=str(dataset_path),
            num_epochs=cfg.num_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            device=cfg.device,
            save_checkpoint_every=cfg.save_checkpoint_every,
            eval_every=cfg.eval_every,
            test_every=cfg.test_every,
            num_test_trajectories=cfg.num_test_trajectories,
            checkpoint_dir=str(checkpoint_dir)
        )
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save current state
        interrupt_checkpoint = checkpoint_dir / "interrupted_checkpoint.pt"
        workspace.agent.save_model(interrupt_checkpoint, save_optimizer=True, save_lr_scheduler=True)
        print(f"Saved interrupted state to: {interrupt_checkpoint}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
        
    finally:
        # Clean up
        workspace.close()
        if args.use_wandb:
            wandb.finish()


def train_with_config(
    config_path: str,
    dataset_path: str,
    checkpoint_dir: str = "./checkpoints",
    device: str = "cuda",
    use_wandb: bool = False,
    wandb_project: str = "metaworld-dp-cnn",
    wandb_run_name: Optional[str] = None,
    **kwargs
) -> DPCNNMetaWorldWorkspace:
    """
    Train diffusion policy with a configuration file.
    
    Args:
        config_path: Path to the configuration file
        dataset_path: Path to the training dataset
        checkpoint_dir: Directory to save checkpoints
        device: Device to run training on
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: WandB project name
        wandb_run_name: WandB run name
        **kwargs: Additional training parameters
        
    Returns:
        Trained workspace
    """
    # Load configuration
    cfg = OmegaConf.load(config_path)
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize workspace
    workspace = DPCNNMetaWorldWorkspace(
        env_config=cfg.env,
        agent_config=cfg.agent,
        dataset_config=cfg.dataset,
        t_act=cfg.get("t_act", 4),
        t_obs=cfg.get("t_obs", 2),
        predict_past=cfg.get("predict_past", False),
        num_upload_successful_videos=cfg.get("num_upload_successful_videos", 5),
        num_upload_failed_videos=cfg.get("num_upload_failed_videos", 5),
        show_images=cfg.get("show_images", False)
    )
    
    # Start training
    workspace.train(
        dataset_path=dataset_path,
        num_epochs=cfg.get("num_epochs", 100),
        batch_size=cfg.get("batch_size", 32),
        learning_rate=cfg.get("learning_rate", 1e-4),
        device=device,
        save_checkpoint_every=cfg.get("save_checkpoint_every", 10),
        eval_every=cfg.get("eval_every", 5),
        test_every=cfg.get("test_every", 20),
        num_test_trajectories=cfg.get("num_test_trajectories", 10),
        checkpoint_dir=str(checkpoint_path)
    )
    
    if use_wandb:
        wandb.finish()
    
    return workspace


if __name__ == "__main__":
    main()
