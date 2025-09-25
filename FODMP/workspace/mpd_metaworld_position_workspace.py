import wandb
import hydra
import numpy as np
import time
import torch
import cv2
from pathlib import Path
from omegaconf import DictConfig
from typing import Dict, List, Optional
from tqdm import tqdm

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.utils.video import save_video_from_array
from FODMP.workspace.base_workspace import BaseWorkspace


class MPDMetaWorldPosition(BaseWorkspace):
    """
    Movement Primitive Diffusion workspace for MetaWorld environments with position control.
    
    This workspace implements movement primitive diffusion for MetaWorld robotic manipulation tasks
    using position-based control. It extends the base workspace with MPD-specific training and 
    evaluation logic for MetaWorld environments.
    """
    
    def __init__(
        self,
        env_config: DictConfig,
        agent_config: DictConfig,
        dataset_config: DictConfig,
        t_act: int = 4,
        t_obs: int = 2,
        predict_past: bool = False,
        num_upload_successful_videos: int = 5,
        num_upload_failed_videos: int = 5,
        show_images: bool = False,
        **kwargs
    ):
        super().__init__(
            env_config=env_config,
            t_act=t_act,
            num_upload_successful_videos=num_upload_successful_videos,
            num_upload_failed_videos=num_upload_failed_videos,
            show_images=show_images,
        )
        
        # Store configurations
        self.agent_config = agent_config
        self.dataset_config = dataset_config
        self.t_obs = t_obs
        self.predict_past = predict_past
        
        # Initialize agent
        self.agent: Optional[BaseAgent] = None
        
        # Training state
        self.current_epoch = 0
        self.best_success_rate = 0.0
        self.training_losses = []
        self.evaluation_metrics = []
        
        # MPD specific parameters
        self.position_control = True
        self.movement_primitive_dim = kwargs.get('movement_primitive_dim', 7)  # Default 7-DOF
        
    def setup_agent(self, device: str = "cuda") -> None:
        """Initialize the movement primitive diffusion agent for MetaWorld position control."""
        # Update agent config with environment-specific parameters
        agent_config = self.agent_config.copy()
        agent_config.device = device
        agent_config.t_obs = self.t_obs
        agent_config.predict_past = self.predict_past
        
        # Add MPD-specific configuration
        agent_config.position_control = self.position_control
        agent_config.movement_primitive_dim = self.movement_primitive_dim
        
        # Instantiate the agent
        self.agent = hydra.utils.instantiate(agent_config)
        
        print(f"Initialized MPD agent for MetaWorld position control on device: {device}")
        print(f"Agent type: {type(self.agent).__name__}")
        print(f"Movement primitive dimension: {self.movement_primitive_dim}")
        
    def setup_dataset(self, dataset_path: str) -> None:
        """Setup the training dataset for movement primitive diffusion."""
        # Update dataset config with path
        dataset_config = self.dataset_config.copy()
        dataset_config.dataset_path = dataset_path
        dataset_config.position_control = self.position_control
        dataset_config.movement_primitive_dim = self.movement_primitive_dim
        
        # Instantiate dataset
        self.dataset = hydra.utils.instantiate(dataset_config)
        
        print(f"Loaded MPD MetaWorld dataset from: {dataset_path}")
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Position control mode: {self.position_control}")
        
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train the MPD agent for one epoch."""
        if self.agent is None:
            raise ValueError("Agent not initialized. Call setup_agent() first.")
            
        self.agent.model.train()
        self.agent.encoder.train()
        
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"MPD MetaWorld Training Epoch {epoch}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Training step
            loss = self.agent.train_step(batch)
            epoch_losses.append(loss)
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss:.4f}"})
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "train/loss": loss,
                    "train/epoch": epoch,
                    "train/batch": batch_idx
                })
        
        avg_loss = np.mean(epoch_losses)
        self.training_losses.append(avg_loss)
        
        return {"train_loss": avg_loss}
        
    def evaluate_agent(self, dataloader, epoch: int) -> Dict[str, float]:
        """Evaluate the MPD agent on validation data."""
        if self.agent is None:
            raise ValueError("Agent not initialized. Call setup_agent() first.")
            
        self.agent.model.eval()
        self.agent.encoder.eval()
        
        eval_losses = []
        start_deviations = []
        end_deviations = []
        
        pbar = tqdm(dataloader, desc=f"MPD MetaWorld Evaluating Epoch {epoch}", leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                eval_loss, start_dev, end_dev = self.agent.evaluate(batch)
                eval_losses.append(eval_loss)
                start_deviations.append(start_dev)
                end_deviations.append(end_dev)
                
                pbar.set_postfix({
                    "eval_loss": f"{eval_loss:.4f}",
                    "start_dev": f"{start_dev:.4f}",
                    "end_dev": f"{end_dev:.4f}"
                })
        
        avg_eval_loss = np.mean(eval_losses)
        avg_start_dev = np.mean(start_deviations)
        avg_end_dev = np.mean(end_deviations)
        
        metrics = {
            "eval_loss": avg_eval_loss,
            "start_deviation": avg_start_dev,
            "end_deviation": avg_end_dev
        }
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "eval/loss": avg_eval_loss,
                "eval/start_deviation": avg_start_dev,
                "eval/end_deviation": avg_end_dev,
                "eval/epoch": epoch
            })
        
        return metrics
        
    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        """Test the MPD agent in the MetaWorld environment."""
        # Store the current agent
        original_agent = self.agent
        self.agent = agent
        
        try:
            # Call parent test_agent method
            results = super().test_agent(agent, num_trajectories)
            
            # Add MPD specific metrics
            results.update({
                "current_epoch": self.current_epoch,
                "best_success_rate": self.best_success_rate,
                "avg_training_loss": np.mean(self.training_losses) if self.training_losses else 0.0,
                "movement_primitive_dim": self.movement_primitive_dim,
                "position_control": self.position_control
            })
            
            return results
            
        finally:
            # Restore original agent
            self.agent = original_agent
            
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
        Main training loop for movement primitive diffusion on MetaWorld.
        
        Args:
            dataset_path: Path to the training dataset
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            device: Device to run training on
            save_checkpoint_every: Save checkpoint every N epochs
            eval_every: Evaluate on validation set every N epochs
            test_every: Test in environment every N epochs
            num_test_trajectories: Number of test trajectories
            checkpoint_dir: Directory to save checkpoints
        """
        # Setup
        self.setup_agent(device)
        self.setup_dataset(dataset_path)
        
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
        
        print(f"Starting MPD MetaWorld position training for {num_epochs} epochs...")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Movement primitive dimension: {self.movement_primitive_dim}")
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluation
            eval_metrics = {}
            if epoch % eval_every == 0:
                eval_metrics = self.evaluate_agent(train_loader, epoch)
                
            # Environment testing
            test_metrics = {}
            if epoch % test_every == 0:
                print(f"\nTesting MPD agent in MetaWorld environment (epoch {epoch})...")
                test_metrics = self.test_agent(self.agent, num_test_trajectories)
                
                # Update best success rate
                current_success_rate = test_metrics.get("success_rate", 0.0)
                if current_success_rate > self.best_success_rate:
                    self.best_success_rate = current_success_rate
                    
                print(f"MPD MetaWorld Success rate: {current_success_rate:.3f}")
                print(f"Best success rate: {self.best_success_rate:.3f}")
            
            # Save checkpoint
            if epoch % save_checkpoint_every == 0:
                checkpoint_file = checkpoint_path / f"mpd_metaworld_position_checkpoint_epoch_{epoch}.pt"
                self.agent.save_model(checkpoint_file, save_optimizer=True, save_lr_scheduler=True)
                print(f"Saved MPD MetaWorld checkpoint: {checkpoint_file}")
            
            # Log epoch summary
            print(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}")
            if eval_metrics:
                print(f"Epoch {epoch}: Eval Loss: {eval_metrics['eval_loss']:.4f}")
            if test_metrics:
                print(f"Epoch {epoch}: MPD MetaWorld Success Rate: {test_metrics['success_rate']:.3f}")
        
        # Save final model
        final_checkpoint = checkpoint_path / "final_mpd_metaworld_position_model.pt"
        self.agent.save_model(final_checkpoint, save_optimizer=True, save_lr_scheduler=True)
        print(f"MPD MetaWorld training completed. Final position model saved: {final_checkpoint}")
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a trained model checkpoint."""
        if self.agent is None:
            raise ValueError("Agent not initialized. Call setup_agent() first.")
            
        self.agent.load_pretrained(checkpoint_path)
        print(f"Loaded MPD MetaWorld position checkpoint from: {checkpoint_path}")
        
    def reset_env(self, caller_locals: Dict) -> np.ndarray:
        """Reset the MetaWorld environment."""
        # Reset with random task if available
        if hasattr(self.env, 'reset_task'):
            return self.env.reset_task()
        else:
            return self.env.reset()
            
    def check_success_hook(self, caller_locals: Dict) -> bool:
        """Check if the MetaWorld task was completed successfully."""
        # MetaWorld environments typically have a success indicator
        info = caller_locals.get("info", {})
        return info.get("success", False) or caller_locals.get("terminated", False)
        
    def render_function(self, caller_locals: Dict) -> np.ndarray:
        """Render the MetaWorld environment."""
        # MetaWorld uses different rendering modes
        try:
            return self.env.render(mode="rgb_array")
        except:
            # Fallback to internal rendering
            return self.env._render_frame(mode="rgb_array")
            
    def post_step_hook(self, caller_locals: Dict) -> None:
        """Post-step hook for additional logging."""
        # Can be used to log additional metrics during evaluation
        pass
        
    def post_episode_hook(self, caller_locals: Dict) -> None:
        """Post-episode hook for additional processing."""
        # Can be used to log episode-specific metrics
        pass
        
    def get_result_dict(self, caller_locals: Dict) -> Dict[str, float]:
        """Get results dictionary with additional MPD metrics."""
        base_results = super().get_result_dict(caller_locals)
        
        # Add MPD specific metrics
        base_results.update({
            "current_epoch": self.current_epoch,
            "best_success_rate": self.best_success_rate,
            "avg_training_loss": np.mean(self.training_losses) if self.training_losses else 0.0,
            "movement_primitive_dim": self.movement_primitive_dim,
            "position_control": float(self.position_control)
        })
        
        return base_results
        
    def get_result_dict_keys(self) -> List[str]:
        """Get keys for the result dictionary."""
        base_keys = super().get_result_dict_keys()
        return base_keys + ["current_epoch", "best_success_rate", "avg_training_loss", 
                           "movement_primitive_dim", "position_control"]
        
    def close(self) -> None:
        """Clean up resources."""
        if self.agent is not None:
            # Clean up agent resources if needed
            pass
        super().close()
