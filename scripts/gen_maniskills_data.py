#!/usr/bin/env python3
"""
ManiSkills Dataset Generation Script

This script generates demonstration datasets for ManiSkills environments
for training Movement Primitive Diffusion and Diffusion Policy models.

Usage:
    python gen_maniskills_data.py --env_name=PickCube-v1 --num_demos=100 --output_dir=./data
"""

import argparse
import gymnasium as gym
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from tqdm import tqdm
import random
import torch

# ManiSkills imports
import mani_skill
# from mani_skill.envs.sapien_env import SapienEnv
# from mani_skill.utils.wrappers import RecordEpisode


class ManiSkillsDataGenerator:
    """
    Data generator for ManiSkills environments.
    
    This class generates demonstration trajectories for ManiSkills tasks
    using various collection strategies including random policies, 
    scripted policies, and human demonstrations.
    """
    
    def __init__(
        self,
        env_name: str,
        num_demos: int = 100,
        max_episode_steps: int = 200,
        obs_mode: str = "rgb",
        render_mode: str = "rgb_array",
        camera_name: str = "front",
        image_size: tuple = (224, 224),
        seed: int = 42
    ):
        """
        Initialize ManiSkills data generator.
        
        Args:
            env_name: Name of the ManiSkills environment
            num_demos: Number of demonstrations to generate
            max_episode_steps: Maximum steps per episode
            obs_mode: Observation mode ("state", "state_dict", "rgb", etc.)
            render_mode: Rendering mode for observations
            camera_name: Camera name for rendering
            image_size: Image size for rendering
            seed: Random seed for reproducibility
        """
        self.env_name = env_name
        self.num_demos = num_demos
        self.max_episode_steps = max_episode_steps
        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.image_size = image_size
        self.seed = seed
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize environment
        self.env = self._create_environment()
        
        # Data storage
        self.trajectories = []
        self.metadata = {
            'env_name': env_name,
            'num_demos': num_demos,
            'max_episode_steps': max_episode_steps,
            'obs_mode': obs_mode,
            'render_mode': render_mode,
            'camera_name': camera_name,
            'image_size': image_size,
            'seed': seed,
            'generation_time': None,
            'success_rate': 0.0
        }
        
    def _create_environment(self) -> gym.Env:
        """Create ManiSkills environment."""
        try:
            # Create ManiSkills environment
            env = gym.make(
                self.env_name,
                obs_mode=self.obs_mode,
                render_mode=self.render_mode
                # camera_cfgs={
                #     "width": self.image_size[0],
                #     "height": self.image_size[1],
                #     "camera_name": self.camera_name
                # }
            )
            
            # Set max episode steps
            env.max_episode_steps = self.max_episode_steps
            
            print(f"Created ManiSkills environment: {self.env_name}")
            print(f"Action space: {env.action_space}")
            print(f"Observation space: {env.observation_space}")
            print(f"Observation mode: {self.obs_mode}")
            
            return env
            
        except Exception as e:
            raise RuntimeError(f"Failed to create ManiSkills environment {self.env_name}: {e}")
    
    def _get_random_action(self) -> np.ndarray:
        """Generate random action."""
        return self.env.action_space.sample()
    
    def _get_scripted_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Generate scripted action based on observation.
        
        This is a simple scripted policy that can be extended for specific tasks.
        """
        # Extract robot and object information from observation
        if self.obs_mode == "state":
            # For state mode, obs is a flat array
            obs_array = obs
            robot_pos = obs_array[:3] if len(obs_array) >= 3 else np.zeros(3)
            object_pos = obs_array[3:6] if len(obs_array) >= 6 else np.zeros(3)
            goal_pos = obs_array[6:9] if len(obs_array) >= 9 else np.zeros(3)
            
        elif self.obs_mode == "state_dict":
            # For state_dict mode, obs is a dictionary
            robot_pos = obs.get('agent', {}).get('qpos', np.zeros(7))[:3]
            object_pos = obs.get('extra', {}).get('object_pos', np.zeros(3))
            goal_pos = obs.get('extra', {}).get('goal_pos', np.zeros(3))
            
        else:
            # For other modes, use default values
            robot_pos = np.zeros(3)
            object_pos = np.zeros(3)
            goal_pos = np.zeros(3)
        
        # Simple reaching policy
        if 'PickCube' in self.env_name:
            # PickCube task: move towards object, then towards goal
            if np.linalg.norm(object_pos - robot_pos) > 0.1:
                # Move towards object
                direction = object_pos - robot_pos
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                action = direction * 0.1
            else:
                # Move towards goal
                direction = goal_pos - robot_pos
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                action = direction * 0.1
                
        elif 'StackCube' in self.env_name:
            # StackCube task: stack cubes
            if np.linalg.norm(object_pos - robot_pos) > 0.1:
                direction = object_pos - robot_pos
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                action = direction * 0.1
            else:
                # Move up and towards goal
                direction = goal_pos - robot_pos
                direction[2] += 0.1  # Move up slightly
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                action = direction * 0.1
                
        elif 'OpenCabinetDoor' in self.env_name:
            # OpenCabinetDoor task: open cabinet doors
            if np.linalg.norm(object_pos - robot_pos) > 0.1:
                direction = object_pos - robot_pos
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                action = direction * 0.1
            else:
                # Pull action
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])  # Pull action
                
        else:
            # Default to random action
            action = self._get_random_action()
            
        # Clip action to valid range
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action
    
    def _is_success(self, info: Dict[str, Any]) -> bool:
        """Check if the episode was successful."""
        return info.get('success', False) or info.get('is_success', False)
    
    def _collect_trajectory(self, policy_type: str = "random") -> Dict[str, Any]:
        """
        Collect a single trajectory.
        
        Args:
            policy_type: Type of policy to use ("random", "scripted")
            
        Returns:
            Dictionary containing trajectory data
        """
        obs = self.env.reset()
        done = False
        step_count = 0
        
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminated': [],
            'truncated': [],
            'infos': [],
            'success': False,
            'episode_length': 0
        }
        
        while not done and step_count < self.max_episode_steps:
            # Get action based on policy type
            if policy_type == "random":
                action = self._get_random_action()
            elif policy_type == "scripted":
                action = self._get_scripted_action(obs)
            else:
                raise ValueError(f"Unknown policy type: {policy_type}")
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Store data
            trajectory['observations'].append(obs)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['terminated'].append(terminated)
            trajectory['truncated'].append(truncated)
            trajectory['infos'].append(info)
            
            # Update state
            obs = next_obs
            done = terminated or truncated
            step_count += 1
            
            # Check for success
            if self._is_success(info):
                trajectory['success'] = True
        
        trajectory['episode_length'] = step_count
        return trajectory
    
    def generate_dataset(
        self, 
        policy_type: str = "random",
        success_rate_threshold: float = 0.0,
        max_attempts: int = None
    ) -> Dict[str, Any]:
        """
        Generate dataset with specified number of demonstrations.
        
        Args:
            policy_type: Type of policy to use ("random", "scripted")
            success_rate_threshold: Minimum success rate threshold
            max_attempts: Maximum number of attempts (None for unlimited)
            
        Returns:
            Dictionary containing dataset and metadata
        """
        print(f"Generating {self.num_demos} demonstrations using {policy_type} policy...")
        
        start_time = time.time()
        successful_trajectories = 0
        attempts = 0
        
        if max_attempts is None:
            max_attempts = self.num_demos * 10  # Reasonable upper bound
        
        pbar = tqdm(total=self.num_demos, desc="Generating demonstrations")
        
        while len(self.trajectories) < self.num_demos and attempts < max_attempts:
            attempts += 1
            
            # Collect trajectory
            trajectory = self._collect_trajectory(policy_type)
            
            # Check if trajectory meets criteria
            if trajectory['success'] or len(self.trajectories) < self.num_demos * success_rate_threshold:
                self.trajectories.append(trajectory)
                if trajectory['success']:
                    successful_trajectories += 1
                pbar.update(1)
            
            # Update progress
            current_success_rate = successful_trajectories / len(self.trajectories) if self.trajectories else 0
            pbar.set_postfix({
                'success_rate': f"{current_success_rate:.3f}",
                'attempts': attempts
            })
        
        pbar.close()
        
        # Update metadata
        self.metadata['generation_time'] = time.time() - start_time
        self.metadata['success_rate'] = successful_trajectories / len(self.trajectories) if self.trajectories else 0
        self.metadata['total_attempts'] = attempts
        self.metadata['policy_type'] = policy_type
        
        print(f"Dataset generation completed!")
        print(f"Total trajectories: {len(self.trajectories)}")
        print(f"Successful trajectories: {successful_trajectories}")
        print(f"Success rate: {self.metadata['success_rate']:.3f}")
        print(f"Total attempts: {attempts}")
        print(f"Generation time: {self.metadata['generation_time']:.2f} seconds")
        
        return {
            'trajectories': self.trajectories,
            'metadata': self.metadata
        }
    
    def save_dataset(self, output_path: str) -> None:
        """
        Save dataset to HDF5 file with JSON metadata.
        
        Args:
            output_path: Path to save the dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        h5_path = output_path.with_suffix('.h5')
        json_path = output_path.with_suffix('.json')
        
        with h5py.File(h5_path, 'w') as f:
            # Create trajectories group
            traj_group = f.create_group('trajectories')
            
            for i, traj in enumerate(self.trajectories):
                traj_subgroup = traj_group.create_group(f'traj_{i}')
                
                # Convert lists to numpy arrays
                traj_subgroup.create_dataset('observations', data=np.array(traj['observations']))
                traj_subgroup.create_dataset('actions', data=np.array(traj['actions']))
                traj_subgroup.create_dataset('rewards', data=np.array(traj['rewards']))
                traj_subgroup.create_dataset('terminated', data=np.array(traj['terminated']))
                traj_subgroup.create_dataset('truncated', data=np.array(traj['truncated']))
                traj_subgroup.create_dataset('infos', data=np.array(traj['infos'], dtype=object))
                traj_subgroup.attrs['success'] = traj['success']
                traj_subgroup.attrs['episode_length'] = traj['episode_length']
            
            # Add metadata
            metadata_group = f.create_group('metadata')
            for key, value in self.metadata.items():
                metadata_group.attrs[key] = value
        
        # Save JSON metadata
        with open(json_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Dataset saved to: {h5_path}")
        print(f"Metadata saved to: {json_path}")
    
    def close(self) -> None:
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


def main():
    """Main function for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate ManiSkills demonstration dataset")
    
    # Environment parameters
    parser.add_argument("--env_name", type=str, default="PickCube-v1",
                       help="ManiSkills environment name (e.g., PickCube-v1)")
    parser.add_argument("--num_demos", type=int, default=100,
                       help="Number of demonstrations to generate")
    parser.add_argument("--max_episode_steps", type=int, default=200,
                       help="Maximum steps per episode")
    
    # Observation parameters
    parser.add_argument("--obs_mode", type=str, default="rgb",
                       choices=["state", "state_dict", "rgb", "depth", "rgb+depth"],
                       help="Observation mode")
    parser.add_argument("--render_mode", type=str, default="rgb_array",
                       help="Rendering mode")
    parser.add_argument("--camera_name", type=str, default="front",
                       help="Camera name for rendering")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224],
                       help="Image size for rendering")
    
    # Generation parameters
    parser.add_argument("--policy_type", type=str, default="scripted",
                       choices=["random", "scripted"],
                       help="Type of policy to use")
    parser.add_argument("--success_rate_threshold", type=float, default=0.3,
                       help="Minimum success rate threshold")
    parser.add_argument("--max_attempts", type=int, default=None,
                       help="Maximum number of attempts")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="../data",
                       help="Output directory for dataset")
    parser.add_argument("--output_name", type=str, default=None,
                       help="Output filename (default: env_name_demos)")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create output filename
    if args.output_name is None:
        args.output_name = f"{args.env_name}_demos"
    
    output_path = Path(args.output_dir) / args.output_name
    
    # Print configuration
    print("=" * 60)
    print("MANISKILLS DATASET GENERATION")
    print("=" * 60)
    print(f"Environment: {args.env_name}")
    print(f"Number of demos: {args.num_demos}")
    print(f"Max episode steps: {args.max_episode_steps}")
    print(f"Observation mode: {args.obs_mode}")
    print(f"Policy type: {args.policy_type}")
    print(f"Success rate threshold: {args.success_rate_threshold}")
    print(f"Output path: {output_path}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    try:
        # Create data generator
        generator = ManiSkillsDataGenerator(
            env_name=args.env_name,
            num_demos=args.num_demos,
            max_episode_steps=args.max_episode_steps,
            obs_mode=args.obs_mode,
            render_mode=args.render_mode,
            camera_name=args.camera_name,
            image_size=tuple(args.image_size),
            seed=args.seed
        )
        
        # Generate dataset
        dataset = generator.generate_dataset(
            policy_type=args.policy_type,
            success_rate_threshold=args.success_rate_threshold,
            max_attempts=args.max_attempts
        )
        
        # Save dataset
        generator.save_dataset(output_path)
        
        print("Dataset generation completed successfully!")
        
    except Exception as e:
        print(f"Dataset generation failed: {e}")
        raise
    
    finally:
        # Clean up
        if 'generator' in locals():
            generator.close()


if __name__ == "__main__":
    main()
