#!/usr/bin/env python3
"""
MetaWorld Dataset Generation Script

This script generates demonstration datasets for MetaWorld environments
for training Movement Primitive Diffusion and Diffusion Policy models.

Usage:
    python gen_metaworld_data.py --env_name=reach-v1 --num_demos=100 --output_dir=./data
"""

import argparse
import gymnasium as gym
import numpy as np
import pickle
import h5py
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from tqdm import tqdm
import random
import torch

# MetaWorld imports
import metaworld
from metaworld.envs import ML1Env


class MetaWorldDataGenerator:
    """
    Data generator for MetaWorld environments.
    
    This class generates demonstration trajectories for MetaWorld tasks
    using various collection strategies including random policies, 
    scripted policies, and human demonstrations.
    """
    
    def __init__(
        self,
        env_name: str,
        num_demos: int = 100,
        max_episode_steps: int = 500,
        render_mode: str = "rgb_array",
        camera_name: str = "corner",
        image_size: tuple = (224, 224),
        seed: int = 42
    ):
        """
        Initialize MetaWorld data generator.
        
        Args:
            env_name: Name of the MetaWorld environment
            num_demos: Number of demonstrations to generate
            max_episode_steps: Maximum steps per episode
            render_mode: Rendering mode for observations
            camera_name: Camera name for rendering
            image_size: Image size for rendering
            seed: Random seed for reproducibility
        """
        self.env_name = env_name
        self.num_demos = num_demos
        self.max_episode_steps = max_episode_steps
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
            'render_mode': render_mode,
            'camera_name': camera_name,
            'image_size': image_size,
            'seed': seed,
            'generation_time': None,
            'success_rate': 0.0
        }
        
    def _create_environment(self) -> gym.Env:
        """Create MetaWorld environment."""
        try:
            # Create ML1 environment
            env = ML1Env(env_name=self.env_name)
            env.max_path_length = self.max_episode_steps
            
            # Set rendering parameters
            if hasattr(env, 'render_mode'):
                env.render_mode = self.render_mode
            if hasattr(env, 'camera_name'):
                env.camera_name = self.camera_name
            if hasattr(env, 'image_size'):
                env.image_size = self.image_size
                
            print(f"Created MetaWorld environment: {self.env_name}")
            print(f"Action space: {env.action_space}")
            print(f"Observation space: {env.observation_space}")
            
            return env
            
        except Exception as e:
            raise RuntimeError(f"Failed to create MetaWorld environment {self.env_name}: {e}")
    
    def _get_random_action(self) -> np.ndarray:
        """Generate random action."""
        return self.env.action_space.sample()
    
    def _get_scripted_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Generate scripted action based on observation.
        
        This is a simple scripted policy that can be extended for specific tasks.
        """
        # Get robot and object positions
        robot_pos = obs.get('robot_pos', np.zeros(3))
        object_pos = obs.get('object_pos', np.zeros(3))
        goal_pos = obs.get('goal_pos', np.zeros(3))
        
        # Simple reaching policy
        if 'reach' in self.env_name.lower():
            # Move towards the goal
            direction = goal_pos - robot_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            # Scale action
            action = direction * 0.1
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
        elif 'pick' in self.env_name.lower():
            # Pick and place policy
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
                
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
        else:
            # Default to random action
            action = self._get_random_action()
            
        return action
    
    def _is_success(self, info: Dict[str, Any]) -> bool:
        """Check if the episode was successful."""
        return info.get('success', False)
    
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
            'dones': [],
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
            trajectory['dones'].append(terminated or truncated)
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
    
    def save_dataset(self, output_path: str, format: str = "pickle") -> None:
        """
        Save dataset to file.
        
        Args:
            output_path: Path to save the dataset
            format: Format to save ("pickle", "h5", "npz")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "pickle":
            self._save_pickle(output_path)
        elif format == "h5":
            self._save_h5(output_path)
        elif format == "npz":
            self._save_npz(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Dataset saved to: {output_path}")
    
    def _save_pickle(self, output_path: Path) -> None:
        """Save dataset in pickle format."""
        data = {
            'trajectories': self.trajectories,
            'metadata': self.metadata
        }
        
        with open(output_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(data, f)
    
    def _save_h5(self, output_path: Path) -> None:
        """Save dataset in HDF5 format."""
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
                traj_subgroup.create_dataset('dones', data=np.array(traj['dones']))
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
    
    def _save_npz(self, output_path: Path) -> None:
        """Save dataset in NPZ format."""
        # Convert trajectories to numpy arrays
        observations = [np.array(traj['observations']) for traj in self.trajectories]
        actions = [np.array(traj['actions']) for traj in self.trajectories]
        rewards = [np.array(traj['rewards']) for traj in self.trajectories]
        dones = [np.array(traj['dones']) for traj in self.trajectories]
        infos = [np.array(traj['infos'], dtype=object) for traj in self.trajectories]
        
        np.savez(
            output_path.with_suffix('.npz'),
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            infos=infos,
            metadata=self.metadata
        )
    
    def close(self) -> None:
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


def main():
    """Main function for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate MetaWorld demonstration dataset")
    
    # Environment parameters
    parser.add_argument("--env_name", type=str, required=True,
                       help="MetaWorld environment name (e.g., reach-v1)")
    parser.add_argument("--num_demos", type=int, default=100,
                       help="Number of demonstrations to generate")
    parser.add_argument("--max_episode_steps", type=int, default=500,
                       help="Maximum steps per episode")
    
    # Rendering parameters
    parser.add_argument("--render_mode", type=str, default="rgb_array",
                       help="Rendering mode")
    parser.add_argument("--camera_name", type=str, default="corner",
                       help="Camera name for rendering")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224],
                       help="Image size for rendering")
    
    # Generation parameters
    parser.add_argument("--policy_type", type=str, default="random",
                       choices=["random", "scripted"],
                       help="Type of policy to use")
    parser.add_argument("--success_rate_threshold", type=float, default=0.0,
                       help="Minimum success rate threshold")
    parser.add_argument("--max_attempts", type=int, default=None,
                       help="Maximum number of attempts")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./data",
                       help="Output directory for dataset")
    parser.add_argument("--output_format", type=str, default="pickle",
                       choices=["pickle", "h5", "npz"],
                       help="Output format for dataset")
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
    print("METAWORLD DATASET GENERATION")
    print("=" * 60)
    print(f"Environment: {args.env_name}")
    print(f"Number of demos: {args.num_demos}")
    print(f"Max episode steps: {args.max_episode_steps}")
    print(f"Policy type: {args.policy_type}")
    print(f"Success rate threshold: {args.success_rate_threshold}")
    print(f"Output path: {output_path}")
    print(f"Output format: {args.output_format}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    try:
        # Create data generator
        generator = MetaWorldDataGenerator(
            env_name=args.env_name,
            num_demos=args.num_demos,
            max_episode_steps=args.max_episode_steps,
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
        generator.save_dataset(output_path, args.output_format)
        
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
