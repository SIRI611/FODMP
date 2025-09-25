"""
ManiSkills dataset implementation for diffusion policy training.

This module provides a PyTorch dataset class for loading and processing
ManiSkills demonstration data for diffusion policy training.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import h5py
import json


class ManiSkillsDataset(Dataset):
    """
    Dataset class for ManiSkills demonstration data.
    
    This dataset loads demonstration trajectories from ManiSkills environments
    and formats them for diffusion policy training. Supports HDF5 format
    with JSON metadata files.
    """
    
    def __init__(
        self,
        dataset_path: str,
        t_obs: int = 2,
        t_act: int = 4,
        predict_past: bool = False,
        normalize_actions: bool = True,
        normalize_observations: bool = True,
        action_noise_std: float = 0.0,
        observation_noise_std: float = 0.0,
        position_control: bool = True,
        movement_primitive_dim: int = 7,
        obs_mode: str = "state",
        **kwargs
    ):
        """
        Initialize ManiSkills dataset.
        
        Args:
            dataset_path: Path to the dataset file (.h5)
            t_obs: Number of observation timesteps
            t_act: Number of action timesteps
            predict_past: Whether to predict past actions
            normalize_actions: Whether to normalize actions
            normalize_observations: Whether to normalize observations
            action_noise_std: Standard deviation for action noise augmentation
            observation_noise_std: Standard deviation for observation noise augmentation
            position_control: Whether to use position-based control
            movement_primitive_dim: Dimension of movement primitives
            obs_mode: Observation mode ('state', 'state_dict', 'rgb', etc.)
        """
        self.dataset_path = Path(dataset_path)
        self.t_obs = t_obs
        self.t_act = t_act
        self.predict_past = predict_past
        self.normalize_actions = normalize_actions
        self.normalize_observations = normalize_observations
        self.action_noise_std = action_noise_std
        self.observation_noise_std = observation_noise_std
        self.position_control = position_control
        self.movement_primitive_dim = movement_primitive_dim
        self.obs_mode = obs_mode
        
        # Load dataset
        self.data = self._load_dataset()
        
        # Compute normalization statistics
        self.action_mean = None
        self.action_std = None
        self.obs_mean = None
        self.obs_std = None
        
        if self.normalize_actions or self.normalize_observations:
            self._compute_normalization_stats()
            
        print(f"Loaded ManiSkills dataset:")
        print(f"  - Dataset path: {self.dataset_path}")
        print(f"  - Number of trajectories: {len(self.data['trajectories'])}")
        print(f"  - Total timesteps: {sum(len(traj['observations']) for traj in self.data['trajectories'])}")
        print(f"  - Observation shape: {self.data['trajectories'][0]['observations'].shape}")
        print(f"  - Action shape: {self.data['trajectories'][0]['actions'].shape}")
        print(f"  - Position control: {self.position_control}")
        print(f"  - Movement primitive dim: {self.movement_primitive_dim}")
        
    def _load_dataset(self) -> Dict[str, Any]:
        """Load dataset from HDF5 file with JSON metadata."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
            
        if self.dataset_path.suffix not in ['.h5', '.hdf5']:
            raise ValueError(f"ManiSkills dataset must be in HDF5 format, got: {self.dataset_path.suffix}")
            
        # Load JSON metadata if available
        metadata_path = self.dataset_path.with_suffix('.json')
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
        # Load HDF5 data
        data = {'metadata': metadata, 'trajectories': []}
        
        with h5py.File(self.dataset_path, 'r') as f:
            # Get trajectory keys (traj_0, traj_1, etc.)
            traj_keys = [key for key in f.keys() if key.startswith('traj_')]
            traj_keys.sort(key=lambda x: int(x.split('_')[1]))  # Sort by episode ID
            
            for traj_key in traj_keys:
                traj_group = f[traj_key]
                trajectory = self._load_trajectory(traj_group)
                data['trajectories'].append(trajectory)
                
        return data
        
    def _load_trajectory(self, traj_group: h5py.Group) -> Dict[str, Any]:
        """Load a single trajectory from HDF5 group."""
        trajectory = {}
        
        # Load actions
        if 'actions' in traj_group:
            trajectory['actions'] = traj_group['actions'][:]
        else:
            raise ValueError("Trajectory missing 'actions' data")
            
        # Load observations
        if 'obs' in traj_group:
            obs_data = traj_group['obs'][:]
            trajectory['observations'] = self._process_observations(obs_data)
        else:
            raise ValueError("Trajectory missing 'obs' data")
            
        # Load termination flags
        trajectory['terminated'] = traj_group.get('terminated', np.zeros(len(trajectory['actions']), dtype=bool))[:]
        trajectory['truncated'] = traj_group.get('truncated', np.zeros(len(trajectory['actions']), dtype=bool))[:]
        
        # Load success/failure flags if available
        trajectory['success'] = traj_group.get('success', np.zeros(len(trajectory['actions']), dtype=bool))[:]
        trajectory['fail'] = traj_group.get('fail', np.zeros(len(trajectory['actions']), dtype=bool))[:]
        
        # Load environment states if available
        if 'env_states' in traj_group:
            trajectory['env_states'] = traj_group['env_states'][:]
            
        return trajectory
        
    def _process_observations(self, obs_data: np.ndarray) -> np.ndarray:
        """Process observations based on observation mode."""
        if self.obs_mode == 'state':
            # Flatten state observations
            if obs_data.ndim > 2:
                # obs_data might be structured, flatten it
                return obs_data.reshape(obs_data.shape[0], -1)
            return obs_data
        elif self.obs_mode == 'state_dict':
            # Handle structured state dictionary observations
            if isinstance(obs_data, np.ndarray) and obs_data.dtype == object:
                # Structured data, flatten each timestep
                flattened_obs = []
                for timestep_obs in obs_data:
                    flattened_obs.append(self._flatten_state_dict(timestep_obs))
                return np.array(flattened_obs)
            return obs_data
        else:
            # For other modes (rgb, depth, etc.), return as-is
            return obs_data
            
    def _flatten_state_dict(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Flatten a state dictionary into a vector."""
        flattened = []
        
        for key, value in state_dict.items():
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                flattened.extend(self._flatten_state_dict(value).flatten())
            elif isinstance(value, np.ndarray):
                flattened.extend(value.flatten())
            elif isinstance(value, (list, tuple)):
                flattened.extend(np.array(value).flatten())
            else:
                flattened.append(value)
                
        return np.array(flattened, dtype=np.float32)
        
    def _compute_normalization_stats(self) -> None:
        """Compute normalization statistics for actions and observations."""
        all_actions = []
        all_observations = []
        
        for traj in self.data['trajectories']:
            all_actions.append(traj['actions'])
            all_observations.append(traj['observations'])
            
        # Stack all data
        all_actions = np.concatenate(all_actions, axis=0)
        all_observations = np.concatenate(all_observations, axis=0)
        
        # Compute statistics
        if self.normalize_actions:
            self.action_mean = np.mean(all_actions, axis=0)
            self.action_std = np.std(all_actions, axis=0) + 1e-8
            
        if self.normalize_observations:
            self.obs_mean = np.mean(all_observations, axis=0)
            self.obs_std = np.std(all_observations, axis=0) + 1e-8
            
        print(f"Computed normalization statistics:")
        if self.normalize_actions:
            print(f"  - Action mean: {self.action_mean}")
            print(f"  - Action std: {self.action_std}")
        if self.normalize_observations:
            print(f"  - Observation mean: {self.obs_mean}")
            print(f"  - Observation std: {self.obs_std}")
            
    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Normalize actions."""
        if self.normalize_actions and self.action_mean is not None:
            return (actions - self.action_mean) / self.action_std
        return actions
        
    def _normalize_observations(self, observations: np.ndarray) -> np.ndarray:
        """Normalize observations."""
        if self.normalize_observations and self.obs_mean is not None:
            return (observations - self.obs_mean) / self.obs_std
        return observations
        
    def _denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Denormalize actions."""
        if self.normalize_actions and self.action_mean is not None:
            return actions * self.action_std + self.action_mean
        return actions
        
    def _denormalize_observations(self, observations: np.ndarray) -> np.ndarray:
        """Denormalize observations."""
        if self.normalize_observations and self.obs_mean is not None:
            return observations * self.obs_std + self.obs_mean
        return observations
        
    def _add_noise(self, data: np.ndarray, noise_std: float) -> np.ndarray:
        """Add Gaussian noise to data."""
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, data.shape)
            return data + noise
        return data
        
    def _convert_to_position_control(self, actions: np.ndarray) -> np.ndarray:
        """Convert actions to position-based control if needed."""
        if self.position_control:
            # For position control, actions are typically joint positions
            # Ensure they match the movement primitive dimension
            if actions.shape[-1] != self.movement_primitive_dim:
                if actions.shape[-1] > self.movement_primitive_dim:
                    # Truncate to required dimension
                    actions = actions[..., :self.movement_primitive_dim]
                else:
                    # Pad with zeros
                    padding = np.zeros((*actions.shape[:-1], self.movement_primitive_dim - actions.shape[-1]))
                    actions = np.concatenate([actions, padding], axis=-1)
        return actions
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        total_samples = 0
        for traj in self.data['trajectories']:
            traj_len = len(traj['observations'])
            if traj_len >= self.t_obs + self.t_act:
                total_samples += traj_len - self.t_obs - self.t_act + 1
        return total_samples
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing observations and actions
        """
        # Find the trajectory and timestep for this index
        traj_idx, timestep = self._get_trajectory_and_timestep(idx)
        traj = self.data['trajectories'][traj_idx]
        
        # Extract observation sequence
        obs_start = timestep
        obs_end = obs_start + self.t_obs
        observations = traj['observations'][obs_start:obs_end]
        
        # Extract action sequence
        if self.predict_past:
            # Predict past actions (for causal prediction)
            act_start = max(0, obs_start - self.t_act)
            act_end = obs_start
        else:
            # Predict future actions (standard)
            act_start = obs_end
            act_end = act_start + self.t_act
            
        actions = traj['actions'][act_start:act_end]
        
        # Convert to position control if needed
        actions = self._convert_to_position_control(actions)
        
        # Ensure we have the right sequence lengths
        if len(observations) < self.t_obs:
            # Pad with last observation
            last_obs = observations[-1]
            while len(observations) < self.t_obs:
                observations = np.vstack([observations, last_obs])
                
        if len(actions) < self.t_act:
            # Pad with zero actions
            while len(actions) < self.t_act:
                actions = np.vstack([actions, np.zeros_like(actions[0])])
                
        # Normalize data
        observations = self._normalize_observations(observations)
        actions = self._normalize_actions(actions)
        
        # Add noise for data augmentation
        observations = self._add_noise(observations, self.observation_noise_std)
        actions = self._add_noise(actions, self.action_noise_std)
        
        # Convert to tensors
        sample = {
            'observations': torch.from_numpy(observations).float(),
            'actions': torch.from_numpy(actions).float()
        }
        
        # Add additional information if available
        if 'success' in traj:
            sample['success'] = torch.from_numpy(traj['success'][act_start:act_end]).bool()
        if 'terminated' in traj:
            sample['terminated'] = torch.from_numpy(traj['terminated'][act_start:act_end]).bool()
            
        return sample
        
    def _get_trajectory_and_timestep(self, idx: int) -> Tuple[int, int]:
        """Get trajectory index and timestep for a given sample index."""
        current_idx = 0
        for traj_idx, traj in enumerate(self.data['trajectories']):
            traj_len = len(traj['observations'])
            if traj_len >= self.t_obs + self.t_act:
                samples_in_traj = traj_len - self.t_obs - self.t_act + 1
                if current_idx + samples_in_traj > idx:
                    timestep = idx - current_idx
                    return traj_idx, timestep
                current_idx += samples_in_traj
                
        raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
    def get_normalizer(self) -> Dict[str, Any]:
        """Get normalization statistics for the dataset."""
        normalizer = {}
        
        if self.normalize_actions and self.action_mean is not None:
            normalizer['actions'] = {
                'mean': self.action_mean,
                'std': self.action_std
            }
            
        if self.normalize_observations and self.obs_mean is not None:
            normalizer['observations'] = {
                'mean': self.obs_mean,
                'std': self.obs_std
            }
            
        return normalizer
        
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        total_timesteps = sum(len(traj['observations']) for traj in self.data['trajectories'])
        
        return {
            'num_trajectories': len(self.data['trajectories']),
            'total_timesteps': total_timesteps,
            'avg_trajectory_length': total_timesteps / len(self.data['trajectories']),
            'observation_shape': self.data['trajectories'][0]['observations'].shape[1:],
            'action_shape': self.data['trajectories'][0]['actions'].shape[1:],
            't_obs': self.t_obs,
            't_act': self.t_act,
            'predict_past': self.predict_past,
            'normalize_actions': self.normalize_actions,
            'normalize_observations': self.normalize_observations,
            'position_control': self.position_control,
            'movement_primitive_dim': self.movement_primitive_dim,
            'obs_mode': self.obs_mode,
            'metadata': self.data['metadata']
        }
        
    def get_trajectory_info(self, traj_idx: int) -> Dict[str, Any]:
        """Get information about a specific trajectory."""
        if traj_idx >= len(self.data['trajectories']):
            raise IndexError(f"Trajectory index {traj_idx} out of range")
            
        traj = self.data['trajectories'][traj_idx]
        
        return {
            'trajectory_length': len(traj['observations']),
            'observation_shape': traj['observations'].shape,
            'action_shape': traj['actions'].shape,
            'success_rate': np.mean(traj.get('success', [False])),
            'termination_rate': np.mean(traj.get('terminated', [False])),
            'has_env_states': 'env_states' in traj
        }
        
    def filter_successful_trajectories(self, min_success_rate: float = 0.5) -> 'ManiSkillsDataset':
        """Create a filtered dataset with only successful trajectories."""
        successful_trajectories = []
        
        for traj in self.data['trajectories']:
            if 'success' in traj:
                success_rate = np.mean(traj['success'])
                if success_rate >= min_success_rate:
                    successful_trajectories.append(traj)
            else:
                # If no success information, include all trajectories
                successful_trajectories.append(traj)
                
        # Create new dataset instance
        filtered_data = self.data.copy()
        filtered_data['trajectories'] = successful_trajectories
        
        # Create new dataset instance
        new_dataset = ManiSkillsDataset.__new__(ManiSkillsDataset)
        new_dataset.__dict__.update(self.__dict__)
        new_dataset.data = filtered_data
        
        print(f"Filtered dataset: {len(successful_trajectories)}/{len(self.data['trajectories'])} trajectories with success rate >= {min_success_rate}")
        
        return new_dataset
