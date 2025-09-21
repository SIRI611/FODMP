"""
MetaWorld dataset implementation for diffusion policy training.

This module provides a PyTorch dataset class for loading and processing
MetaWorld demonstration data for diffusion policy training.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pickle
import h5py


class MetaWorldDataset(Dataset):
    """
    Dataset class for MetaWorld demonstration data.
    
    This dataset loads demonstration trajectories from MetaWorld environments
    and formats them for diffusion policy training.
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
        **kwargs
    ):
        """
        Initialize MetaWorld dataset.
        
        Args:
            dataset_path: Path to the dataset file
            t_obs: Number of observation timesteps
            t_act: Number of action timesteps
            predict_past: Whether to predict past actions
            normalize_actions: Whether to normalize actions
            normalize_observations: Whether to normalize observations
            action_noise_std: Standard deviation for action noise augmentation
            observation_noise_std: Standard deviation for observation noise augmentation
        """
        self.dataset_path = Path(dataset_path)
        self.t_obs = t_obs
        self.t_act = t_act
        self.predict_past = predict_past
        self.normalize_actions = normalize_actions
        self.normalize_observations = normalize_observations
        self.action_noise_std = action_noise_std
        self.observation_noise_std = observation_noise_std
        
        # Load dataset
        self.data = self._load_dataset()
        
        # Compute normalization statistics
        self.action_mean = None
        self.action_std = None
        self.obs_mean = None
        self.obs_std = None
        
        if self.normalize_actions or self.normalize_observations:
            self._compute_normalization_stats()
            
        print(f"Loaded MetaWorld dataset:")
        print(f"  - Dataset path: {self.dataset_path}")
        print(f"  - Number of trajectories: {len(self.data['trajectories'])}")
        print(f"  - Total timesteps: {sum(len(traj['observations']) for traj in self.data['trajectories'])}")
        print(f"  - Observation shape: {self.data['observations'][0].shape}")
        print(f"  - Action shape: {self.data['actions'][0].shape}")
        
    def _load_dataset(self) -> Dict[str, Any]:
        """Load dataset from file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
            
        # Try different file formats
        if self.dataset_path.suffix == '.pkl':
            return self._load_pickle_dataset()
        elif self.dataset_path.suffix == '.h5' or self.dataset_path.suffix == '.hdf5':
            return self._load_h5_dataset()
        elif self.dataset_path.suffix == '.npz':
            return self._load_npz_dataset()
        else:
            raise ValueError(f"Unsupported dataset format: {self.dataset_path.suffix}")
            
    def _load_pickle_dataset(self) -> Dict[str, Any]:
        """Load dataset from pickle file."""
        with open(self.dataset_path, 'rb') as f:
            data = pickle.load(f)
        return data
        
    def _load_h5_dataset(self) -> Dict[str, Any]:
        """Load dataset from HDF5 file."""
        data = {}
        with h5py.File(self.dataset_path, 'r') as f:
            # Load trajectories
            data['trajectories'] = []
            for traj_idx in range(len(f['trajectories'])):
                traj = {
                    'observations': f['trajectories'][traj_idx]['observations'][:],
                    'actions': f['trajectories'][traj_idx]['actions'][:],
                    'rewards': f['trajectories'][traj_idx]['rewards'][:],
                    'dones': f['trajectories'][traj_idx]['dones'][:],
                    'infos': f['trajectories'][traj_idx]['infos'][:]
                }
                data['trajectories'].append(traj)
                
            # Load metadata
            if 'metadata' in f:
                data['metadata'] = dict(f['metadata'].attrs)
                
        return data
        
    def _load_npz_dataset(self) -> Dict[str, Any]:
        """Load dataset from NPZ file."""
        data = np.load(self.dataset_path, allow_pickle=True)
        
        # Convert to expected format
        trajectories = []
        for i in range(len(data['observations'])):
            traj = {
                'observations': data['observations'][i],
                'actions': data['actions'][i],
                'rewards': data.get('rewards', [None] * len(data['observations']))[i],
                'dones': data.get('dones', [None] * len(data['observations']))[i],
                'infos': data.get('infos', [None] * len(data['observations']))[i]
            }
            trajectories.append(traj)
            
        return {
            'trajectories': trajectories,
            'metadata': dict(data.get('metadata', {}).item()) if 'metadata' in data else {}
        }
        
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
            'normalize_observations': self.normalize_observations
        }
