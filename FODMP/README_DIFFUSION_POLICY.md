# Diffusion Policy Implementation for FODMP

This document describes the comprehensive diffusion policy implementation for the FODMP (Foundation of Diffusion Movement Primitives) framework, specifically designed for MetaWorld robotic manipulation tasks.

## Overview

The diffusion policy implementation provides a complete framework for training and deploying diffusion-based policies for robotic manipulation. It integrates with the existing FODMP codebase and leverages the movement-primitive-diffusion library for core diffusion functionality.

## Key Components

### 1. DPCNNMetaWorldWorkspace (`workspace/dp_cnn_metaworld_workspace.py`)

The main workspace class that orchestrates the entire training and evaluation pipeline:

- **Agent Integration**: Seamlessly integrates with `DiscreteTimeDiffusionAgent` from movement-primitive-diffusion
- **Environment Support**: Full support for MetaWorld environments with proper observation and action handling
- **Training Pipeline**: Complete training loop with evaluation, checkpointing, and logging
- **Video Logging**: Automatic video generation for successful and failed trajectories
- **WandB Integration**: Built-in support for experiment tracking and visualization

#### Key Features:
- Configurable observation and action horizons (`t_obs`, `t_act`)
- Support for both future and past action prediction
- Automatic normalization of observations and actions
- Comprehensive evaluation metrics
- Robust error handling and checkpointing

### 2. MetaWorldDataset (`datasets/metaworld_dataset.py`)

A PyTorch dataset class for loading and processing MetaWorld demonstration data:

- **Multiple Formats**: Support for pickle, HDF5, and NPZ file formats
- **Flexible Configuration**: Configurable observation/action horizons and normalization
- **Data Augmentation**: Built-in noise injection for robust training
- **Efficient Loading**: Optimized data loading with proper memory management

#### Key Features:
- Automatic normalization statistics computation
- Support for variable-length trajectories
- Configurable noise augmentation
- Comprehensive dataset information and statistics

### 3. Training Script (`train_dp.py`)

A comprehensive training script with full command-line interface:

- **Hydra Integration**: Full configuration management with Hydra
- **Command Line Interface**: Extensive command-line options for all parameters
- **Resume Training**: Support for resuming from checkpoints
- **Flexible Logging**: Optional WandB integration with configurable project settings

#### Usage Examples:
```bash
# Basic training
python train_dp.py --config-name=metaworld_dp_cnn --dataset_path=/path/to/dataset

# Training with custom parameters
python train_dp.py --config-name=metaworld_dp_cnn \
    --dataset_path=/path/to/dataset \
    --num_epochs=200 \
    --batch_size=64 \
    --learning_rate=2e-4 \
    --use_wandb \
    --wandb_project=my-dp-project

# Resume training
python train_dp.py --config-name=metaworld_dp_cnn \
    --dataset_path=/path/to/dataset \
    --resume_from=./checkpoints/checkpoint_epoch_50.pt
```

### 4. Configuration System

The implementation uses Hydra for configuration management with a hierarchical structure:

- **Environment Configuration**: MetaWorld environment setup and parameters
- **Agent Configuration**: Diffusion policy model architecture and training parameters
- **Dataset Configuration**: Data loading and preprocessing parameters
- **Training Configuration**: Training hyperparameters and evaluation settings

#### Configuration Files:
- `conf/real_world/experiments/metaworld_dp_cnn.yaml`: Main configuration template
- `conf/real_world/agent_config/diffusion_policy_unet1d_agent.yaml`: Agent configuration
- `conf/real_world/agent_config/diffusion_policy_transformer_agent.yaml`: Alternative agent configuration

### 5. Example Script (`examples/diffusion_policy_example.py`)

A comprehensive example script demonstrating all aspects of the implementation:

- **Dataset Creation**: Generates sample datasets for testing
- **Training Demonstration**: Shows how to train a diffusion policy
- **Inference Demonstration**: Demonstrates how to use trained models for prediction
- **Evaluation Demonstration**: Shows how to evaluate model performance

## Architecture

The diffusion policy implementation follows a modular architecture:

```
FODMP/
├── workspace/
│   ├── base_workspace.py              # Base workspace class
│   └── dp_cnn_metaworld_workspace.py # Diffusion policy workspace
├── datasets/
│   ├── __init__.py
│   └── metaworld_dataset.py           # MetaWorld dataset class
├── train_dp.py                        # Training script
├── examples/
│   ├── __init__.py
│   └── diffusion_policy_example.py   # Example usage
└── conf/
    └── real_world/
        ├── experiments/
        │   └── metaworld_dp_cnn.yaml  # Main configuration
        └── agent_config/
            ├── diffusion_policy_unet1d_agent.yaml
            └── diffusion_policy_transformer_agent.yaml
```

## Key Features

### 1. Diffusion Policy Core
- **Noise Scheduling**: Uses DDPM scheduler for diffusion process
- **Model Architecture**: Supports both UNet1D and Transformer architectures
- **EMA Weights**: Exponential moving average for stable training
- **Inference Optimization**: Configurable number of inference steps

### 2. Environment Integration
- **MetaWorld Support**: Full integration with MetaWorld environments
- **Observation Processing**: Handles both state and image observations
- **Action Execution**: Proper action sequence execution with configurable horizons
- **Success Detection**: Automatic success/failure detection for evaluation

### 3. Training Infrastructure
- **Checkpointing**: Automatic model saving and loading
- **Evaluation**: Regular evaluation during training
- **Logging**: Comprehensive logging with WandB integration
- **Error Handling**: Robust error handling and recovery

### 4. Data Processing
- **Normalization**: Automatic data normalization
- **Augmentation**: Built-in data augmentation
- **Format Support**: Multiple dataset format support
- **Memory Efficiency**: Optimized data loading

## Usage Guide

### 1. Basic Training

```python
from FODMP.workspace.dp_cnn_metaworld_workspace import DPCNNMetaWorldWorkspace

# Initialize workspace
workspace = DPCNNMetaWorldWorkspace(
    env_config=env_config,
    agent_config=agent_config,
    dataset_config=dataset_config,
    t_act=4,
    t_obs=2
)

# Train the model
workspace.train(
    dataset_path="/path/to/dataset",
    num_epochs=100,
    batch_size=32,
    device="cuda"
)
```

### 2. Using Configuration Files

```python
from train_dp import train_with_config

# Train with configuration
workspace = train_with_config(
    config_path="conf/real_world/experiments/metaworld_dp_cnn.yaml",
    dataset_path="/path/to/dataset",
    checkpoint_dir="./checkpoints",
    device="cuda",
    use_wandb=True,
    wandb_project="my-project"
)
```

### 3. Inference

```python
# Load trained model
workspace.load_checkpoint("./checkpoints/final_model.pt")

# Run inference
observations = torch.randn(1, 2, 20)  # batch_size=1, t_obs=2, obs_dim=20
actions = workspace.agent.predict(
    observation={"observations": observations},
    extra_inputs={}
)
```

### 4. Evaluation

```python
# Test in environment
results = workspace.test_agent(workspace.agent, num_trajectories=10)
print(f"Success rate: {results['success_rate']:.3f}")
```

## Configuration Parameters

### Environment Configuration
- `env_name`: MetaWorld task name (e.g., "reach-v2", "push-v2")
- `time_limit`: Maximum episode length
- `render_mode`: Rendering mode for visualization

### Agent Configuration
- `t_obs`: Number of observation timesteps
- `t_act`: Number of action timesteps
- `predict_past`: Whether to predict past actions
- `num_inference_steps`: Number of denoising steps during inference
- `use_ema`: Whether to use exponential moving average

### Training Configuration
- `num_epochs`: Number of training epochs
- `batch_size`: Training batch size
- `learning_rate`: Learning rate for optimizer
- `eval_every`: Evaluation frequency
- `test_every`: Environment testing frequency

### Dataset Configuration
- `normalize_actions`: Whether to normalize actions
- `normalize_observations`: Whether to normalize observations
- `action_noise_std`: Action noise augmentation
- `observation_noise_std`: Observation noise augmentation

## Performance Considerations

### 1. Memory Usage
- Use appropriate batch sizes based on available GPU memory
- Consider using gradient accumulation for large effective batch sizes
- Monitor memory usage during training

### 2. Training Speed
- Use multiple workers for data loading
- Enable mixed precision training if supported
- Consider using smaller inference steps for faster evaluation

### 3. Model Performance
- Tune the number of inference steps for optimal performance/speed trade-off
- Use EMA weights for more stable training
- Regular evaluation helps identify overfitting

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Reduce model size

2. **Poor Training Performance**
   - Check data normalization
   - Verify observation/action dimensions
   - Adjust learning rate

3. **Slow Inference**
   - Reduce number of inference steps
   - Use smaller models
   - Enable model optimization

### Debug Tips

1. **Enable Debug Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Visualize Training Progress**
   - Use WandB for real-time monitoring
   - Plot training curves
   - Monitor evaluation metrics

3. **Check Data Quality**
   - Visualize dataset samples
   - Verify normalization statistics
   - Check for data anomalies

## Future Enhancements

### Planned Features
- Support for more environment types (ManiSkill, custom environments)
- Additional model architectures (Vision Transformer, etc.)
- Advanced data augmentation techniques
- Multi-task learning support
- Real-time inference optimization

### Contributing
Contributions are welcome! Please refer to the main FODMP repository for contribution guidelines.

## References

1. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu/)
2. [MetaWorld: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning](https://meta-world.github.io/)
3. [Movement Primitive Diffusion](https://github.com/columbia-ai-robotics/movement-primitive-diffusion)

## License

This implementation follows the same license as the main FODMP project.
