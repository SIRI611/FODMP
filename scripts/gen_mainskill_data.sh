#!/bin/bash

# ManiSkills Dataset Generation Script
# This script generates demonstration datasets for ManiSkills environments

# Default parameters
ENV_NAME="PickCube-v1"
NUM_DEMOS=100
OUTPUT_DIR="../data"
OBS_MODE="state"
POLICY_TYPE="scripted"
SUCCESS_RATE_THRESHOLD=0.3
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env_name)
            ENV_NAME="$2"
            shift 2
            ;;
        --num_demos)
            NUM_DEMOS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --obs_mode)
            OBS_MODE="$2"
            shift 2
            ;;
        --policy_type)
            POLICY_TYPE="$2"
            shift 2
            ;;
        --success_rate_threshold)
            SUCCESS_RATE_THRESHOLD="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --env_name ENV_NAME              ManiSkills environment name (default: PickCube-v1)"
            echo "  --num_demos NUM_DEMOS            Number of demonstrations (default: 100)"
            echo "  --output_dir OUTPUT_DIR          Output directory (default: ./data)"
            echo "  --obs_mode OBS_MODE              Observation mode: state, state_dict, rgb, etc. (default: state)"
            echo "  --policy_type POLICY_TYPE        Policy type: random or scripted (default: random)"
            echo "  --success_rate_threshold THRESHOLD  Minimum success rate (default: 0.0)"
            echo "  --seed SEED                      Random seed (default: 42)"
            echo "  -h, --help                      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=========================================="
echo "MANISKILLS DATASET GENERATION"
echo "=========================================="
echo "Environment: $ENV_NAME"
echo "Number of demos: $NUM_DEMOS"
echo "Output directory: $OUTPUT_DIR"
echo "Observation mode: $OBS_MODE"
echo "Policy type: $POLICY_TYPE"
echo "Success rate threshold: $SUCCESS_RATE_THRESHOLD"
echo "Seed: $SEED"
echo "=========================================="

# Run the Python script
python3 ./scripts/gen_maniskills_data.py \
    --env_name="$ENV_NAME" \
    --num_demos="$NUM_DEMOS" \
    --output_dir="$OUTPUT_DIR" \
    --obs_mode="$OBS_MODE" \
    --policy_type="$POLICY_TYPE" \
    --success_rate_threshold="$SUCCESS_RATE_THRESHOLD" \
    --seed="$SEED"

echo "ManiSkills dataset generation completed!"
