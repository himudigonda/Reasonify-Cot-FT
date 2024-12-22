#!/bin/bash

echo "Starting CoT Generator training..."
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent of scripts directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Add the project root to PYTHONPATH and run the training script
PYTHONPATH=$PROJECT_ROOT python3 src/response_generator_trainer.py
echo "Response Generator training complete."
