#!/bin/bash

# Create project directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/cot_generator
mkdir -p models/response_generator
mkdir -p logs
mkdir -p src
mkdir -p scripts

# Create necessary files
touch src/data_processing.py
touch src/cot_generator_trainer.py
touch src/response_generator_trainer.py
touch src/inference.py
touch src/utils.py
touch scripts/setup.sh
touch scripts/train_cot_generator.sh
touch scripts/train_response_generator.sh
touch scripts/download_data.sh
touch requirements.txt
touch README.md

# Make scripts executable
chmod +x scripts/*.sh

echo "Project structure created."
