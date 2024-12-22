#!/bin/bash

# Project directory
PROJECT_DIR="CoTBros"

# Create directories
mkdir -p "$PROJECT_DIR"/data/raw
mkdir -p "$PROJECT_DIR"/data/processed
mkdir -p "$PROJECT_DIR"/models/cot_generator
mkdir -p "$PROJECT_DIR"/models/response_generator
mkdir -p "$PROJECT_DIR"/logs
mkdir -p "$PROJECT_DIR"/src
mkdir -p "$PROJECT_DIR"/scripts

# Create files
touch "$PROJECT_DIR"/src/data_processing.py
touch "$PROJECT_DIR"/src/cot_training.py
touch "$PROJECT_DIR"/src/response_training.py
touch "$PROJECT_DIR"/src/inference.py
touch "$PROJECT_DIR"/src/utils.py
touch "$PROJECT_DIR"/scripts/download_data.sh
touch "$PROJECT_DIR"/scripts/train_cot.sh
touch "$PROJECT_DIR"/scripts/train_response.sh
touch "$PROJECT_DIR"/requirements.txt
touch "$PROJECT_DIR"/README.md
echo "Project structure created."
