#!/bin/bash

# Create directories
mkdir -p data
mkdir -p src
mkdir -p scripts
mkdir -p notebooks

# Create files in src
touch src/__init__.py
touch src/data_loading.py
touch src/model_loading.py
touch src/trainer.py
touch src/evaluation.py
touch src/utils.py

# Create files in scripts
touch scripts/run.sh

# Create empty requirements.txt
touch requirements.txt

echo "Project structure setup complete."
