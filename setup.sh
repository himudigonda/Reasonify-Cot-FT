#!/bin/bash

# Create the project directory structure
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/splits
mkdir -p models/model1/saved_weights
mkdir -p models/model2/saved_weights
mkdir -p scripts
mkdir -p logs
mkdir -p config
mkdir -p notebooks

# Create empty placeholder config files
touch models/model1/config.json
touch models/model2/config.json
touch config/params.json

# Create basic python files
touch scripts/data_processing.py
touch scripts/train_model1.py
touch scripts/train_model2.py
touch scripts/evaluate.py
touch scripts/inference.py
touch scripts/utils.py

# Create a basic requirements.txt file
cat << EOF > requirements.txt
torch
transformers
datasets
tqdm
pandas
numpy
scikit-learn
EOF

# Create a basic README.md file
cat << EOF > README.md
# Conversational Chain of Thought Project

This project implements a two-model conversational system for generating a chain of thought based conversation.
EOF

echo "Project structure and requirements.txt created successfully!"
echo "Now remember to activate your virtual environment and install the packages using: 'pip install -r requirements.txt'"
