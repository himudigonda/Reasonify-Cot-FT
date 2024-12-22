#!/bin/bash

# Create the project directory structure
mkdir -p cot-project/data/raw
mkdir -p cot-project/data/processed
mkdir -p cot-project/data/splits
mkdir -p cot-project/models/model1/saved_weights
mkdir -p cot-project/models/model2/saved_weights
mkdir -p cot-project/scripts
mkdir -p cot-project/logs
mkdir -p cot-project/config
mkdir -p cot-project/notebooks

# Create empty placeholder config files
touch cot-project/models/model1/config.json
touch cot-project/models/model2/config.json
touch cot-project/config/params.json

# Create basic python files
touch cot-project/scripts/data_processing.py
touch cot-project/scripts/train_model1.py
touch cot-project/scripts/train_model2.py
touch cot-project/scripts/evaluate.py
touch cot-project/scripts/inference.py
touch cot-project/scripts/utils.py

# Create a basic requirements.txt file
cat << EOF > cot-project/requirements.txt
torch
transformers
datasets
tqdm
pandas
numpy
scikit-learn
EOF

# Create a basic README.md file
cat << EOF > cot-project/README.md
# Conversational Chain of Thought Project

This project implements a two-model conversational system for generating a chain of thought based conversation.
EOF

echo "Project structure and requirements.txt created successfully!"
echo "Now remember to activate your virtual environment and install the packages using: 'pip install -r cot-project/requirements.txt'"
