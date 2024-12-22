#!/bin/bash

echo "Downloading dataset..."
python3 -c 'from datasets import load_dataset; load_dataset("kaist-ai/CoT-Collection", cache_dir="./data")'
echo "Dataset downloaded to data."

echo "Downloading Model..."
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
mkdir -p models
python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('$MODEL_NAME', cache_dir='./models'); model.save_pretrained('./models/pretrained_model'); tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME', cache_dir='./models'); tokenizer.save_pretrained('./models/pretrained_model')"
echo "Model and tokenizer downloaded to ./models/pretrained_model"
