import torch
import json
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score
import pandas as pd
from .utils import load_config, load_data, create_dataloader, log, setup_logging
import os

# Function to evaluate Model 1
def evaluate_model1(dataloader, model, tokenizer, device, max_length, log_file):
  """Evaluates Model 1."""
  log(log_file, "Starting Model 1 evaluation")
  model.eval()
  all_preds = []
  all_labels = []

  with torch.no_grad():
    for batch in dataloader:
      input_ids = batch[0].to(device)
      target_ids = batch[1].to(device)
      outputs = model.generate(input_ids = input_ids, max_new_tokens = max_length)
      preds = tokenizer.batch_decode(outputs, skip_special_tokens = True)
      labels = tokenizer.batch_decode(target_ids, skip_special_tokens = True)

      all_preds.extend(preds)
      all_labels.extend(labels)

  accuracy = accuracy_score(all_labels, all_preds)
  log(log_file, f"Model 1 Evaluation - Accuracy : {accuracy:.4f}")
  return accuracy

# Function to evaluate Model 2
def evaluate_model2(dataloader, model, tokenizer, device, max_length, log_file):
  """Evaluates Model 2."""
  log(log_file, "Starting Model 2 evaluation")
  model.eval()
  all_preds = []
  all_labels = []

  with torch.no_grad():
    for batch in dataloader:
      input_ids = batch[0].to(device)
      target_ids = batch[1].to(device)
      rationale_ids = batch[2].to(device)

      combined_ids = torch.cat((input_ids, rationale_ids), dim = 1).to(device)
      outputs = model.generate(input_ids = combined_ids, max_new_tokens = max_length)
      preds = tokenizer.batch_decode(outputs, skip_special_tokens = True)
      labels = tokenizer.batch_decode(target_ids, skip_special_tokens = True)

      all_preds.extend(preds)
      all_labels.extend(labels)

  accuracy = accuracy_score(all_labels, all_preds)
  log(log_file, f"Model 2 Evaluation - Accuracy : {accuracy:.4f}")
  return accuracy

def main():
  # Load configurations
  params = load_config("config/params.json")
  if not params:
    return
  model1_config = load_config("config/model1_config.json")
  if not model1_config:
    return
  model2_config = load_config("config/model2_config.json")
  if not model2_config:
    return

  # Logging setup
  log_file = "logs/training_log.txt"
  setup_logging(log_file)
  log(log_file, "Starting Evaluation")

  # Set device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  log(log_file, f"Using device: {device}")

  # Load Models and Tokenizers
  log(log_file, "Loading models and tokenizers")
  tokenizer1 = AutoTokenizer.from_pretrained(model1_config["model_name"])
  model1 = AutoModelForCausalLM.from_pretrained(model1_config["model_name"]).to(device)
  model1.load_state_dict(torch.load(os.path.join(model1_config["output_dir"], "model1.pt")))


  tokenizer2 = AutoTokenizer.from_pretrained(model2_config["model_name"])
  model2 = AutoModelForCausalLM.from_pretrained(model2_config["model_name"]).to(device)
  model2.load_state_dict(torch.load(os.path.join(model2_config["output_dir"], "model2.pt")))

  log(log_file, "Models and Tokenizers Loaded Successfully")

  # Load the test data
  log(log_file, "Loading test data")
  test_df = load_data("data/splits/test.parquet")

  # Create dataloaders
  test_dataloader = create_dataloader(test_df, batch_size = model1_config["batch_size"], shuffle=False, rationale_col="rationale_ids")
  log(log_file, "Dataloaders Created")

  # Evaluate the models
  log(log_file, "Evaluating models")
  model1_accuracy = evaluate_model1(test_dataloader, model1, tokenizer1, device, model1_config["max_length"], log_file)
  model2_accuracy = evaluate_model2(test_dataloader, model2, tokenizer2, device, model2_config["max_length"], log_file)
  log(log_file, "Evaluation Complete")

  log(log_file, f"Model 1 accuracy is {model1_accuracy:.4f}")
  log(log_file, f"Model 2 accuracy is {model2_accuracy:.4f}")
if __name__ == '__main__':
  main()
