import json
import os
from datasets import load_dataset
import pandas as pd
import torch

def load_config(config_path):
    """Loads configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"-[info] utils.load_config : Loaded config from {config_path}")
            return config
    except FileNotFoundError:
        print(f"-[error] utils.load_config : Config file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"-[error] utils.load_config : Invalid JSON format in {config_path}")
        return None

def load_data(file_path):
  """Loads data from parquet file"""
  print(f"-[info] utils.load_data : Loading data from {file_path}")
  try:
    df = pd.read_parquet(file_path)
    print(f"-[debug] utils.load_data : Data Loaded Successfully")
    return df
  except FileNotFoundError:
      print(f"-[error] utils.load_data : File not found at {file_path}")
      return None
  except Exception as e:
    print(f"-[error] utils.load_data : {e}")
    return None

def create_dataloader(data, batch_size, shuffle=True, input_col='input_ids', target_col='labels', rationale_col = None):
    """Creates a PyTorch DataLoader from a Pandas DataFrame."""
    print(f"-[info] utils.create_dataloader : Creating DataLoader")

    input_ids = torch.tensor(data[input_col].tolist())
    target_ids = torch.tensor(data[target_col].tolist())
    dataset_tensors = [input_ids, target_ids]

    if rationale_col is not None:
      rationale_ids = torch.tensor(data[rationale_col].tolist())
      dataset_tensors.append(rationale_ids)

    dataset = torch.utils.data.TensorDataset(*dataset_tensors)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"-[debug] utils.create_dataloader : Dataloader created successfully")
    return dataloader


def save_model(model, tokenizer, output_path, model_name):
  """Saves the model and the tokenizer"""
  print(f"-[info] utils.save_model : Saving model to {output_path}")
  try:
      os.makedirs(output_path, exist_ok = True)
      model_path = os.path.join(output_path, f"{model_name}.pt")
      torch.save(model.state_dict(), model_path)
      tokenizer.save_pretrained(output_path)
      print(f"-[info] utils.save_model : Model and tokenizer saved to: {output_path}")

  except Exception as e:
    print(f"-[error] utils.save_model : Failed to save model and tokenizer {e}")

def setup_logging(log_file):
  """Initializes a basic logging system to a file."""
  print(f"-[info] utils.setup_logging : Setting up logging to {log_file}")
  if not os.path.exists(os.path.dirname(log_file)):
    os.makedirs(os.path.dirname(log_file))
  with open(log_file, 'w') as f:
    print(f"-[debug] utils.setup_logging : Logging file cleared")


def log(log_file, message):
  """Appends a log message to the specified file."""
  with open(log_file, 'a') as f:
        f.write(message + "\n")
        print(f"-[debug] utils.log : {message}")


def add_padding_token(tokenizer):
  """Adds padding token if one is missing"""
  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
      print(f"-[debug] utils.add_padding_token : Added padding token")
