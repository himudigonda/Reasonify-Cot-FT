import torch
import json
import os
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from .utils import load_config, load_data, create_dataloader, save_model, setup_logging, log
from tqdm import tqdm

def train_step(model1, model2, batch, device, tokenizer, model1_max_length, model2_max_length):
  """Trains both models in a single step with combined loss."""
  model1.train()
  model2.train()

  input_ids = batch[0].to(device)
  target_ids = batch[1].to(device)
  rationale_ids = batch[2].to(device)

  # Generate the CoT from model 1
  cot_outputs = model1.generate(input_ids=input_ids, max_new_tokens = model1_max_length)
  cot_text = tokenizer.batch_decode(cot_outputs, skip_special_tokens = True)
  # Tokenize the CoT
  cot_ids = tokenizer(cot_text, padding="max_length", truncation = True, max_length=model1_max_length, return_tensors = "pt").input_ids.to(device)

  # Concatenate the input and CoT
  combined_input = torch.cat((input_ids, cot_ids), dim = 1).to(device)
  combined_input = combined_input[:, :model2_max_length]

  # Training model 1
  model1.zero_grad()
  model1_outputs = model1(input_ids = input_ids, labels = rationale_ids)
  model1_loss = model1_outputs.loss

  # Training model 2
  model2.zero_grad()
  model2_outputs = model2(input_ids = combined_input, labels = target_ids)
  model2_loss = model2_outputs.loss

  # Combined Loss
  combined_loss = model1_loss + model2_loss
  combined_loss.backward()

  return combined_loss, model1_loss, model2_loss

def validate(model1, model2, val_dataloader, device, tokenizer, model1_max_length, model2_max_length):
  """Validates both models on the validation dataset."""
  model1.eval()
  model2.eval()

  total_loss = 0
  total_m1_loss = 0
  total_m2_loss = 0
  with torch.no_grad():
    for batch in val_dataloader:
      loss, model1_loss, model2_loss = train_step(model1, model2, batch, device, tokenizer, model1_max_length, model2_max_length)
      total_loss += loss.item()
      total_m1_loss += model1_loss.item()
      total_m2_loss += model2_loss.item()
  avg_loss = total_loss / len(val_dataloader)
  avg_m1_loss = total_m1_loss / len(val_dataloader)
  avg_m2_loss = total_m2_loss / len(val_dataloader)
  return avg_loss, avg_m1_loss, avg_m2_loss

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

  # Setup Logging
  log_file = "logs/training_log.txt"
  setup_logging(log_file)
  log(log_file, "Starting combined training process")

  # Set device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  log(log_file, f"Using device: {device}")

  # Load models and tokenizer
  log(log_file, "Loading Models and Tokenizer")
  tokenizer = AutoTokenizer.from_pretrained(model1_config["model_name"])
  model1 = AutoModelForCausalLM.from_pretrained(model1_config["model_name"]).to(device)
  model2 = AutoModelForCausalLM.from_pretrained(model2_config["model_name"]).to(device)
  log(log_file, "Models and tokenizer loaded")

  # Load data
  log(log_file, "Loading data")
  train_df = load_data("data/splits/train.parquet")
  val_df = load_data("data/splits/val.parquet")
  log(log_file, "Data Loaded")


  # Create Dataloaders
  train_dataloader = create_dataloader(train_df, batch_size=model1_config["batch_size"], shuffle=True, rationale_col = "rationale_ids")
  val_dataloader = create_dataloader(val_df, batch_size = model1_config["batch_size"], shuffle = False, rationale_col = "rationale_ids")
  log(log_file, "Dataloaders Created")

  # Setup optimizer
  optimizer1 = AdamW(model1.parameters(), lr=model1_config["learning_rate"])
  optimizer2 = AdamW(model2.parameters(), lr=model2_config["learning_rate"])
  log(log_file, "Optimizers created")

  # set up scheduler
  total_steps = len(train_dataloader) * model1_config["num_epochs"]
  warmup_steps = int(total_steps * model1_config["warmup_ratio"])
  scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
  scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
  log(log_file, "Schedulers created")


  # Training Loop
  best_val_loss = float('inf')
  for epoch in range(model1_config["num_epochs"]):
      log(log_file, f"Starting epoch: {epoch+1}/{model1_config['num_epochs']}")

      # Training Loop
      model1.train()
      model2.train()

      train_loss = 0
      train_m1_loss = 0
      train_m2_loss = 0
      progress_bar = tqdm(enumerate(train_dataloader), total = len(train_dataloader), desc = f"Epoch {epoch+1}/{model1_config['num_epochs']}")
      for i, batch in progress_bar:
        combined_loss, m1_loss, m2_loss = train_step(model1, model2, batch, device, tokenizer, model1_config["max_length"], model2_config["max_length"])
        optimizer1.step()
        optimizer2.step()
        scheduler1.step()
        scheduler2.step()
        train_loss += combined_loss.item()
        train_m1_loss += m1_loss.item()
        train_m2_loss += m2_loss.item()
        progress_bar.set_postfix({'train_loss': combined_loss.item()})

      avg_train_loss = train_loss / len(train_dataloader)
      avg_train_m1_loss = train_m1_loss / len(train_dataloader)
      avg_train_m2_loss = train_m2_loss / len(train_dataloader)
      log(log_file, f"Epoch: {epoch+1}/{model1_config['num_epochs']}, Average Training Loss: {avg_train_loss:.4f}, Model 1 Loss: {avg_train_m1_loss}, Model 2 Loss: {avg_train_m2_loss}")

      # Validation Loop
      val_loss, val_m1_loss, val_m2_loss = validate(model1, model2, val_dataloader, device, tokenizer, model1_config["max_length"], model2_config["max_length"])
      log(log_file, f"Epoch: {epoch+1}/{model1_config['num_epochs']}, Average Validation Loss: {val_loss:.4f}, Model 1 Validation Loss: {val_m1_loss:.4f}, Model 2 Validation Loss: {val_m2_loss:.4f}")

       # Save best model
      if val_loss < best_val_loss:
          best_val_loss = val_loss
          log(log_file, f"Epoch: {epoch+1}/{model1_config['num_epochs']}, Validation loss improved, saving models...")
          save_model(model1, tokenizer, model1_config["output_dir"], "model1")
          save_model(model2, tokenizer, model2_config["output_dir"], "model2")


  log(log_file, "Finished training process")

if __name__ == '__main__':
  main()
