import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import load_config, log, setup_logging
import os

def generate_cot(user_input, model, tokenizer, device, max_length, log_file):
  """Generates CoT using Model 1"""
  log(log_file, "Generating CoT")
  model.eval() # Evaluation mode
  input_ids = tokenizer(user_input, return_tensors="pt").input_ids.to(device)
  log(log_file, "Tokenized user input")

  with torch.no_grad():
     outputs = model.generate(input_ids=input_ids, max_new_tokens = max_length)
     log(log_file, "Model 1 Generated output")

  cot_text = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
  log(log_file, f"Chain of Thought: {cot_text}")
  return cot_text

def generate_answer(user_input, cot_text, model, tokenizer, device, max_length, log_file):
  """Generates answer using Model 2"""
  log(log_file, "Generating Answer")
  model.eval() # Evaluation mode
  combined_input = user_input + " " + cot_text

  input_ids = tokenizer(combined_input, return_tensors="pt").input_ids.to(device)
  log(log_file, "Tokenized combined input")

  with torch.no_grad():
    outputs = model.generate(input_ids = input_ids, max_new_tokens = max_length)
    log(log_file, "Model 2 Generated output")

  answer_text = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
  log(log_file, f"Generated Answer: {answer_text}")
  return answer_text


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

  # Set up Logging
  log_file = "logs/training_log.txt"
  setup_logging(log_file)
  log(log_file, "Starting Interactive Inference")

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
  log(log_file, "Models and Tokenizers Loaded")

  # Interactive Loop
  while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
      break
    cot = generate_cot(user_input, model1, tokenizer1, device, model1_config["max_length"], log_file)
    answer = generate_answer(user_input, cot, model2, tokenizer2, device, model2_config["max_length"], log_file)
    print(f"Model 2 (Answer): {answer}")

if __name__ == "__main__":
  main()
