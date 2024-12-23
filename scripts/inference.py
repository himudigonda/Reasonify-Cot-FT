import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load configurations
with open("config/params.json", "r") as f:
    params = json.load(f)
with open("models/model1/config.json", "r") as f:
    model1_config = json.load(f)
with open("models/model2/config.json", "r") as f:
    model2_config = json.load(f)


def generate_cot(user_input, model, tokenizer, device):
  """Generates CoT using Model 1"""
  model.eval() # Evaluation mode
  input_ids = tokenizer(user_input, return_tensors="pt").input_ids.to(device)

  with torch.no_grad():
     outputs = model.generate(input_ids=input_ids, max_new_tokens = model1_config["max_length"])
  cot_text = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
  return cot_text

def generate_answer(user_input, cot_text, model, tokenizer, device):
  """Generates answer using Model 2"""
  model.eval() # Evaluation mode
  combined_input = user_input + " " + cot_text

  input_ids = tokenizer(combined_input, return_tensors="pt").input_ids.to(device)
  with torch.no_grad():
    outputs = model.generate(input_ids = input_ids, max_new_tokens = model2_config["max_length"])
  answer_text = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
  return answer_text


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load Models and Tokenizers
    tokenizer1 = AutoTokenizer.from_pretrained(model1_config["model_name"])
    model1 = AutoModelForCausalLM.from_pretrained(model1_config["model_name"]).to(device)
    model1.load_state_dict(torch.load("models/model1/saved_weights/model1.pt"))


    tokenizer2 = AutoTokenizer.from_pretrained(model2_config["model_name"])
    model2 = AutoModelForCausalLM.from_pretrained(model2_config["model_name"]).to(device)
    model2.load_state_dict(torch.load("models/model2/saved_weights/model2.pt"))


    # Interactive Loop
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        cot = generate_cot(user_input, model1, tokenizer1, device)
        print(f"Model 1 (Chain of Thought): {cot}")
        answer = generate_answer(user_input, cot, model2, tokenizer2, device)
        print(f"Model 2 (Answer): {answer}")
