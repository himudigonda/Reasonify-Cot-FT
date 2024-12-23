import torch
import json
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Load configurations
with open("config/params.json", "r") as f:
    params = json.load(f)
with open("models/model1/config.json", "r") as f:
    model1_config = json.load(f)
with open("models/model2/config.json", "r") as f:
    model2_config = json.load(f)

# Function to load data from JSON file
def load_data(file_path):
    with open(file_path, 'r') as f:
         return json.load(f)

def create_dataloader(data, batch_size, shuffle=False):
    input_ids = torch.tensor(data["input_ids"])
    target_ids = torch.tensor(data["target_ids"])
    rationale_ids = torch.tensor(data["rationale_ids"])
    dataset = TensorDataset(input_ids, target_ids, rationale_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def evaluate_model1(dataloader, model, tokenizer, device):
    """Evaluates Model 1."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
            for batch in tqdm(dataloader, desc = "Evaluation"):
                input_ids = batch[0].to(device)
                target_ids = batch[1].to(device)
                outputs = model.generate(input_ids = input_ids, max_new_tokens = model1_config["max_length"])
                preds = tokenizer.batch_decode(outputs, skip_special_tokens = True)
                labels = tokenizer.batch_decode(target_ids, skip_special_tokens = True)

                all_preds.extend(preds)
                all_labels.extend(labels)


    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Model 1 Evaluation - Accuracy : {accuracy:.4f}")

def evaluate_model2(dataloader, model, tokenizer, device):
    """Evaluates Model 2."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
            for batch in tqdm(dataloader, desc = "Evaluation"):
                input_ids = batch[0].to(device)
                target_ids = batch[1].to(device)
                rationale_ids = batch[2].to(device)

                combined_ids = torch.cat((input_ids, rationale_ids), dim = 1).to(device)
                outputs = model.generate(input_ids = combined_ids, max_new_tokens = model2_config["max_length"])
                preds = tokenizer.batch_decode(outputs, skip_special_tokens = True)
                labels = tokenizer.batch_decode(target_ids, skip_special_tokens = True)
                all_preds.extend(preds)
                all_labels.extend(labels)


    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Model 2 Evaluation - Accuracy : {accuracy:.4f}")

if __name__ == '__main__':
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


    # Load the test data
    test_data = load_data("data/splits/test.json")


    # Create dataloaders
    test_dataloader = create_dataloader(test_data, batch_size = model1_config["batch_size"], shuffle=False)

    # Evaluate the models
    evaluate_model1(test_dataloader, model1, tokenizer1, device)
    evaluate_model2(test_dataloader, model2, tokenizer2, device)
    print("Evaluation Complete")
