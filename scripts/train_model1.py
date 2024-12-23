import torch
import json
import os
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.nn import CrossEntropyLoss


# Load configurations
with open("config/params.json", "r") as f:
    params = json.load(f)
with open("models/model1/config.json", "r") as f:
    model1_config = json.load(f)


# Function to load data from JSON file
def load_data(file_path):
    with open(file_path, 'r') as f:
         return json.load(f)

def create_dataloader(data, batch_size, shuffle=True):
    input_ids = torch.tensor(data["input_ids"])
    target_ids = torch.tensor(data["rationale_ids"])
    dataset = TensorDataset(input_ids, target_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def train_model1(train_dataloader, val_dataloader, model, optimizer, scheduler, num_epochs, device):
    """Trains Model 1."""
    model.train()  # Set model to training mode
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
      print(f"Epoch: {epoch+1}/{num_epochs}")
      total_train_loss = 0
      for batch in tqdm(train_dataloader, desc="Training"):
            input_ids = batch[0].to(device)
            target_ids = batch[1].to(device)

            model.zero_grad()  # Zero the gradients
            outputs = model(input_ids=input_ids, labels=target_ids)

            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

      avg_train_loss = total_train_loss / len(train_dataloader)

      # Validation Loop
      model.eval() # Set model to evaluation mode
      total_val_loss = 0

      with torch.no_grad():
            for batch in tqdm(val_dataloader, desc = "Validation"):
                input_ids = batch[0].to(device)
                target_ids = batch[1].to(device)
                outputs = model(input_ids = input_ids, labels = target_ids)
                total_val_loss += outputs.loss.item()


      avg_val_loss = total_val_loss / len(val_dataloader)

      print(f"   - Training Loss: {avg_train_loss:.4f}")
      print(f"   - Validation Loss: {avg_val_loss:.4f}")

      # Save best model
      if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("      - Saving model weights")
            os.makedirs("models/model1/saved_weights", exist_ok=True)
            torch.save(model.state_dict(), "models/model1/saved_weights/model1.pt")

if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model1_config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(model1_config["model_name"]).to(device)


    # Load the data
    train_data = load_data("data/splits/train.json")
    val_data = load_data("data/splits/val.json")


    # Create dataloaders
    train_dataloader = create_dataloader(train_data, batch_size=model1_config["batch_size"])
    val_dataloader = create_dataloader(val_data, batch_size=model1_config["batch_size"], shuffle=False)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=model1_config["learning_rate"])

    # set up scheduler
    total_steps = len(train_dataloader) * model1_config["num_epochs"]
    warmup_steps = int(total_steps * model1_config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


    # Train the model
    train_model1(train_dataloader, val_dataloader, model, optimizer, scheduler, num_epochs = model1_config["num_epochs"], device = device)

    print("Model 1 training complete.")
