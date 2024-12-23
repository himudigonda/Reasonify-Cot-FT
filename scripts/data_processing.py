import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Load configuration
with open("config/params.json", "r") as f:
    params = json.load(f)
with open("models/model1/config.json", "r") as f:
    model1_config = json.load(f)
with open("models/model2/config.json", "r") as f:
    model2_config = json.load(f)


def tokenize_and_split(data, tokenizer, max_length):
    """Tokenizes data and creates train/val/test splits."""
    source_texts = [example["source"] for example in data]
    target_texts = [example["target"] for example in data]
    rationale_texts = [example["rationale"] for example in data]

    tokenized_sources = tokenizer(source_texts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    tokenized_targets = tokenizer(target_texts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    tokenized_rationales = tokenizer(rationale_texts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")


    train_data, test_data, train_target, test_target, train_rationale, test_rationale = train_test_split(
        tokenized_sources["input_ids"], tokenized_targets["input_ids"], tokenized_rationales["input_ids"], test_size=params["test_size"], random_state=42
    )

    train_data, val_data, train_target, val_target, train_rationale, val_rationale = train_test_split(
        train_data, train_target, train_rationale, test_size=params["val_size"], random_state=42
    )

    train = {"input_ids":train_data, "target_ids":train_target, "rationale_ids": train_rationale}
    val = {"input_ids":val_data, "target_ids":val_target, "rationale_ids": val_rationale}
    test = {"input_ids":test_data, "target_ids":test_target, "rationale_ids": test_rationale}

    return train, val, test

def save_splits(train_data, val_data, test_data, output_dir):
        """Saves the train/val/test splits in the specified output directory"""
        os.makedirs(output_dir, exist_ok=True)

        for split_name, data in zip(["train", "val", "test"], [train_data, val_data, test_data]):
            filename = os.path.join(output_dir, f"{split_name}.json")
            with open(filename, 'w') as file:
                json.dump({
                    "input_ids": data["input_ids"].tolist(),
                    "target_ids": data["target_ids"].tolist(),
                     "rationale_ids": data["rationale_ids"].tolist(),
                }, file)

if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("kaist-ai/CoT-Collection")
    dataset = dataset["train"]

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model1_config["model_name"])

    # Tokenize and Split the data
    train, val, test = tokenize_and_split(dataset, tokenizer, model1_config["max_length"])


    # Save the splits
    save_splits(train, val, test, "data/splits/")

    print("Data tokenized and split successfully.")
