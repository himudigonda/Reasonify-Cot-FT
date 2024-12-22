import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from src.utils import setup_logging
import os
import logging

logger = setup_logging("logs", logging.INFO)

def get_dataloaders(data_path, model_name, batch_size=16, max_length=512):
    """Loads dataset, preprocess, and creates datasets/dataloaders for both models."""
    logger.info(f"Loading dataset from {data_path} and preparing dataloaders")

    dataset = load_dataset(data_path, name="en") # Assuming the name is 'en'
    train_dataset = dataset['train']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model 1 Dataset
    class CoTDataset(Dataset):
        def __init__(self, dataset, tokenizer, max_length=512):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            item = self.dataset[idx]
            source = item['source']
            target = item['rationale']
            tokenized_source = self.tokenizer(source, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            tokenized_target = self.tokenizer(target, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            return {
              'input_ids': tokenized_source['input_ids'].squeeze(),
              'attention_mask': tokenized_source['attention_mask'].squeeze(),
              'labels': tokenized_target['input_ids'].squeeze(),
             }

    train_dataset_model1 = CoTDataset(train_dataset, tokenizer, max_length=max_length)

    # Model 2 Dataset
    class ResponseDataset(Dataset):
        def __init__(self, dataset, tokenizer, max_length=512):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            item = self.dataset[idx]
            source = item['source']
            rationale = item['rationale']
            target = item['target']
            combined = f"Query: {source} Reason: {rationale}"
            tokenized_combined = self.tokenizer(combined, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            tokenized_target = self.tokenizer(target, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            return {
                'input_ids': tokenized_combined['input_ids'].squeeze(),
                'attention_mask': tokenized_combined['attention_mask'].squeeze(),
                'labels': tokenized_target['input_ids'].squeeze(),
                }

    train_dataset_model2 = ResponseDataset(train_dataset, tokenizer, max_length=max_length)

    # Create data loaders
    model1_dataloader = DataLoader(train_dataset_model1, batch_size=batch_size, shuffle=True)
    model2_dataloader = DataLoader(train_dataset_model2, batch_size=batch_size, shuffle=True)

    logger.info(f"Dataloaders ready.")
    return model1_dataloader, model2_dataloader


if __name__ == '__main__':
    data_path = "kaist-ai/CoT-Collection"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model1_dataloader, model2_dataloader = get_dataloaders(data_path, model_name)

    # Example: Fetching and printing a batch
    print("Example Batch for Model 1:")
    example_batch = next(iter(model1_dataloader))
    print(example_batch)

    print("Example Batch for Model 2:")
    example_batch = next(iter(model2_dataloader))
    print(example_batch)

    logger.info("Example batches printed.")
