# src/data_loading.py
from datasets import load_dataset
import torch
print("-[DEBUG] data_loading.py : data_loading module imported")


def load_cot_dataset(dataset_name="kaist-ai/CoT-Collection", split="train", fraction=0.001, validation_split=0.1):
    """Loads the CoT dataset from Hugging Face.

    Args:
      dataset_name: The name of the dataset
      split: The split to load from
      fraction: The fraction of the dataset to load
      validation_split: Fraction of data to use for validation if needed
    """
    print(f"-[INFO] data_loading.py/load_cot_dataset : Loading dataset: {dataset_name}, split: {split}, fraction: {fraction}")
    try:
        dataset = load_dataset(dataset_name, split=split)
        if fraction < 1.0:
            dataset = dataset.select(range(int(len(dataset) * fraction)))

        if split == "validation":
            # Take last validation_split % for validation
            start_idx = int(len(dataset) * (1 - validation_split))
            dataset = dataset.select(range(start_idx, len(dataset)))
        elif split == "train":
            # Take first (1-validation_split)% for training
            end_idx = int(len(dataset) * (1 - validation_split))
            dataset = dataset.select(range(end_idx))

        print(f"-[INFO] data_loading.py/load_cot_dataset : Dataset loaded successfully. Length: {len(dataset)}")
        return dataset
    except Exception as e:
        print(f"-[ERROR] data_loading.py/load_cot_dataset : Error loading dataset: {e}")
        return None

def prepare_dataloader(dataset, tokenizer, batch_size, shuffle=True, max_length=512):
  """
  Prepares a DataLoader for training.

    Args:
        dataset: The loaded Hugging Face Dataset.
        tokenizer: The tokenizer to use.
        batch_size: The batch size for training.
        shuffle: Whether to shuffle the dataset.
        max_length: The max length for the tokenizer.

  Returns:
      A torch DataLoader
  """
  print(f"-[INFO] data_loading.py/prepare_dataloader : Preparing dataloader with batch size {batch_size}")
  from torch.utils.data import DataLoader
  from src.utils import format_cot_input

  def collate_fn(batch):
    # formats each example into the correct format
    formatted_batch = [format_cot_input(example, tokenizer, max_length) for example in batch]

    # initialize a dictionary for the batch
    collated_batch = {
      "input_ids": torch.cat([example["input_ids"] for example in formatted_batch]),
      "attention_mask": torch.cat([example["attention_mask"] for example in formatted_batch])
    }
    return collated_batch

  dataloader = DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=collate_fn)

  print(f"-[INFO] data_loading.py/prepare_dataloader : Dataloader prepared successfully. Length: {len(dataloader)}")
  return dataloader


if __name__ == "__main__":
    # Example usage (when running this file directly)
    dataset = load_cot_dataset()
    if dataset:
        print("-[INFO] data_loading.py : Example dataset loaded successfully.")
    else:
        print("-[ERROR] data_loading.py : Example dataset failed to load.")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    dataloader = prepare_dataloader(dataset, tokenizer, batch_size=1, shuffle=True, max_length=512)
    print("-[INFO] data_loading.py : Example data loader created with batch_size 16")
    for batch in dataloader:
        print("-[DEBUG] data_loading.py : Example data loader batch")
        print("-[DEBUG] data_loading.py : input_ids shape: ", batch["input_ids"].shape)
        print("-[DEBUG] data_loading.py : attention_mask shape: ", batch["attention_mask"].shape)
        break
