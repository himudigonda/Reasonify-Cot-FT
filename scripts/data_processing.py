import json
import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import load_config, add_padding_token

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

    # Load the tokenizer (load once)
    tokenizer = AutoTokenizer.from_pretrained(model1_config["model_name"])
    add_padding_token(tokenizer)

    def tokenize_function(examples):
        """Tokenizes a batch of texts using the tokenizer."""
        source_tokens = tokenizer(examples["source"], padding="max_length", truncation=True, max_length=model1_config["max_length"])
        target_tokens = tokenizer(examples["target"], padding="max_length", truncation=True, max_length=model1_config["max_length"])
        rationale_tokens = tokenizer(examples["rationale"], padding="max_length", truncation=True, max_length=model1_config["max_length"])
        return {
            "input_ids": source_tokens["input_ids"],
            "labels": target_tokens["input_ids"],
            "rationale_ids": rationale_tokens["input_ids"]
            }

    def tokenize_and_split(df, sample_size_percent):
      """Tokenizes data and creates train/val/test splits."""
      print(f"-[info] data_processing.tokenize_and_split : Starting tokenization and splitting")

      # Sample the DataFrame
      if sample_size_percent < 1.0:
          df = df.sample(frac=sample_size_percent, random_state=42)
          print(f"-[info] data_processing.tokenize_and_split : Reduced dataset to {sample_size_percent*100:.2f} %")

      # Calculate split sizes
      total_rows = len(df)
      test_size = int(total_rows * params["test_size"])
      val_size = int(total_rows * params["val_size"])
      train_size = total_rows - test_size - val_size

       # Split using iloc
      test_df = df.iloc[:test_size]
      val_df = df.iloc[test_size:test_size + val_size]
      train_df = df.iloc[test_size + val_size:]
      print(f"-[debug] data_processing.tokenize_and_split : Created train, test and validation splits")

      print

      # Tokenize each dataset split
      train_tokenized = train_df.apply(lambda x: tokenize_function(x), axis=1, result_type='expand')
      print(f"-[debug] data_processing.tokenize_and_split : Tokenized train split")
      val_tokenized = val_df.apply(lambda x: tokenize_function(x), axis=1, result_type='expand')
      print(f"-[debug] data_processing.tokenize_and_split : Tokenized val split")
      test_tokenized = test_df.apply(lambda x: tokenize_function(x), axis=1, result_type='expand')
      print(f"-[debug] data_processing.tokenize_and_split : Tokenized test split")


      train = train_tokenized.rename(columns={"input_ids": "input_ids", "labels": "labels", "rationale_ids": "rationale_ids"})
      val = val_tokenized.rename(columns={"input_ids": "input_ids", "labels": "labels", "rationale_ids": "rationale_ids"})
      test = test_tokenized.rename(columns={"input_ids": "input_ids", "labels": "labels", "rationale_ids": "rationale_ids"})
      print(f"-[debug] data_processing.tokenize_and_split : Renamed columns")

      print(f"-[info] data_processing.tokenize_and_split : Tokenization and splitting completed successfully")
      return train, val, test

    def save_splits(train_data, val_data, test_data, output_dir):
        """Saves the train/val/test splits in the specified output directory"""
        print(f"-[info] data_processing.save_splits : Starting saving of splits")
        os.makedirs(output_dir, exist_ok=True)
        print(f"-[debug] data_processing.save_splits : Created output directories")

        for split_name, data in zip(["train", "val", "test"], [train_data, val_data, test_data]):
            filename = os.path.join(output_dir, f"{split_name}.parquet")
            df = pd.DataFrame(data)
            print(f"-[debug] data_processing.save_splits : Saving {split_name} split to {filename}")
            df.to_parquet(filename)
            print(f"-[debug] data_processing.save_splits : Saved {split_name} split")

        print(f"-[info] data_processing.save_splits : Saving of splits completed")

    print(f"-[info] data_processing.main : Starting Data Processing")

    # Load the dataset
    print(f"-[info] data_processing.main : Loading dataset")
    dataset = load_dataset("kaist-ai/CoT-Collection", trust_remote_code=True)
    dataset = dataset["train"]
    print(f"-[info] data_processing.main : Dataset loaded successfully")

    # Convert the dataset to a Pandas DataFrame
    print(f"-[debug] data_processing.main : Converting dataset to Pandas DataFrame")
    df = pd.DataFrame(dataset.to_pandas())
    print(f"-[debug] data_processing.main : Successfully converted dataset to Pandas DataFrame")

    # Tokenize and Split the data
    sample_size = params.get("sample_size", 1.0)
    train, val, test = tokenize_and_split(df, sample_size)

    # Save the splits
    save_splits(train, val, test, "data/splits/")
    print(f"-[info] data_processing.main : Data tokenized and split successfully.")
    print(f"-[info] data_processing.main : Completed Data Processing")

if __name__ == '__main__':
  main()
