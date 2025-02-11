# src/trainer.py
from src.data_loading import load_cot_dataset, prepare_dataloader
from src.model_loading import load_model_and_tokenizer
from src.evaluation import evaluate
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
import os
from accelerate import Accelerator
print("-[DEBUG] trainer.py : trainer module imported")


def train(model, tokenizer, train_dataloader, optimizer, num_epochs, output_dir, eval_every=100, small_eval_size=200):
  """
    Fine-tunes the model on the training dataset.

    Args:
        model: The model to be fine-tuned.
        tokenizer: The tokenizer to use.
        train_dataloader: The training dataloader.
        optimizer: The optimizer to use.
        num_epochs: Number of epochs to train for.
        output_dir: The directory to save the model to.
        eval_every: Evaluate every this many training steps.
        small_eval_size: The size of the evaluation subset.
  """
  print("-[INFO] trainer.py/train : Starting training")

  accelerator = Accelerator() # init accelerator object

  model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader) # prepare the model, optimizer and dataloader

  if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"-[INFO] trainer.py/train : Created output directory: {output_dir}")
  else:
      print(f"-[INFO] trainer.py/train : Output directory already exists: {output_dir}")

  train_steps = 0

  for epoch in range(num_epochs):
    print(f"-[INFO] trainer.py/train : Starting Epoch: {epoch + 1}")
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_main_process):
      # forward pass
      outputs = model(**batch, labels=batch["input_ids"]) # use the batch dictionary

      loss = outputs.loss
      # print(f"-[DEBUG] trainer.py/train : Batch loss: {loss.item()}")

      # backward pass
      accelerator.backward(loss)
      optimizer.step()
      optimizer.zero_grad()

      train_steps += 1
      # Evaluate every eval_every steps
    if train_steps % eval_every == 0:
        print(f"-[INFO] trainer.py/train : Running Evaluation at training step {train_steps}")
        # Load validation split from train data
        eval_dataset = load_cot_dataset(split="validation")  # Use same fraction as training
        if not eval_dataset:
            print("-[ERROR] trainer.py/train : Error loading evaluation dataset")
            continue
        # get a small subset for evaluation
        small_eval_dataset = eval_dataset.select(range(small_eval_size))
        # prepare dataloader
        eval_dataloader = prepare_dataloader(small_eval_dataset, tokenizer, batch_size=1, shuffle=False)
        eval_dataloader = accelerator.prepare(eval_dataloader) # prepare eval dataloader

        # evaluate and get the accuracy
        accuracy = evaluate(model, tokenizer, eval_dataloader, output_dir, accelerator)
        # save the metric value in a txt file
        if accelerator.is_main_process:
          with open(os.path.join(output_dir, f"accuracy_step_{train_steps}.txt"), "w") as f:
            f.write(f"{accuracy['accuracy']}")
        print(f"-[INFO] trainer.py/train : Evaluation at step {train_steps} complete. Accuracy: {accuracy['accuracy']}")

    print(f"-[INFO] trainer.py/train : Epoch {epoch+1} complete.")

  # Save the model at end of the training.
  if accelerator.is_main_process:
    try:
        print(f"-[INFO] trainer.py/train : Saving the model to: {output_dir}")
        accelerator.save_model(model, output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"-[INFO] trainer.py/train : Model saved successfully.")
    except Exception as e:
        print(f"-[ERROR] trainer.py/train : Error saving the model: {e}")

  print("-[INFO] trainer.py/train : Training complete.")


if __name__ == "__main__":
    # Example usage when running this file directly
    print("-[INFO] trainer.py : Running as main")
    dataset = load_cot_dataset()
    if dataset:
        print("-[INFO] trainer.py : Dataset loaded successfully.")
    else:
        print("-[ERROR] trainer.py : Dataset failed to load.")
        exit()

    model, tokenizer = load_model_and_tokenizer()
    if model and tokenizer:
        print("-[INFO] trainer.py : Model and tokenizer loaded successfully.")
    else:
        print("-[ERROR] trainer.py : Model and/or tokenizer failed to load.")
        exit()

    train_dataloader = prepare_dataloader(dataset, tokenizer, batch_size=1, shuffle=True, max_length=512)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1
    output_dir = "./trained_model"

    train(model, tokenizer, train_dataloader, optimizer, num_epochs, output_dir)
