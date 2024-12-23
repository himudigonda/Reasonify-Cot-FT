# src/evaluation.py
from src.data_loading import load_cot_dataset, prepare_dataloader
from src.model_loading import load_model_and_tokenizer
from src.utils import extract_answer
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from evaluate import load
import os
from accelerate import Accelerator
print("-[DEBUG] evaluation.py : evaluation module imported")

def evaluate(model, tokenizer, eval_dataloader, output_dir, accelerator):
  """
  Evaluates the model on the given dataset and calculates the accuracy.

  Args:
    model: The model to be evaluated.
    tokenizer: The tokenizer to use.
    eval_dataloader: The dataloader for evaluation data.
    output_dir: directory to save the model
    accelerator: The accelerator object.
  """
  print("-[INFO] evaluation.py/evaluate : Starting evaluation")

  model.eval() # sets model to evaluation mode
  print(f"-[INFO] evaluation.py/evaluate : Model set to eval mode")

  if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"-[INFO] evaluation.py/evaluate : Created output directory: {output_dir}")
  else:
      print(f"-[INFO] evaluation.py/evaluate : Output directory already exists: {output_dir}")

  accuracy_metric = load("accuracy")
  total_predictions = 0
  all_predictions = []
  all_references = []

  for batch in tqdm(eval_dataloader, desc=f"Evaluating", disable=not accelerator.is_main_process):
    with torch.no_grad():
        outputs = model.generate(**batch, max_new_tokens=256) # use the batch dictionary

        # extract the answer and decode the input id
        predictions = [extract_answer(output, tokenizer) for output in outputs]

        # decode all input ids
        references = [tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True) for i in range(len(batch["input_ids"]))]

        # accumulate the results
        all_predictions.extend(predictions)
        all_references.extend(references)

        total_predictions += len(predictions)

  # gather results across all GPUs
  all_predictions = accelerator.gather(all_predictions)
  all_references = accelerator.gather(all_references)

  # calculate accuracy on rank 0
  if accelerator.is_main_process:
      accuracy_metric.add_batch(predictions=all_predictions, references=all_references)
      accuracy = accuracy_metric.compute()
      print(f"-[INFO] evaluation.py/evaluate : Evaluation complete. Accuracy: {accuracy['accuracy'] * 100:.2f}%")
      return accuracy
  else:
        return {"accuracy": 0.0}


if __name__ == "__main__":
    # Example usage when running this file directly
    print("-[INFO] evaluation.py : Running as main")
    dataset = load_cot_dataset()
    if dataset:
        print("-[INFO] evaluation.py : Dataset loaded successfully.")
    else:
        print("-[ERROR] evaluation.py : Dataset failed to load.")
        exit()

    model, tokenizer = load_model_and_tokenizer()
    if model and tokenizer:
        print("-[INFO] evaluation.py : Model and tokenizer loaded successfully.")
    else:
        print("-[ERROR] evaluation.py : Model and/or tokenizer failed to load.")
        exit()

    eval_dataloader = prepare_dataloader(dataset, tokenizer, batch_size=16, shuffle=False, max_length=512)
    accelerator = Accelerator()
    eval_dataloader = accelerator.prepare(eval_dataloader)
    output_dir = "./evaluated_model"
    if os.path.exists("./trained_model"):
      accelerator.load_model(model, "./trained_model")
      print("-[INFO] evaluation.py : Loaded saved model")

    accuracy = evaluate(model, tokenizer, eval_dataloader, output_dir, accelerator)
    if accelerator.is_main_process:
        print(f"-[INFO] evaluation.py : The accuracy of model is {accuracy}")
