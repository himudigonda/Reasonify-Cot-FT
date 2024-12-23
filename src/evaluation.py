# src/evaluation.py
from src.data_loading import load_cot_dataset, prepare_dataloader
from src.model_loading import load_model_and_tokenizer
from src.utils import extract_answer
from tqdm import tqdm
import torch
from torch.optim import AdamW
from evaluate import load
import os
print("-[DEBUG] evaluation.py : evaluation module imported")

def evaluate(model, tokenizer, eval_dataloader, output_dir, device):
  """
  Evaluates the model on the given dataset and calculates the accuracy.

  Args:
    model: The model to be evaluated.
    tokenizer: The tokenizer to use.
    eval_dataloader: The dataloader for evaluation data.
    output_dir: directory to save the model
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

  for batch in tqdm(eval_dataloader, desc=f"Evaluating"):
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # generating output from the model
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=256)

        # extract the answer and decode the input id
        predictions = [extract_answer(output, tokenizer) for output in outputs]

        # decode all input ids
        references = [tokenizer.decode(input_ids[i], skip_special_tokens=True) for i in range(len(input_ids))]

        # accumulate the results
        all_predictions.extend(predictions)
        all_references.extend(references)

        total_predictions += len(predictions)

  # calculate accuracy
  accuracy_metric.add_batch(predictions=all_predictions, references=all_references)
  accuracy = accuracy_metric.compute()
  print(f"-[INFO] evaluation.py/evaluate : Evaluation complete. Accuracy: {accuracy['accuracy'] * 100:.2f}%")
  return accuracy


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
    gpu_devices = get_gpu_info()
    device = "cuda" if gpu_devices else "cpu"
    output_dir = "./evaluated_model"
    if os.path.exists("./trained_model"):
      model.load_pretrained("./trained_model")
      print("-[INFO] evaluation.py : Loaded saved model")

    accuracy = evaluate(model, tokenizer, eval_dataloader, output_dir, device)
    print(f"-[INFO] evaluation.py : The accuracy of model is {accuracy}")
