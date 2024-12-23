# src/model_loading.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import get_gpu_info, get_model_size
import torch
print("-[DEBUG] model_loading.py : model_loading module imported")

def load_model_and_tokenizer(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    """Loads the pre-trained model and tokenizer."""
    print(f"-[INFO] model_loading.py/load_model_and_tokenizer : Loading model: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"-[INFO] model_loading.py/load_model_and_tokenizer : Tokenizer loaded.")

        # Check if the tokenizer already has a pad token
        if tokenizer.pad_token is None:
          print(f"-[INFO] model_loading.py/load_model_and_tokenizer : Tokenizer does not have pad token, setting it to eos_token")
          tokenizer.pad_token = tokenizer.eos_token
        else:
          print(f"-[INFO] model_loading.py/load_model_and_tokenizer : Tokenizer already has pad token, value is: {tokenizer.pad_token}")

        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     low_cpu_mem_usage=True,
                                                     torch_dtype=torch.bfloat16,
                                                     )

        print(f"-[INFO] model_loading.py/load_model_and_tokenizer : Model loaded.")
        gpu_devices = get_gpu_info()
        if gpu_devices:
          print(f"-[INFO] model_loading.py/load_model_and_tokenizer : Moving model to GPU")
          model.to("cuda")

        model_size = get_model_size(model)
        print(f"-[INFO] model_loading.py/load_model_and_tokenizer : Model Size : {model_size}")
        return model, tokenizer
    except Exception as e:
        print(f"-[ERROR] model_loading.py/load_model_and_tokenizer : Error loading model/tokenizer: {e}")
        return None, None

if __name__ == "__main__":
    # Example usage (when running this file directly)
    model, tokenizer = load_model_and_tokenizer()
    if model and tokenizer:
        print("-[INFO] model_loading.py : Example model and tokenizer loaded successfully.")
    else:
        print("-[ERROR] model_loading.py : Example model and/or tokenizer failed to load.")
