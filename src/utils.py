# src/utils.py

import torch
import os
print("-[DEBUG] utils.py : utils module imported")

def format_cot_input(example, tokenizer, max_length=512):
    """Formats the input for Chain-of-Thought training."""
    # print(f"-[DEBUG] utils.py/format_cot_input : Formatting input for example: {example}")

    prompt = f"{example['source']}\n\n{example['rationale']}\n[RESULT]\n{example['target']}"

    # print(f"-[DEBUG] utils.py/format_cot_input : Formatted prompt:\n{prompt}")

    tokenized_input = tokenizer(prompt,
                              truncation=True,
                              max_length=max_length,
                              padding="max_length",
                              return_tensors="pt")

    # print(f"-[DEBUG] utils.py/format_cot_input : Tokenized input keys: {tokenized_input.keys()}")

    return tokenized_input


def extract_answer(model_output, tokenizer, result_token="[RESULT]"):
    """Extracts the generated answer from the model's output."""
    # print(f"-[DEBUG] utils.py/extract_answer : Starting answer extraction")

    result_ids = tokenizer.encode(result_token, add_special_tokens=False)
    # print(f"-[DEBUG] utils.py/extract_answer : result_ids are: {result_ids}")


    decoded_output = tokenizer.decode(model_output, skip_special_tokens=True)
    # print(f"-[DEBUG] utils.py/extract_answer : decoded_output: {decoded_output}")

    try:

        result_index = decoded_output.find(result_token)
        if result_index != -1:
            extracted_text = decoded_output[result_index + len(result_token) :].strip()
            # print(f"-[DEBUG] utils.py/extract_answer : Extracted answer: {extracted_text}")
            return extracted_text
        else:
          # print(f"-[DEBUG] utils.py/extract_answer : Result token not found, returning the full output")
          return decoded_output
    except Exception as e:
        print(f"-[ERROR] utils.py/extract_answer : Error during answer extraction: {e}")
        return None

def get_gpu_info():
    """
    Gets GPU device information and returns it
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        print(f"-[INFO] utils.py/get_gpu_info : GPU is available: {device_names}")
        return device_names
    else:
        print("-[INFO] utils.py/get_gpu_info : GPU is not available")
        return None

def get_model_size(model):
    """Returns the size of the model in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    print(f"-[INFO] utils.py/get_model_size : Model Size is: {size_mb:.2f} MB")
    return size_mb
