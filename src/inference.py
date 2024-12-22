import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import setup_logging
import logging

logger = setup_logging("logs", logging.INFO)

def run_inference(
    cot_generator_model_path="models/cot_generator/finetuned_cot_generator",
    response_generator_model_path="models/response_generator/finetuned_response_generator",
    model_name = "meta-llama/Llama-3.2-3B-Instruct",
    test_queries=None,
    max_length=512,
    ):
    """Runs inference using the fine-tuned models."""
    if test_queries is None:
        test_queries = [
          "What is the capital of France?",
          "Summarize the plot of Hamlet.",
          "What are the health benefits of eating spinach?",
           "Solve the equation: 2 * (5 + 3) - 10",
        ]
        logger.info(f"Using default queries since none was provided:{test_queries}")

    logger.info(f"Loading CoT Generator model from: {cot_generator_model_path}")
    cot_generator_model = AutoModelForCausalLM.from_pretrained(cot_generator_model_path)
    cot_generator_tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Loading Response Generator model from: {response_generator_model_path}")
    response_generator_model = AutoModelForCausalLM.from_pretrained(response_generator_model_path)
    response_generator_tokenizer = AutoTokenizer.from_pretrained(model_name)

    for query in test_queries:
        logger.info(f"Input Query: {query}")

        # Generate CoT
        cot_generator_inputs = cot_generator_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        cot_generator_outputs = cot_generator_model.generate(**cot_generator_inputs, max_new_tokens=max_length)
        cot = cot_generator_tokenizer.decode(cot_generator_outputs[0], skip_special_tokens=True)
        logger.info(f"Generated Chain of Thought: {cot}")


        # Generate Response
        combined_input = f"Query: {query} Reason: {cot}"
        response_generator_inputs = response_generator_tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        response_generator_outputs = response_generator_model.generate(**response_generator_inputs, max_new_tokens=max_length)
        response = response_generator_tokenizer.decode(response_generator_outputs[0], skip_special_tokens=True)
        logger.info(f"Final Response: {response}")

    logger.info("Inference complete.")

if __name__ == '__main__':
    run_inference()
