import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import logging
import os
from src.data_processing import get_dataloaders
from src.utils import setup_logging
from accelerate import Accelerator
from sklearn.metrics import accuracy_score

logger = setup_logging("logs", logging.INFO)

def train_response_generator(model_name="meta-llama/Llama-3.2-3B-Instruct", data_path = "kaist-ai/CoT-Collection", batch_size=16, learning_rate=2e-5, num_epochs=3, max_length=512, warmup_steps=100,gradient_accumulation_steps=1):
    """Fine-tunes the Response Generator (Model 2)."""
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    logger.info(f"Starting training for Response Generator with model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_dir = "models/response_generator"
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _, model2_dataloader = get_dataloaders(data_path, model_name, batch_size=batch_size, max_length=max_length)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(model2_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    model, optimizer, model2_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, model2_dataloader, lr_scheduler)

    writer = SummaryWriter(log_dir="logs/tensorboard")
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    global_step=0

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(model2_dataloader):
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            epoch_loss += loss.detach().float()
            accelerator.backward(loss)
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step+=1
                writer.add_scalar('loss/train', loss.detach().float().item(), global_step=global_step)
                progress_bar.update(1)
                if accelerator.is_local_main_process:
                    logger.info(f"[TRAIN] Epoch {epoch + 1} | Loss: {loss.detach().float().item():.2f} | Step: {global_step}")
        epoch_loss /= len(model2_dataloader)
        logger.info(f"[TRAIN] Epoch {epoch + 1} | Average loss: {epoch_loss:.2f}")

    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).save_pretrained(f"{model_dir}/finetuned_response_generator", save_function=accelerator.save)
    logger.info(f"Response Generator model saved to {model_dir}/finetuned_response_generator")

    writer.close()

if __name__ == '__main__':
    train_response_generator()
