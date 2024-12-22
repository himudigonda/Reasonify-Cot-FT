
# CoTBros Project

This project implements a two-model system for Chain-of-Thought (CoT) reasoning and response generation, as part of an academic research project. The system leverages the LLaMA-3.2-3B-Instruct model, fine-tuning it into two separate models:

1. **CoT Generator:** Generates a step-by-step reasoning path for a given query.
2. **Response Generator:** Produces a final response using the query and the generated reasoning path.

## Directory Structure

```

├── data/
│   ├── raw/             # Raw dataset (downloaded)
│   ├── processed/       # Preprocessed dataset for both models
├── models/
│   ├── cot_generator/   # Fine-tuned CoT Generator model weights
│   ├── response_generator/ # Fine-tuned Response Generator model weights
│   └── pretrained_model # Model weights from HuggingFace hub
├── logs/                # Logs for training and inference, TensorBoard logs
├── src/
│   ├── data_processing.py # Data preprocessing script, creates dataset and dataloader
│   ├── cot_generator_trainer.py # Training script for CoT Generator
│   ├── response_generator_trainer.py # Training script for Response Generator
│   ├── inference.py     # Inference script
│   ├── utils.py       # Utility functions (logging)
├── scripts/
│   ├── setup.sh        # Script to set up the project
│   ├── download_data.sh # Script to download dataset and models
│   ├── train_cot_generator.sh # Script to train CoT Generator
│   ├── train_response_generator.sh # Script to train Response Generator
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
```

## Setup

1.  **Create Project Structure:**
    ```bash
    chmod +x scripts/setup.sh
    ./scripts/setup.sh
    ```

2.  **Create Mamba Environment:**
    ```bash
    mamba create -n cotbros python=3.10 -y
    mamba activate cotbros
    pip3 install -r requirements.txt
    ```
3.  **Download Dataset and Models:**
    ```bash
    chmod +x scripts/download_data.sh
    ./scripts/download_data.sh
    ```

## Training
4. **Train the CoT Generator Model:**
   ```bash
   chmod +x scripts/train_cot_generator.sh
   ./scripts/train_cot_generator.sh
   ```
5. **Train the Response Generator Model:**
    ```bash
    chmod +x scripts/train_response_generator.sh
    ./scripts/train_response_generator.sh
    ```

## Inference
6.  **Run Inference:**
   ```bash
    python3 src/inference.py
   ```
## TensorBoard
6.  **View TensorBoard:**
    ```bash
    tensorboard --logdir logs/tensorboard
    ```

## Important Notes
*  **Logging**: All training and inference logs are created in `logs` folder.
*  **Tensorboard:** The tensorboard files are also created within `logs`.
*  **Models:** The models are downloaded within `models` folder.
*  **Data**: The data is downloaded within `data` folder.


**Step 3: How to Run**

1.  **Create and Navigate to the Project Directory:**
    ```bash
    mkdir CoTBros
    cd CoTBros
    ```
2.  **Run setup script:**
    ```bash
    chmod +x scripts/setup.sh
    ./scripts/setup.sh
    ```
3.  **Create and Activate Mamba Environment:**
    ```bash
    mamba create -n cotbros python=3.10 -y
    mamba activate cotbros
    pip3 install -r requirements.txt
    ```
4.  **Download Data and Models:**
    ```bash
    chmod +x scripts/download_data.sh
    ./scripts/download_data.sh
    ```
5.  **Train CoT Generator Model:**
    ```bash
    chmod +x scripts/train_cot_generator.sh
    ./scripts/train_cot_generator.sh
    ```
6.  **Train Response Generator Model:**
    ```bash
    chmod +x scripts/train_response_generator.sh
    ./scripts/train_response_generator.sh
    ```

7. **Run Inference:**
    ```bash
    python3 src/inference.py
    ```

8.  **View TensorBoard:** In a separate terminal, run:
    ```bash
    tensorboard --logdir logs/tensorboard
    ```
