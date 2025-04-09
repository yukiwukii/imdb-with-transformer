import os
import time
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    BartForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from huggingface_hub import login
import datetime

# Load your HF token from .env file
load_dotenv()
token = os.getenv("HF_TOKEN")
login(token)

def format_time(seconds):
    """Format time in a human-readable way"""
    return str(datetime.timedelta(seconds=int(seconds)))

def setup_directories(model_name):
    """Create all necessary directories for the model"""
    # Create directories
    MODEL_DIR = f"./models/{model_name}/full" 
    FINAL_MODEL_PATH = f"{MODEL_DIR}/final_model"
    LOGS_DIR = f"./logs/{model_name}/full"
    TENSORBOARD_PATH = f"{LOGS_DIR}/tensorboard"
    FIGURES_DIR = f"{LOGS_DIR}/figures" 

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FINAL_MODEL_PATH, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_PATH, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    return {
        "MODEL_DIR": MODEL_DIR,
        "FINAL_MODEL_PATH": FINAL_MODEL_PATH,
        "LOGS_DIR": LOGS_DIR,
        "TENSORBOARD_PATH": TENSORBOARD_PATH,
        "FIGURES_DIR": FIGURES_DIR
    }

def load_and_prepare_data(model_name, max_length=512):
    """Load and prepare the IMDB dataset with the specified model's tokenizer"""
    imdb = load_dataset("imdb")
    
    # Use the specified model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained("yukiwuki/IMDB_BART")
    
    # Define the tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Tokenize dataset
    tokenized_imdb = imdb.map(tokenize_function, batched=True)
    
    # Prepare the dataset with the right format
    tokenized_imdb = tokenized_imdb.remove_columns(["text"])
    tokenized_imdb = tokenized_imdb.rename_column("label", "labels")
    tokenized_imdb.set_format("torch")
    
    # Prepare data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Split train/val/test set
    train_val_split = tokenized_imdb["train"].train_test_split(test_size=0.2)
    train_split = train_val_split["train"]
    val_split = train_val_split["test"]
    test_split = tokenized_imdb["test"]
    
    return {
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split
    }

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    # Check the actual shape/type of predictions to handle properly
    if isinstance(predictions, tuple):
        # Some models return a tuple, use the first element (logits)
        predictions = predictions[0]
    
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def get_model(model_name):
    """Get the model configured for sequence classification"""
    # For BART, we use BartForSequenceClassification
    model = BartForSequenceClassification.from_pretrained(
        "yukiwuki/IMDB_BART", 
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    
    return model

def generate_confusion_matrix(predictions, true_labels, save_path):
    """Generate and save a confusion matrix plot"""
    cm = confusion_matrix(true_labels, predictions)
    
    # Create a pretty confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["NEGATIVE", "POSITIVE"],
                yticklabels=["NEGATIVE", "POSITIVE"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the confusion matrix plot
    plt.savefig(save_path)
    plt.close()
    
    return cm

def run_finetuning(model_name):
    """Run the full finetuning process for the specified model"""
    print(f"\n{'='*80}\nStarting full finetuning of {model_name}\n{'='*80}")
    
    # Setup directories with model name
    dirs = setup_directories(model_name)
    
    # Load and prepare data with the specified model
    data = load_and_prepare_data(model_name)

    model = get_model(model_name)
    
    # Set up the evaluation trainer
    eval_trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"{dirs['LOGS_DIR']}/eval_results",
            per_device_eval_batch_size=16,
            report_to="none"
        ),
        tokenizer=data["tokenizer"],
        eval_dataset=data["test_split"],
        data_collator=data["data_collator"],
        compute_metrics=compute_metrics
    )
    
    # Combining evaluation and prediction in a single pass
    print("Running prediction and evaluation")
    test_predictions = eval_trainer.predict(data["test_split"])
    predictions = np.argmax(test_predictions.predictions, axis=1)
    true_labels = test_predictions.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    acc = accuracy_score(true_labels, predictions)
    eval_results = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    # Generate and save confusion matrix
    cm_path = f"{dirs['FIGURES_DIR']}/confusion_matrix.png"
    cm = generate_confusion_matrix(predictions, true_labels, cm_path)
    
    # Log the evaluation results including confusion matrix
    with open(f"{dirs['LOGS_DIR']}/test_results.txt", 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Finetuning method: full\n")
            f.write(f"Evaluation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Test metrics:\n")
            for metric_name, value in eval_results.items():
                f.write(f"{metric_name}: {value}\n")
            
            f.write("\nConfusion Matrix:\n")
            f.write(f"Saved to: {cm_path}\n\n")
            f.write("Raw confusion matrix values:\n")
            f.write(str(cm))
    
    print(f"Evaluation results for {model_name}: {eval_results}")
    print(f"Confusion matrix saved to: {cm_path}")
    
    return eval_results

def main():
    model_name = "bart-eval"
    
    print(f"Starting full finetuning for model: {model_name}")
    
    # Run finetuning
    results = run_finetuning(model_name)
    
    print(f"\nFull finetuning completed for model {model_name}.")
    print(f"Results: {results}")

if __name__ == "__main__":
    main()