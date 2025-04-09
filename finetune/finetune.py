import os
import time
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
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

# Load your HF token from .env file
load_dotenv()
token = os.getenv("HF_TOKEN")
login(token)

# Model to finetune
model_name = "hf_og"

# Create directories with clear separation
MODEL_DIR = f"./models/{model_name}"  # For model checkpoints (can be gitignored)
FINAL_MODEL_PATH = f"{MODEL_DIR}/final_model"
LOGS_DIR = f"./logs/{model_name}"  # For logs (can be committed to git)
TENSORBOARD_PATH = f"{LOGS_DIR}/tensorboard"
FIGURES_DIR = f"{LOGS_DIR}/figures"  # For saving confusion matrix plot

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_PATH, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_PATH, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

imdb = load_dataset("imdb")

# Use BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Tokenize dataset
tokenized_imdb = imdb.map(lambda e: tokenizer(e["text"], truncation=True, padding=True), batched=True)

# Prepare data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Split train/val/test set
train_val_split = tokenized_imdb["train"].train_test_split(test_size=0.2)
train_split = train_val_split["train"]
val_split = train_val_split["test"]
test_split = tokenized_imdb["test"]

# Define metrics computation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Define model initialization function
def model_init():
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2, id2label=id2label, label2id=label2id
    )
    return model

# Function to generate and save confusion matrix
def generate_confusion_matrix(predictions, true_labels, save_path):
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

# Main function
def main():
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,  # Checkpoints go here (can be gitignored)
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        # TensorBoard configuration - logs go here (can be committed)
        logging_dir=TENSORBOARD_PATH,
        logging_strategy="steps",
        logging_steps=100,
        report_to=["tensorboard"],
        # Track all metrics
        metric_for_best_model="f1",
    )

    # Initialize trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=val_split,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    model = None
    
    # Check if there's an existing checkpoint in the final model directory
    if os.path.exists(FINAL_MODEL_PATH) and any(os.listdir(FINAL_MODEL_PATH)):
        print(f"Loading fine-tuned model from: {FINAL_MODEL_PATH}")
        model = AutoModelForSequenceClassification.from_pretrained(
            FINAL_MODEL_PATH,
            num_labels=2
        )
        
        # Record that we're using a pre-trained model
        with open(f"{LOGS_DIR}/model_info.txt", 'w') as f:
            f.write(f"Used pre-trained model from: {FINAL_MODEL_PATH}\n")
            f.write(f"Model loaded at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    else:
        print("Fine-tuned model does not exist. Fine-tuning now.")
        
        # Start timing the training
        start_time = time.time()
        
        # Run training
        train_result = trainer.train()
        
        # Calculate training time
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Log training metrics and time
        with open(f"{LOGS_DIR}/training_results.txt", 'w') as f:
            f.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n\n")
            f.write("Training metrics:\n")
            for key, value in train_result.metrics.items():
                f.write(f"{key}: {value}\n")
        
        # Save the best model to the final_model directory
        trainer.save_model(FINAL_MODEL_PATH)
        model = trainer.model
        print(f"Best model saved to: {FINAL_MODEL_PATH}")

    # Create evaluation arguments
    eval_args = TrainingArguments(
        output_dir=f"{LOGS_DIR}/eval_results",  # Evaluation results go to logs
        per_device_eval_batch_size=16,
        evaluation_strategy="no",
        save_strategy="no",
        report_to=["tensorboard"],
        logging_dir=TENSORBOARD_PATH,
    )

    # Set up the evaluation trainer
    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        eval_dataset=test_split,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Run evaluation
    eval_results = eval_trainer.evaluate()
    
    # Get predictions for confusion matrix
    test_predictions = eval_trainer.predict(test_split)
    predictions = np.argmax(test_predictions.predictions, axis=1)
    true_labels = test_predictions.label_ids
    
    # Generate and save confusion matrix
    cm_path = f"{FIGURES_DIR}/confusion_matrix.png"
    cm = generate_confusion_matrix(predictions, true_labels, cm_path)
    
    # Log the evaluation results including confusion matrix
    with open(f"{LOGS_DIR}/test_results.txt", 'w') as f:
        f.write(f"Evaluation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Test metrics:\n")
        for metric_name, value in eval_results.items():
            f.write(f"{metric_name}: {value}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(f"Saved to: {cm_path}\n\n")
        f.write("Raw confusion matrix values:\n")
        f.write(str(cm))
    
    print(f"Evaluation results: {eval_results}")
    print(f"Confusion matrix saved to: {cm_path}")

if __name__ == "__main__":
    main()