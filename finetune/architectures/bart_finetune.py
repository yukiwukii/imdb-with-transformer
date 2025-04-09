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
import gc

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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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
    """Compute evaluation metrics with memory management"""
    predictions, labels = eval_pred
    # Check the actual shape/type of predictions to handle properly
    if isinstance(predictions, tuple):
        # Some models return a tuple, use the first element (logits)
        predictions = predictions[0]
    
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    # Clear memory explicitly
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
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
        model_name, 
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    
    return model

def evaluate_model(model, data, dirs):
    """
    Evaluate model on test data with optimized memory usage
    """
    print(f"Starting model evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Set up the evaluation trainer with smaller batch size
    eval_batch_size = 4  # Reduced batch size
    eval_trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"{dirs['LOGS_DIR']}/eval_results",
            per_device_eval_batch_size=eval_batch_size,
            report_to="none",
            dataloader_drop_last=False  # Ensure we process all examples
        ),
        tokenizer=data["tokenizer"],
        data_collator=data["data_collator"]
    )
    
    # Split test dataset into chunks to avoid OOM
    print("Running prediction with memory optimization")
    chunk_size = 500  # Process smaller chunks at a time
    all_predictions = []
    all_labels = []
    
    # Get total dataset size for progress reporting
    total_examples = len(data["test_split"])
    print(f"Test dataset size: {total_examples} examples")
    
    try:
        for i in range(0, total_examples, chunk_size):
            # Clear memory before processing each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Define the end index for this chunk
            end_idx = min(i + chunk_size, total_examples)
            
            # Select a subset of test data
            test_subset = data["test_split"].select(range(i, end_idx))
            
            # Run prediction with no_grad to save memory
            with torch.no_grad():
                test_predictions = eval_trainer.predict(test_subset)
            
            # Process predictions for this chunk
            predictions = test_predictions.predictions
            if isinstance(predictions, tuple):
                # Some models return a tuple, use the first element (logits)
                predictions = predictions[0]
            chunk_predictions = np.argmax(predictions, axis=1)
            chunk_labels = test_predictions.label_ids
            
            # Store results
            all_predictions.extend(chunk_predictions)
            all_labels.extend(chunk_labels)
            
            # Report progress
            progress = (end_idx / total_examples) * 100
            print(f"Processed examples {i} to {end_idx-1} ({progress:.1f}% complete)")
            
            # Force garbage collection and clear CUDA cache again
            del test_predictions, test_subset, chunk_predictions, chunk_labels
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA OOM detected. Debug information:")
            if torch.cuda.is_available():
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
            # Try to recover by clearing everything and reducing chunk size
            del model, eval_trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Re-raise the exception with additional information
            raise RuntimeError(f"CUDA out of memory during evaluation at example {i}. Try reducing batch size or chunk size.") from e
        else:
            raise  # Re-raise if it's not an OOM error
        
    # Convert to numpy arrays for metric calculation
    predictions = np.array(all_predictions)
    true_labels = np.array(all_labels)
    
    # Calculate metrics
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
    
    # Create a pretty confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["NEGATIVE", "POSITIVE"],
                yticklabels=["NEGATIVE", "POSITIVE"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the confusion matrix plot
    plt.savefig(cm_path)
    plt.close()
    
    # Log the evaluation results including confusion matrix
    with open(f"{dirs['LOGS_DIR']}/test_results.txt", 'w') as f:
            f.write(f"Evaluation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Test metrics:\n")
            for metric_name, value in eval_results.items():
                f.write(f"{metric_name}: {value}\n")
            
            f.write("\nConfusion Matrix:\n")
            f.write(f"Saved to: {cm_path}\n\n")
            f.write("Raw confusion matrix values:\n")
            f.write(str(cm))
    
    print(f"Evaluation completed. Results: {eval_results}")
    print(f"Confusion matrix saved to: {cm_path}")
    
    return eval_results

def run_finetuning(model_name):
    """Run the full finetuning process for the specified model"""
    print(f"\n{'='*80}\nStarting full finetuning of {model_name}\n{'='*80}")
    
    # Setup directories with model name
    dirs = setup_directories(model_name)
    
    # Load and prepare data with the specified model
    data = load_and_prepare_data(model_name)
    
    # Create early stopping callback
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=dirs["MODEL_DIR"],
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_dir=dirs["TENSORBOARD_PATH"],
        logging_strategy="steps",
        logging_steps=100,
        report_to=["tensorboard"],
        metric_for_best_model="f1",
    )
    
    # Check if there's an existing checkpoint
    model = None
    if os.path.exists(dirs["FINAL_MODEL_PATH"]) and any(os.listdir(dirs["FINAL_MODEL_PATH"])):
        print(f"Loading fine-tuned model from: {dirs['FINAL_MODEL_PATH']}")
        
        model = BartForSequenceClassification.from_pretrained(
            dirs["FINAL_MODEL_PATH"],
            num_labels=2
        )
            
        # Record that we're using a pre-trained model
        with open(f"{dirs['LOGS_DIR']}/model_info.txt", 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Used pre-trained model from: {dirs['FINAL_MODEL_PATH']}\n")
            f.write(f"Model loaded at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Finetuning method: full\n")
    else:
        print(f"Fine-tuned model for {model_name} does not exist. Fine-tuning now.")
        
        # Initialize model for full finetuning
        model = get_model(model_name)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data["train_split"],
            eval_dataset=data["val_split"],
            tokenizer=data["tokenizer"],
            data_collator=data["data_collator"],
            compute_metrics=compute_metrics,
            callbacks=[early_stopping_callback],
        )
        
        # Time the training
        start_time = time.time()
        
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        
        # Log training metrics and time
        with open(f"{dirs['LOGS_DIR']}/training_results.txt", 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Finetuning method: full\n")
            f.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training time: {format_time(training_time)}\n\n")
            f.write("Training metrics:\n")
            for key, value in train_result.metrics.items():
                f.write(f"{key}: {value}\n")
        
        # Save the best model
        trainer.save_model(dirs["FINAL_MODEL_PATH"])
        model = trainer.model
        print(f"Best model saved to: {dirs['FINAL_MODEL_PATH']}")
    
    # Explicitly clean up trainer to free memory before evaluation
    if 'trainer' in locals():
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Run evaluation with memory optimization
    eval_results = evaluate_model(model, data, dirs)
    
    print(f"\nFull finetuning completed for model {model_name}.")
    print(f"Results: {eval_results}")
    
    return eval_results

def main():
    parser = argparse.ArgumentParser(description='Run full finetuning on BART models')
    parser.add_argument('--model', type=str, default="facebook/bart-base",
                        help='Model name or path from Hugging Face (default: facebook/bart-base)')
    args = parser.parse_args()
    
    model_name = args.model
    
    print(f"Starting full finetuning for model: {model_name}")
    
    # Run finetuning
    results = run_finetuning(model_name)
    
    print(f"\nFull finetuning completed for model {model_name}.")
    print(f"Results: {results}")

if __name__ == "__main__":
    main()