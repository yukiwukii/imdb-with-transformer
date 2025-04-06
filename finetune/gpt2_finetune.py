import os
import time
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    IA3Config,
    TaskType,
    PeftModel
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

# Finetuning methods
METHODS = ["full", "lora", "prefix", "ia3"]

def format_time(seconds):
    """Format time in a human-readable way"""
    return str(datetime.timedelta(seconds=int(seconds)))

def setup_directories(model_name, method_name):
    """Create all necessary directories for a specific model and finetuning method"""
    # Create directories with clear separation
    MODEL_DIR = f"./models/{model_name}/{method_name}"  # For model checkpoints (can be gitignored)
    FINAL_MODEL_PATH = f"{MODEL_DIR}/final_model"
    LOGS_DIR = f"./logs/{model_name}/{method_name}"  # For logs (can be committed to git)
    TENSORBOARD_PATH = f"{LOGS_DIR}/tensorboard"
    FIGURES_DIR = f"{LOGS_DIR}/figures"  # For saving confusion matrix plot

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

def load_and_prepare_data(model_name, max_length=None):
    """Load and prepare the IMDB dataset with the specified model's tokenizer"""
    imdb = load_dataset("imdb")
    
    # Use the specified model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # GPT2 tokenizer doesn't have a padding token by default, so we need to set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use a consistent max_length if not specified
    if max_length is None:
        max_length = 512  # Default for GPT2
    
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
        "test_split": test_split,
        "max_length": max_length
    }

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
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

def get_model_and_hidden_size(model_name):
    """Get the model and determine its hidden size"""
    # For GPT2, we need to use GPT2ForSequenceClassification
    model = GPT2ForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        pad_token_id=50256,  # EOS token for GPT2
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    
    # Get the hidden size from the model config
    config = model.config
    hidden_size = getattr(config, "n_embd", 768)  # GPT2 uses n_embd instead of hidden_size
    
    return model, hidden_size

def get_model_for_method(model_name, method):
    """Get the appropriate model based on the model name and finetuning method"""
    # Get base model and hidden size
    base_model, hidden_size = get_model_and_hidden_size(model_name)
    
    if method == "full":
        # Full finetuning - return the model as is
        return base_model
    
    elif method == "lora":
        # LoRA configuration - adapted for GPT2
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"],  # GPT2 attention modules
            bias="none",
            inference_mode=False
        )
        model = get_peft_model(base_model, peft_config)
        return model
    
    elif method == "prefix":
        # Prefix Tuning configuration - with updated encoder_hidden_size
        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=20,
            prefix_projection=True,
            encoder_hidden_size=hidden_size  # Use the model's actual hidden size
        )
        peft_config.modules_to_save = ["score"]  # GPT2 uses 'score' instead of 'classifier'
        model = get_peft_model(base_model, peft_config)
        return model
    
    elif method == "ia3":
        # IA3 configuration - adapted for GPT2
        peft_config = IA3Config(
            task_type=TaskType.SEQ_CLS,
            target_modules=["c_attn", "c_proj", "c_fc"],  # GPT2 modules
            feedforward_modules=["c_fc"],
            inference_mode=False
        )
        model = get_peft_model(base_model, peft_config)
        return model
    
    else:
        raise ValueError(f"Unknown method: {method}")

def model_init_factory(model_name, method):
    """Create a model_init function for the specified model and method"""
    def model_init():
        return get_model_for_method(model_name, method)
    return model_init

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

def run_finetuning(model_name, method):
    """Run the finetuning process for a specific model and method"""
    print(f"\n{'='*80}\nStarting finetuning of {model_name} with method: {method}\n{'='*80}")
    
    # Setup directories with model name
    dirs = setup_directories(model_name, method)
    
    # Configure max_length based on method
    # Prefix tuning typically needs consistent sequence length
    max_length = 492 if method == "prefix" else 512
    
    # Load and prepare data with the specified model and max_length
    data = load_and_prepare_data(model_name, max_length)
    
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
        
        if method == "full":
            model = GPT2ForSequenceClassification.from_pretrained(
                dirs["FINAL_MODEL_PATH"],
                num_labels=2
            )
        else:
            # For PEFT methods, load using PeftModel.from_pretrained
            base_model, _ = get_model_and_hidden_size(model_name)
            model = PeftModel.from_pretrained(base_model, dirs["FINAL_MODEL_PATH"])
            
        # Record that we're using a pre-trained model
        with open(f"{dirs['LOGS_DIR']}/model_info.txt", 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Used pre-trained model from: {dirs['FINAL_MODEL_PATH']}\n")
            f.write(f"Model loaded at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Finetuning method: {method}\n")
    else:
        print(f"Fine-tuned model for {model_name} with method {method} does not exist. Fine-tuning now.")
        
        # Initialize trainer with the model name
        trainer = Trainer(
            model_init=model_init_factory(model_name, method),
            args=training_args,
            train_dataset=data["train_split"],
            eval_dataset=data["val_split"],
            tokenizer=data["tokenizer"],
            data_collator=data["data_collator"],
            compute_metrics=compute_metrics,
            callbacks=[early_stopping_callback],
        )
        
        # Start timing the training
        start_time = time.time()
        
        # Run training
        train_result = trainer.train()
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Log training metrics and time
        with open(f"{dirs['LOGS_DIR']}/training_results.txt", 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Finetuning method: {method}\n")
            f.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training time: {format_time(training_time)}\n\n")
            f.write("Training metrics:\n")
            for key, value in train_result.metrics.items():
                f.write(f"{key}: {value}\n")
        
        # Save the best model
        trainer.save_model(dirs["FINAL_MODEL_PATH"])
        model = trainer.model
        print(f"Best model saved to: {dirs['FINAL_MODEL_PATH']}")
    
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
    print("Running prediction and evaluation in a single pass...")
    test_predictions = eval_trainer.predict(data["test_split"])
    predictions = np.argmax(test_predictions.predictions, axis=1)
    true_labels = test_predictions.label_ids
    
    # Calculate evaluation metrics manually
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
            f.write(f"Finetuning method: {method}\n")
            f.write(f"Evaluation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Test metrics:\n")
            for metric_name, value in eval_results.items():
                f.write(f"{metric_name}: {value}\n")
            
            f.write("\nConfusion Matrix:\n")
            f.write(f"Saved to: {cm_path}\n\n")
            f.write("Raw confusion matrix values:\n")
            f.write(str(cm))
    
    print(f"Evaluation results for {model_name} with method {method}: {eval_results}")
    print(f"Confusion matrix saved to: {cm_path}")
    
    return eval_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run finetuning on GPT2 models with various methods')
    parser.add_argument('--model', type=str, default="gpt2",
                        help='Model name or path from Hugging Face (default: gpt2)')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS,
                        help=f'Finetuning methods to run (default: all {METHODS})')
    args = parser.parse_args()
    
    model_name = args.model
    methods_to_run = args.methods
    
    print(f"Starting finetuning for model: {model_name}")
    print(f"Methods to run: {methods_to_run}")
    
    # Create base directory for model comparisons
    compare_dir = f"./logs/{model_name}/comparison"
    os.makedirs(compare_dir, exist_ok=True)
    
    # Run finetuning for each method
    results = {}
    for method in methods_to_run:
        results[method] = run_finetuning(model_name, method)
    
    # Create a comparison of results
    comparison_file = f"{compare_dir}/comparison_results.txt"
    with open(comparison_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Comparison completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"{'Method':<10} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}\n")
        f.write('-' * 60 + '\n')
        
        for method, result in results.items():
            f.write(f"{method:<10} {result['accuracy']:<10.4f} {result['f1']:<10.4f} {result['precision']:<10.4f} {result['recall']:<10.4f}\n")
    
    print(f"\nAll finetuning methods completed for model {model_name}.")
    print(f"Results comparison saved to {comparison_file}")

if __name__ == "__main__":
    main()