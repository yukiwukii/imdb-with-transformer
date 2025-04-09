import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    GPT2Tokenizer,
    GPT2ForSequenceClassification
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import optuna
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from dotenv import load_dotenv
from huggingface_hub import login

# Load your HF token from .env file
load_dotenv()
token = os.getenv("HF_TOKEN")
login(token)

# Model to finetune
model_name = "gpt2"

# Create directories
os.makedirs(f"./hyperparameter_tuning/{model_name}", exist_ok=True)
os.makedirs(f"./hyperparameter_plots/{model_name}", exist_ok=True)

imdb = load_dataset("imdb")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
tokenized_imdb = imdb.map(lambda e: tokenizer(e["text"], truncation=True, padding=True), batched=True)

# Prepare data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Split train/val/test set
train_val_split = tokenized_imdb["train"].train_test_split(test_size = 0.2)
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
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
    return model

# Define objective function for Optuna
def objective(trial):
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    patience = trial.suggest_float("patience", 1, 3)
    num_epochs = 5

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=patience)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./hyperparameter_tuning/{model_name}/trial_{trial.number}",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=warmup_ratio,
        logging_dir=f"./{model_name}_log",
        logging_steps=500,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    
    # Train and evaluate
    try:
        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Trial {trial.number} - Results: {eval_results}")
        return eval_results["eval_accuracy"]
    except Exception as e:
        print(f"Trial {trial.number} - Failed with error: {e}")
        return 0.0  # Return low score for failed trials

# Visualization functions
def visualize_study_with_matplotlib(study):
    """Create matplotlib visualizations for hyperparameter optimization results"""
    trials_df = study.trials_dataframe()
    
    # Plot optimization history
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trials_df["number"], trials_df["value"], "o-")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Accuracy")
    ax.set_title("Hyperparameter Optimization History")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"./hyperparameter_plots/{model_name}/optimization_history.png")
    
    # Plot parameter importances and get top params for pairwise plotting
    top_params = []
    try:
        param_importances = optuna.importance.get_param_importances(study)
        params = list(param_importances.keys())
        importance_values = list(param_importances.values())
        
        # Store top parameters if available
        if len(params) >= 2:
            top_params = params[:2]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(params, importance_values)
        ax.set_xlabel("Importance")
        ax.set_title("Hyperparameter Importance")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(f"./hyperparameter_plots/{model_name}/parameter_importance.png")
    except Exception as e:
        print(f"Could not compute parameter importances: {e}")
    
    # Plot each parameter individually
    param_names = ["learning_rate", "batch_size", "weight_decay", "patience", "warmup_ratio"]
    for param in param_names:
        param_col = f"params_{param}"
        if param_col in trials_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(trials_df[param_col], trials_df["value"])
            ax.set_xlabel(param)
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Impact of {param} on Accuracy")
            ax.grid(True)
            if param == "learning_rate":
                ax.set_xscale("log")
            fig.tight_layout()
            fig.savefig(f"./hyperparameter_plots/{model_name}/param_{param}.png")
    
    # Plot pairwise relationship for most important parameters if available
    if len(top_params) >= 2:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                trials_df[f"params_{top_params[0]}"], 
                trials_df[f"params_{top_params[1]}"],
                c=trials_df["value"],
                cmap="viridis",
                s=100
            )
            ax.set_xlabel(top_params[0])
            ax.set_ylabel(top_params[1])
            ax.set_title(f"Relationship between {top_params[0]} and {top_params[1]}")
            if top_params[0] == "learning_rate":
                ax.set_xscale("log")
            if top_params[1] == "learning_rate":
                ax.set_yscale("log")
            fig.colorbar(scatter, label="Accuracy")
            fig.tight_layout()
            fig.savefig(f"./hyperparameter_plots/{model_name}/top_params_relationship.png")
        except Exception as e:
            print(f"Error creating pairwise plot: {e}")

# Main function
def main(n_trials=10):
    print(f"Starting hyperparameter tuning with {n_trials} trials...")
    
    # Create and run the study
    study = optuna.create_study(direction="maximize", 
                              study_name="bert_finetune_hyperparameter_tuning")
    study.optimize(objective, n_trials=n_trials)
    
    # Get best trial info
    best_trial = study.best_trial
    print("\nBest trial:")
    print(f"  Value (Accuracy): {best_trial.value}")
    print("  Parameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_study_with_matplotlib(study)
    
    # Find the best checkpoint path
    base_path = f"./hyperparameter_tuning/{model_name}/trial_{best_trial.number}"
    best_checkpoint_path = None
    
    if os.path.exists(base_path):
        checkpoints = [d for d in os.listdir(base_path) if d.startswith('checkpoint-')]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
            best_checkpoint_path = os.path.join(base_path, latest_checkpoint)
            print(f"\nBest model checkpoint found at: {best_checkpoint_path}")
        else:
            print(f"\nNo checkpoint found in {base_path}.")
            return None, best_trial.params, None
    else:
        print(f"\nTrial directory {base_path} does not exist.")
        return None, best_trial.params, None
    
    # Evaluate the best model on the full test set
    if best_checkpoint_path:
        print("\nEvaluating the best model on the full test set...")
        try:
            best_model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_path)
            
            eval_args = TrainingArguments(
                output_dir="./eval_results",
                per_device_eval_batch_size=16,
                report_to="none"
            )
            
            print(f"Full test set label distribution: {Counter(tokenized_imdb['test']['label'])}")
            
            eval_trainer = Trainer(
                model=best_model,
                args=eval_args,
                tokenizer=tokenizer,
                eval_dataset=test_split,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )
            
            final_results = eval_trainer.evaluate()
            print(f"Final evaluation results on full test set: {final_results}")
            
            return best_checkpoint_path, best_trial.params, final_results
        except Exception as e:
            print(f"Error evaluating best model: {e}")
            return best_checkpoint_path, best_trial.params, None
    
    return None, best_trial.params, None

if __name__ == "__main__":
    # Set the number of trials - increase for better results
    n_trials = 10  # Adjust based on available computation time
    
    # Run the hyperparameter tuning
    best_model_path, best_params, final_results = main(n_trials=n_trials)
    
    # Print summary
    print("\n" + "="*50)
    print("Hyperparameter Tuning Summary")
    print("="*50)
    
    if best_model_path:
        print(f"Best model saved at: {best_model_path}")
    else:
        print("No best model path was found.")
    
    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    if final_results:
        print("\nFinal evaluation metrics:")
        for metric, value in final_results.items():
            print(f"  {metric}: {value}")
    
    print("\nVisualization files are available in:")
    print("  ./hyperparameter_plots/ (static images)")
