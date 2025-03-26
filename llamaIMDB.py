import ollama
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def analyze_sentiment(model_name, text):
    """
    Analyze the sentiment of a text using an Ollama model.
    
    Args:
        model_name (str): Name of the Ollama model to use
        text (str): Text to analyze
            
    Returns:
        tuple: (sentiment_label, justification)
            - sentiment_label (int or None): 1 for positive sentiment, 0 for negative sentiment, None if failed
            - justification (str): The model's full response text
    """
    prompt = (
        "Analyze the sentiment of the following movie review. "
        "First explain your reasoning in 1-2 sentences, then on a new line provide your final "
        "classification as either 'POSITIVE' or 'NEGATIVE'.\n\n"
        f"Review: {text}"
    )
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert in text sentiment analysis. Provide reasoning followed by classification."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract full response
        justification = response['message']['content'].strip()
        response_text = justification.upper()
        
        # Determine sentiment based on presence of POSITIVE/NEGATIVE
        if "POSITIVE" in response_text:
            return 1, justification
        elif "NEGATIVE" in response_text:
            return 0, justification
        else:
            print(f"No clear sentiment found in: {response_text[:50]}...")
            return None, justification
            
    except Exception as e:
        error_msg = f"Error analyzing sentiment: {str(e)}"
        print(error_msg)
        return None, error_msg


def evaluate_model(model_name, dataset_name="imdb", num_samples=100):
    """
    Evaluate a model's performance on sentiment analysis.
    
    Args:
        model_name (str): Name of the Ollama model to use
        dataset_name (str): Name of the HuggingFace dataset
        num_samples (int): Number of samples to test
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(dataset_name, split='test')
    
    # Select samples
    test_samples = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"Evaluating {model_name} on {len(test_samples)} samples from {dataset_name}...")
    
    # Run predictions
    true_labels = []
    predicted_labels = []
    results = []
    failed_count = 0
    
    for i, sample in enumerate(tqdm(test_samples)):
        text = sample['text']
        true_label = sample['label']
        
        # Get prediction
        predicted_label, justification = analyze_sentiment(model_name, text)
        
        if predicted_label is not None:
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            
            # Store result
            results.append({
                'id': i,
                'full_text': text,  # Store the complete text
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'justification': justification  # Store the model's justification
            })
        else:
            failed_count += 1
            results.append({
                'id': i,
                'full_text': text,  # Store the complete text
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
                'true_label': true_label,
                'predicted_label': 'FAILED',
                'justification': justification  # Store the error message or response
            })
    
    # Calculate metrics if we have predictions
    if true_labels and predicted_labels:
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
        
        # Calculate other metrics
        metrics = {
            'model': model_name,
            'accuracy': float(accuracy_score(true_labels, predicted_labels)),
            'precision': float(precision_score(true_labels, predicted_labels, zero_division=0)),
            'recall': float(recall_score(true_labels, predicted_labels, zero_division=0)),
            'f1_score': float(f1_score(true_labels, predicted_labels, zero_division=0)),
            'true_positive': int(tp),
            'false_positive': int(fp),
            'true_negative': int(tn),
            'false_negative': int(fn),
            'failed_count': failed_count,
            'total_processed': len(test_samples),
            'failure_rate': failed_count / len(test_samples)
        }
    else:
        # Handle the case where all predictions failed
        metrics = {
            'model': model_name,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'true_positive': 0,
            'false_positive': 0,
            'true_negative': 0,
            'false_negative': 0,
            'failed_count': failed_count,
            'total_processed': len(test_samples),
            'failure_rate': 1.0
        }
    
    # Save results
    results_df = pd.DataFrame(results)
    filename = f"results/{model_name.replace(':', '_')}_{num_samples}_samples.csv"
    results_df.to_csv(filename, index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_filename = f"results/{model_name.replace(':', '_')}_{num_samples}_metrics.csv"
    metrics_df.to_csv(metrics_filename, index=False)
    
    print(f"Results saved to {filename}")
    print(f"Metrics saved to {metrics_filename}")
    
    return metrics


def evaluate_models(models, dataset_name="imdb", num_samples=100):
    """
    Evaluate multiple models on sentiment analysis.
    
    Args:
        models (list): List of model names to evaluate
        dataset_name (str): Name of the HuggingFace dataset
        num_samples (int): Number of samples to test
        
    Returns:
        DataFrame: Combined metrics for all models
    """
    all_metrics = []
    
    for model in models:
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model}")
        print(f"{'='*50}")
        
        try:
            metrics = evaluate_model(model, dataset_name, num_samples)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Failed to evaluate {model}: {str(e)}")
            all_metrics.append({
                'model': model,
                'error': str(e),
                'accuracy': 0.0,
                'f1_score': 0.0
            })
    
    # Save combined metrics
    combined_df = pd.DataFrame(all_metrics)
    combined_filename = f"results/combined_metrics_{num_samples}_samples.csv"
    combined_df.to_csv(combined_filename, index=False)
    
    print(f"\nCombined metrics saved to {combined_filename}")
    
    return combined_df


def main():
    # List of models to evaluate
    models = ['llama3.2:1b', 'llama3.2:3b', 'llama3.1:7b']
    
    # Number of samples to test
    num_samples = 1000
    
    print(f"Starting sentiment analysis on {num_samples} samples")
    
    # Evaluate models
    results = evaluate_models(models, "imdb", num_samples)
    
    # Print summary
    print("\nSentiment Analysis Summary:")
    for _, row in results.iterrows():
        print(f"\nModel: {row['model']}")
        
        if 'error' in row and not pd.isna(row['error']):
            print(f"ERROR: {row['error']}")
            continue
            
        print(f"Accuracy: {row['accuracy']:.4f}")
        print(f"F1 Score: {row['f1_score']:.4f}")
        print(f"Failed Responses: {row['failed_count']}/{row['total_processed']} "
              f"({row['failure_rate']*100:.2f}%)")
    
    print("\nSentiment analysis complete. Check 'results' directory for detailed outputs.")


if __name__ == "__main__":
    main()