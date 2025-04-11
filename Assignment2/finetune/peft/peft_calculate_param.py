import os
import argparse
import torch
from transformers import AutoModelForSequenceClassification
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    IA3Config,
    TaskType
)

def count_parameters(model):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total_params,
        "trainable": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0
    }

def get_model_and_hidden_size(model_name):
    """Get the model and determine its hidden size"""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    
    # Get the hidden size from the model config
    config = model.config
    hidden_size = getattr(config, "hidden_size", 768)  # Default to 768 if not found
    
    return model, hidden_size

def get_model_for_method(model_name, method):
    """Get the appropriate model based on the model name and finetuning method"""
    # Get base model and hidden size
    base_model, hidden_size = get_model_and_hidden_size(model_name)
    
    if method == "full":
        # Full finetuning - return the model as is
        return base_model
    
    elif method == "lora":
        # LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],
            bias="none",
            inference_mode=False
        )
        model = get_peft_model(base_model, peft_config)
        return model
    
    elif method == "prefix":
        # Prefix Tuning configuration
        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=20,
            prefix_projection=True,
            encoder_hidden_size=hidden_size
        )
        peft_config.modules_to_save = ["classifier"]
        model = get_peft_model(base_model, peft_config)
        return model
    
    elif method == "ia3":
        # IA3 configuration
        peft_config = IA3Config(
            task_type=TaskType.SEQ_CLS,
            target_modules=["query", "key", "value", "output.dense"],
            feedforward_modules=["output.dense"],
            inference_mode=False
        )
        model = get_peft_model(base_model, peft_config)
        return model
    
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate parameter efficiency for different PEFT methods')
    parser.add_argument('--model', type=str, default="bert-base-cased",
                        help='Model name or path from Hugging Face (default: bert-base-cased)')
    args = parser.parse_args()
    
    model_name = args.model
    methods = ["full", "lora", "prefix", "ia3"]
    
    # Create output directory
    output_dir = f"./logs/{model_name}/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # File to save parameter information
    param_file = f"{output_dir}/parameter_comparison.txt"
    
    # Get parameter information for each method
    results = {}
    
    with open(param_file, 'w') as f:
        f.write(f"Parameter Efficiency Analysis for {model_name}\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"{'Method':<10} {'Total Params':<15} {'Trainable Params':<20} {'Trainable %':<15}\n")
        f.write("-" * 80 + "\n")
        
        for method in methods:
            print(f"Analyzing method: {method}")
            
            # Get model with the method
            model = get_model_for_method(model_name, method)
            
            # Count parameters
            params = count_parameters(model)
            results[method] = params
            
            # Write to file
            f.write(f"{method:<10} {params['total']:<15,d} {params['trainable']:<20,d} {params['trainable_percent']:.4f}%\n")
        
        # Add summary section
        f.write("\n\nSummary (Parameter Efficiency)\n")
        f.write("-" * 40 + "\n")
        baseline = results["full"]["trainable"]
        
        for method in methods:
            if method == "full":
                continue
            
            params = results[method]
            efficiency = 100 - (params["trainable"] / baseline * 100)
            
            f.write(f"{method:<10}: {efficiency:.2f}% reduction in trainable parameters compared to full fine-tuning\n")
    
    print(f"Parameter analysis completed. Results saved to {param_file}")

if __name__ == "__main__":
    main()