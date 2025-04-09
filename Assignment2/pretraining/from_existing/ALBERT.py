from datasets import load_dataset, disable_caching
import os
import tokenizers
from transformers import AlbertTokenizer, AlbertForPreTraining, DataCollatorForLanguageModeling, Trainer, TrainingArguments, EarlyStoppingCallback, BertForPreTraining
import random
from torch.utils.data import DataLoader
import torch

random.seed(42)

os.environ["HF_DATASETS_CACHE"] = "/scratch/users/ntu/shiu0005/hf_cache"
os.environ["TMPDIR"] = "/scratch/users/ntu/shiu0005/temp"
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

dataset = load_dataset("imdb")
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

dataset = dataset["unsupervised"].map(lambda e: tokenizer(e["text"], truncation=True, padding=True), batched=True)


train_test_split = dataset.train_test_split(test_size=0.2)

train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")


model = AlbertForPreTraining.from_pretrained("albert-base-v2")
print('No of parameters: ', model.num_parameters())


def create_nsp_examples(examples):
    sentence1 = []
    sentence2 = []
    labels = []
    
    all_sentences = []
    for text in examples["text"]:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        all_sentences.append(sentences)
    
    # Create pairs
    for sentences in all_sentences:
        for i in range(len(sentences)-1):
            # Positive example
            sentence1.append(sentences[i])
            sentence2.append(sentences[i+1])
            labels.append(1)
            
            # Negative example
            other_doc_idx = random.randint(0, len(all_sentences)-1)
            while other_doc_idx == all_sentences.index(sentences) and len(all_sentences) > 1:
                other_doc_idx = random.randint(0, len(all_sentences)-1)
            
            random_sentence = random.choice(all_sentences[other_doc_idx])
            sentence1.append(sentences[i])
            sentence2.append(random_sentence)
            labels.append(0)
    
    return {
        "sentence1": sentence1,
        "sentence2": sentence2,
        "label": labels
    }


nsp_dataset = train_dataset.map(
    create_nsp_examples,
    batched=True,
    remove_columns=["text", "input_ids", "token_type_ids", "attention_mask", "labels"],
    batch_size=32
)

val_dataset = val_dataset.map(
    create_nsp_examples,
    batched=True,
    remove_columns=["text", "input_ids", "token_type_ids", "attention_mask", "labels"],
    batch_size=32
)


class DataCollatorForBertPretraining:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )
    
    def __call__(self, examples):
        # Process for NSP
        batch = {
            "input_ids": [self.tokenizer(
                text=example["sentence1"],
                text_pair=example["sentence2"],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )["input_ids"][0] for example in examples],
            "sentence_order_label": torch.tensor([example["label"] for example in examples])
        }
        
        # Process for MLM
        mlm_batch = self.mlm_collator([{"input_ids": ids} for ids in batch["input_ids"]])
        batch["input_ids"] = mlm_batch["input_ids"]
        batch["labels"] = mlm_batch["labels"]
        
        return batch

collator = DataCollatorForBertPretraining(tokenizer)


early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

class BertPretrainingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forward pass
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
            sentence_order_label=inputs.get("sentence_order_label")
        )
        
        # Extract losses
        loss = outputs.loss
        
        # Log separate losses
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "loss": loss.item()
            })
        
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./albertoutput",
    overwrite_output_dir=True,
    num_train_epochs=100,
    prediction_loss_only=True,
    logging_dir='./albertlogs',
    logging_steps=500,
    report_to="tensorboard",
    remove_unused_columns=False,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = BertPretrainingTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=nsp_dataset,
    eval_dataset=val_dataset,
    callbacks=[early_stopping_callback],
)

trainer.train()