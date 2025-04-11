from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

model_name = "FacebookAI/roberta-base" #SET MODEL NAME HERE
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)
model.eval()

dataset = load_dataset("imdb", split="test")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

dataloader = DataLoader(tokenized_dataset, batch_size=8, collate_fn=data_collator)

total_loss = 0
total_correct = 0
total_masked = 0

with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        mask = labels != -100
        preds = logits.argmax(dim=-1)
        correct = (preds[mask] == labels[mask]).sum().item()
        total_correct += correct
        total_masked += mask.sum().item()
        total_loss += loss.item()

avg_loss = total_loss / len(dataloader)
accuracy = total_correct / total_masked

print(f"\nMLM Evaluation on IMDB Test Set:")
print(f"Avg Loss: {avg_loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")