from datasets import load_dataset
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, EarlyStoppingCallback

dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = dataset["unsupervised"].map(lambda e: tokenizer(e["text"], truncation=True, padding=True), batched=True)


train_test_split = dataset.train_test_split(test_size=0.2)

train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
print('No of parameters: ', model.num_parameters())


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

training_args = TrainingArguments(
    output_dir='output/',
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[early_stopping_callback],
)

trainer.train()
trainer.save_model('output/')