from datasets import load_dataset, concatenate_datasets
from transformers import pipeline, MarianTokenizer
from tqdm import tqdm
import numpy as np

imdb = load_dataset("imdb", split="train")

second_lang = "de"

first_tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{second_lang}")
second_tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{second_lang}-en")

first_translation = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{second_lang}")
second_translation = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{second_lang}-en")

def back_translate(text):
    first_tokens = first_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    first_text = first_tokenizer.decode(first_tokens['input_ids'][0], skip_special_tokens=True)
    translated_tokens = first_translation(first_text, max_length=512)
    
    second_tokens = second_tokenizer(translated_tokens[0]['translation_text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
    second_text = second_tokenizer.decode(second_tokens['input_ids'][0], skip_special_tokens=True)
    back_translated = second_translation(second_text, max_length=512)[0]['translation_text']
    
    return back_translated

def augment_batch(batch):
    augmented_texts = []
    for text in tqdm(batch["text"]):
        augmented_texts.append(back_translate(text))
    return {"text": augmented_texts, "label": batch["label"]}

augmented = imdb.map(
    augment_batch,
    batched=True,
    batch_size=32,
    remove_columns=imdb.column_names
)

final_dataset = concatenate_datasets([imdb, augmented])
print(f"Original: {len(imdb)}, Augmented: {len(augmented)}, Total: {len(final_dataset)}")

final_dataset.save_to_disk("./imdb_combined")
augmented.save_to_disk("./imdb_augmented")