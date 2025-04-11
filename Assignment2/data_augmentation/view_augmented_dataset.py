from datasets import load_dataset

# Load the IMDB dataset
dataset = load_dataset("HamsterShiu/IMDB_Combined")

offset = 1000
print(dataset["train"][0 + offset])
print(dataset["train"][25000 + offset])