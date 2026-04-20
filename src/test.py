from datasets import load_dataset

dataset = load_dataset("theatticusproject/cuad-qa")

print(dataset)
print(dataset["train"][0])
print(dataset["train"].column_names)