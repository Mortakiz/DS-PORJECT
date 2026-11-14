# train_sft.py
import os, json, torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

sft_file = config['sft_data_file']
epochs = config['sft_epochs']
lr = float(config['sft_learning_rate'])
batch_size = config['sft_batch_size']

print("Loading CPT model...")
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/cpt_final")
model = AutoModelForCausalLM.from_pretrained("./checkpoints/cpt_final")

special_tokens_dict = {'additional_special_tokens': ['<|user|>', '<|assistant|>', '<|end|>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
print(f"Added {num_added_toks} special tokens.")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open(sft_file, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

dataset = Dataset.from_dict({"text": raw_data})
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

def mask_labels(example):
    input_ids = example['input_ids']
    labels = list(input_ids)
    assistant_token_id = tokenizer.convert_tokens_to_ids('<|assistant|>')
    try:
        idx = input_ids.index(assistant_token_id)
        for i in range(idx + 1):
            labels[i] = -100
    except ValueError:
        labels = [-100] * len(labels)
    example['labels'] = labels
    return example

tokenized_dataset = tokenized_dataset.map(mask_labels)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

training_args = TrainingArguments(
    output_dir="./checkpoints/sft",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=lr,
    save_strategy="epoch",
    logging_steps=5,
    eval_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
)

print("Starting SFT training...")
trainer.train()
trainer.save_model("./checkpoints/sft_final")
tokenizer.save_pretrained("./checkpoints/sft_final")
print("SFT Complete!")