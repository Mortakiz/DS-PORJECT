# train_cpt.py
import os, math, torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['model_name']
train_file = config['cpt_train_file']
val_file = config['cpt_val_file']
block_size = config['cpt_block_size']
epochs = config['cpt_epochs']
lr = float(config['cpt_learning_rate'])
batch_size = config['cpt_batch_size']
grad_accum = config['cpt_gradient_accumulation_steps']

print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

print("Loading datasets...")
raw_datasets = load_dataset("text", data_files={"train": train_file, "validation": val_file})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=block_size, padding="max_length")

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def calculate_perplexity(model, dataloader):
    model.eval()
    total_loss = 0
    total_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_steps += 1
    return math.exp(total_loss / max(total_steps, 1))

val_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator)

print("Calculating initial perplexity...")
initial_perplexity = calculate_perplexity(model, val_dataloader)
print(f"Initial Perplexity: {initial_perplexity:.2f}")

training_args = TrainingArguments(
    output_dir="./checkpoints/cpt",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=lr,
    save_strategy="epoch",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

print("Starting CPT training...")
trainer.train()
trainer.save_model("./checkpoints/cpt_final")
tokenizer.save_pretrained("./checkpoints/cpt_final")

print("Calculating final perplexity...")
final_perplexity = calculate_perplexity(model, val_dataloader)
print(f"Final Perplexity: {final_perplexity:.2f}")
print(f"\nCPT Complete! Perplexity went from {initial_perplexity:.2f} -> {final_perplexity:.2f}")