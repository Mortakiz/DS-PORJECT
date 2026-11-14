import pandas as pd
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

df = pd.read_csv("SEED-TEST-CSV.csv")
print(f"Loaded {len(df)} test questions.")

models_to_test = {
    "base": "aubmindlab/aragpt2-base",
    "cpt": "./checkpoints/cpt_final",
    "sft": "./checkpoints/sft_final"
}

os.makedirs("./results", exist_ok=True)
special_tokens_dict = {'additional_special_tokens': ['<|user|>', '<|assistant|>', '<|end|>']}

def generate_answer(model, tokenizer, question):
    prompt = f"<|user|> {question} <|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.0,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return full_text.split("<|assistant|>")[-1].replace("<|end|>", "").strip()

for model_name, model_path in models_to_test.items():
    print(f"\n--- Scoring: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    if model_name == "sft":
        tokenizer.add_special_tokens(special_tokens_dict)
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if model_name == "sft":
        model.resize_token_embeddings(len(tokenizer))
        
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    results = []
    for i, row in df.iterrows():
        answer = generate_answer(model, tokenizer, str(row['question']))
        results.append({
            "id": row['id'], 
            "question": row['question'],
            "expected_answer": row['expected_answer'],
            "model_answer": answer, 
            "score": "", 
            "note": ""
        })
        print(f"  {i+1}/{len(df)}")

    pd.DataFrame(results).to_csv(f"./results/{model_name}_results.csv", index=False, encoding='utf-8-sig')
    print(f"Saved ./results/{model_name}_results.csv")

print("\nAll results saved.")