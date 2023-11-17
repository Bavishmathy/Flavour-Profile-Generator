#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer

# Initialize the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_flavor_profile(molecule_a, molecule_b):
    # Combine the two molecules and create a prompt
    prompt = f"Molecule A is related to {molecule_a} and Molecule B is related to {molecule_b}. Taste Description is:"

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text using the GPT-2 model with attention_mask and pad_token_id set
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1, 
                                pad_token_id=tokenizer.eos_token_id,
                                no_repeat_ngram_size=2,
                                top_k=40,
                                top_p=0.90,
                                temperature=0.8,
                                do_sample=True)  
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract just the taste description part
    taste_description = generated_text[len(prompt):].strip()

    return taste_description

# Example usage
molecule_a = "Vanillic Acid"
molecule_b = "Vanillin"
generated_taste = generate_flavor_profile(molecule_a, molecule_b)
print(f"Taste Description for {molecule_a} and {molecule_b}: {generated_taste}")

