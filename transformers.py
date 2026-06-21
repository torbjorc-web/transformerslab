# Imports and Instantiation
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Instantiate DistilGPT-2's tokenizer and model using .from_pretrained
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

# Tokenization and Generation

# 2. Assign pt_tensors the input text's tokens in PyTorch tensor form
def encode_text_as_pt_tensor(text):
    pt_tensors = tokenizer.encode(text, return_tensors="pt")
    return pt_tensors

print(encode_text_as_pt_tensor("hello, world!"))

from transformers import set_seed

# 3. Use set_seed to make output deterministic (pass 42)
set_seed(42)

prompt = "Your prompt here!"
tokens = encode_text_as_pt_tensor(prompt)

# 4. Generate completion using greedy search method
output_tokens = model.generate(tokens, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# 5. Decode the resulting tokens
def decode_tokens(tokens):
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return text

decode_tokens(output_tokens[0])

# Experimenting with Generation Strategies

# 6. Beam search function - call with 2, 6, and 14 beams
def generate_with_beam_search(prompt, num_beams):
    tokens = encode_text_as_pt_tensor(prompt)
    output = model.generate(tokens, max_length=50, num_return_sequences=1, num_beams=num_beams, pad_token_id=tokenizer.eos_token_id)
    completion = decode_tokens(output[0])
    print(completion)
    return completion

prompt = "The future of artificial intelligence is"

# Call three times with different beam counts
generate_with_beam_search(prompt, 2)
generate_with_beam_search(prompt, 6)
generate_with_beam_search(prompt, 14)

# 7. N-gram penalty function - call with 2, 3, and 4
def generate_with_ngram_penalty(prompt, n_gram_penalty, num_beams=6):
    tokens = encode_text_as_pt_tensor(prompt)
    output = model.generate(tokens, max_length=50, num_return_sequences=1, num_beams=num_beams, no_repeat_ngram_size=n_gram_penalty, pad_token_id=tokenizer.eos_token_id)
    completion = decode_tokens(output[0])
    print(completion)
    return completion

# Call three times with different n-gram values
generate_with_ngram_penalty(prompt, 2)
generate_with_ngram_penalty(prompt, 3)
generate_with_ngram_penalty(prompt, 4)

# 8. Sampling with temperature and top_k - experiment with different values
def generate_with_sampling(prompt, temperature, top_k, n_gram_penalty=2):
    tokens = encode_text_as_pt_tensor(prompt)
    output = model.generate(tokens, max_length=50, num_return_sequences=1, do_sample=True, temperature=temperature, top_k=top_k, no_repeat_ngram_size=n_gram_penalty, pad_token_id=tokenizer.eos_token_id)
    completion = decode_tokens(output[0])
    print(f"Temperature: {temperature}\nTop K: {top_k}\n {completion}")
    return completion

# Make multiple calls with different settings
generate_with_sampling(prompt, 0.6, 50)
generate_with_sampling(prompt, 1.0, 100)
generate_with_sampling(prompt, 1.5, 200)

# Using CodeCarbon

from codecarbon import track_emissions

# 9. Add CodeCarbon decorator above the function
@track_emissions
def generate_with_sampling(prompt, temperature, top_k, n_gram_penalty=2):
    tokens = encode_text_as_pt_tensor(prompt)
    output = model.generate(tokens, max_length=50, num_return_sequences=1, do_sample=True, temperature=temperature, top_k=top_k, no_repeat_ngram_size=n_gram_penalty, pad_token_id=tokenizer.eos_token_id)
    completion = decode_tokens(output[0])
    print(f"Temperature: {temperature}\nTop K: {top_k}\n {completion}")
    return completion

generate_with_sampling("Carbon dioxide is a", 0.6, 50)

import pandas as pd

# 10. Load emissions.csv with pandas and display with head()
emissions = pd.read_csv("emissions.csv")
print(emissions.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# 11. Plot emissions by duration using scatter plot
plt.scatter(emissions['duration'], emissions['emissions'], color='blue', alpha=0.6)
plt.title('Emissions by Duration')
plt.xlabel('Duration (seconds)')
plt.ylabel('Emissions (kg CO2)')
plt.grid(True)
plt.show()
