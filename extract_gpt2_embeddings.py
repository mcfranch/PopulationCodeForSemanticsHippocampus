import os
import gc
import random

import numpy as np
import pandas as pd
import torch
from torch.nn import DataParallel
from transformers import GPT2Tokenizer, GPT2Model


# Config
NUM_LAYERS = 37          # 36 transformer layers + 1 embedding layer
MAX_WINDOW_SIZE = 128
SENTENCE_PUNCT = {',', '.', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}'}

INPUT_FILE = "podcastWordsWithAll.xlsx"
OUTPUT_DIR = "GPT-2 embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Fix seeds for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Load model and tokenizer
print("Loading GPT-2 Large model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2Model.from_pretrained('gpt2-large')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use all available GPUs, run in eval mode (no dropout)
model = DataParallel(model).to(device)
tokenizer.pad_token = tokenizer.eos_token
model.eval()


# Read input words
print(f"Reading input from: {INPUT_FILE}")
df = pd.read_excel(INPUT_FILE, usecols=['Text'])
all_words = df['Text'].tolist()
total_words = len(all_words)


def find_word_tokens(all_tokens, target_word, start_idx):
    """Walk the token stream from start_idx and return the token indices that
    together reconstruct target_word (skipping standalone punctuation)."""
    current_pos = start_idx
    found_tokens = []
    found_indices = []
    clean_target = target_word.strip().strip(',.!?;:()[]{}')

    while current_pos < len(all_tokens):
        token = all_tokens[current_pos]
        clean_token = token.replace('Ġ', '')
        if clean_token in SENTENCE_PUNCT:
            current_pos += 1
            continue

        found_tokens.append(token)
        found_indices.append(current_pos)
        reconstructed = ''.join(t.replace('Ġ', '') for t in found_tokens)
        if reconstructed.lower() == clean_target.lower():
            return found_indices, current_pos + 1
        current_pos += 1

    return [], current_pos


def get_hidden_states_for_window(window_words, target_layer):
    """Slide through window_words one word at a time, keeping the context under
    MAX_WINDOW_SIZE tokens, and collect the mean hidden state for each word."""
    embeddings = []
    current_context = []
    window_start_idx = 0

    print("\n" + "=" * 50)
    print(f"Processing layer {target_layer}")
    print("=" * 50)

    for i, target_word in enumerate(window_words):
        current_context.append(target_word)

        # Trim the front of the context until it fits the window
        test_tokens = tokenizer.tokenize(' '.join(current_context))
        while len(test_tokens) > MAX_WINDOW_SIZE and window_start_idx < len(current_context) - 1:
            window_start_idx += 1
            test_tokens = tokenizer.tokenize(' '.join(current_context[window_start_idx:]))

        # Tokenize the current context window
        context_text = ' '.join(current_context[window_start_idx:])
        encoding = tokenizer(context_text, return_tensors='pt', add_special_tokens=True).to(device)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        if i == window_start_idx:
            start_idx = 0
        else:
            prev_context = ' '.join(current_context[window_start_idx:-1])
            start_idx = len(tokenizer.tokenize(prev_context))

        found_indices, _ = find_word_tokens(all_tokens, target_word, start_idx)

        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            output_hidden_states=True)
        hidden_states = outputs.hidden_states[target_layer][0]

        if found_indices:
            emb = torch.mean(hidden_states[found_indices], dim=0).cpu().numpy()
        else:
            emb = np.zeros(model.module.config.hidden_size)

        emb = np.array(emb, ndmin=1)
        embeddings.append(emb)

        del outputs, hidden_states
        torch.cuda.empty_cache()

    return embeddings, len(current_context)


def process_layer(layer_idx):
    """Run every word through the model for a single layer and save the result."""
    print(f"\nProcessing layer {layer_idx}...")
    layer_data = []
    window_start = 0

    while window_start < total_words:
        window_words = all_words[window_start:]
        embeddings, n = get_hidden_states_for_window(window_words, layer_idx)
        for w, e in zip(window_words[:n], embeddings):
            layer_data.append([w] + e.tolist())
        window_start += n

    cols = ['Word'] + [f'dim_{i}' for i in range(model.module.config.hidden_size)]
    df_layer = pd.DataFrame(layer_data, columns=cols)
    out_path = os.path.join(OUTPUT_DIR, f'layer_{layer_idx} 128.xlsx')
    df_layer.to_excel(out_path, index=False)
    print(f"Saved {out_path}")
    gc.collect()


if __name__ == "__main__":
    for layer_idx in range(NUM_LAYERS):
        process_layer(layer_idx)
    print(f"\nAll {NUM_LAYERS} layers processed and saved.")
