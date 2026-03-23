import os
import pickle
import requests
import numpy as np

# Define corpus URL and local paths (using TinyShakespeare as a reliable poetic dataset)
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_PATH = 'poetry_corpus.txt'

def download_data():
    """Download the dataset if it does not exist locally."""
    if not os.path.exists(DATA_PATH):
        print("Downloading corpus...")
        response = requests.get(DATA_URL)
        with open(DATA_PATH, 'w', encoding='utf-8') as f:
            f.write(response.text)

def main():
    download_data()

    # Read the corpus
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Total characters in corpus: {len(text):,}")

    # Build character-level vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create character-to-integer and integer-to-character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode the entire text into integers
    data = np.array([stoi[c] for c in text], dtype=np.uint16)

    # Split into train (90%) and validation (10%) sets
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Save to binary files for efficient data loading during training
    train_data.tofile('train.bin')
    val_data.tofile('val.bin')

    # Save metadata (mappings) to decode generated text in Stage IV
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi
    }
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
        
    print("Processing complete. Saved train.bin, val.bin, and meta.pkl.")

if __name__ == '__main__':
    main()