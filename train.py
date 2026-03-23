import os
import time
import math
import pickle
import numpy as np
import torch
from model import CharGPT, GPTConfig

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 20000          # 20k-50k suggested
eval_interval = 500        # Evaluate loss every 500 steps
generate_interval = 1000   # Generate poetry samples every 1000 steps
learning_rate = 3e-4
warmup_iters = 1000        # Learning rate warmup
eval_iters = 200           # Number of batches to estimate loss
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load metadata for vocabulary size and decoding
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
itos = meta['itos']
decode = lambda l: ''.join([itos[i] for i in l])

# Load data using memory mapping for efficiency
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')

def get_batch(split):
    """Fetch a randomized batch of inputs (x) and targets (y)."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    """Evaluate mean loss and BPC (Bits Per Character) over train and val splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        mean_loss = losses.mean().item()
        # BPC = CrossEntropy (base e) / ln(2)
        bpc = mean_loss / math.log(2)
        out[split] = {'loss': mean_loss, 'bpc': bpc}
    model.train()
    return out

@torch.no_grad()
def generate_sample(model, max_new_tokens=200):
    """Generate a sample text to observe the emergence of poetic structures."""
    model.eval()
    # Start with a single newline character as the prompt
    context = torch.tensor([[meta['stoi']['\n']]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        # Crop context to block_size to prevent crashing
        idx_cond = context[:, -block_size:]
        logits, _ = model(idx_cond)
        # Focus on the last time step
        logits = logits[:, -1, :]
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # Append to the sequence
        context = torch.cat((context, idx_next), dim=1)
    
    print("\n" + "="*50)
    print(f"POETRY SAMPLE (Iter {iter_num}):")
    print(decode(context[0].tolist()))
    print("="*50 + "\n")
    model.train()

def get_lr(it):
    """Linear warmup followed by constant learning rate."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    return learning_rate

# Initialize Model and Optimizer
config = GPTConfig(vocab_size=vocab_size, block_size=block_size)
model = CharGPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Initializing training on {device}...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

# Training Loop
best_val_loss = float('inf')
t0 = time.time()

for iter_num in range(max_iters):
    # Update learning rate
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate and print logs
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Step {iter_num:4d} | Train Loss: {losses['train']['loss']:.4f} (BPC: {losses['train']['bpc']:.4f}) | "
              f"Val Loss: {losses['val']['loss']:.4f} (BPC: {losses['val']['bpc']:.4f}) | LR: {lr:.2e}")
        
        # Save checkpoint if validation loss improves
        if losses['val']['loss'] < best_val_loss:
            best_val_loss = losses['val']['loss']
            torch.save(model.state_dict(), 'ckpt.pt')

    # Generate sample
    if iter_num > 0 and iter_num % generate_interval == 0:
        generate_sample(model)

    # Forward, Backward, Update
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

t1 = time.time()
print(f"Training completed in {(t1-t0)/60:.2f} minutes.")