import torch.nn as nn
import torch.nn.functional as F

text = open("shakespeare.txt").read()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}  # "string to int"
itos = {i: c for i, c in enumerate(chars)}  # "int to string"

def encode(text):
    return [stoi[c] for c in text]

def decode(indices):
    return ''.join([itos[i] for i in indices])

import torch
data = torch.tensor(encode(text), dtype=torch.long)
# should be ~1M tokens for tiny shakespeare
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")

block_size = 64   # how many characters per training chunk
batch_size = 32   # how many chunks per batch

# first split your data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

x, y = get_batch('train')

n_embd = 64      # embedding dimension
n_heads = 4
head_size = n_embd // n_heads  # = 16

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('causal_mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x shape: (batch, time, channels)
        # return shape: (batch, time, head_size)
        B, T, C = x.shape
        K = self.key(x)
        Q = self.query(x)
        values = self.value(x)
        attention = Q @ K.transpose(-2, -1) * K.shape[-1] ** -0.5
        attention = attention.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        out = attention @ values
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        # create a list of n_heads Head objects — look up nn.ModuleList
        # add a self.proj linear layer: (n_embd -> n_embd) for the output projection
        self.attention_heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # run each head on x, concatenate outputs along the last dim
        # then apply self.proj
        out = torch.cat([h(x) for h in self.attention_heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # linear: n_embd -> 4 * n_embd
        # ReLU
        # linear: 4 * n_embd -> n_embd
        self.net = nn.Sequential(
           nn.Linear(n_embd,4 * n_embd),
           nn.ReLU(),
           nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.multihead = MultiHeadAttention(n_heads, n_embd//n_heads)
        self.feedforward = FeedForward(n_embd)
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)
        # FeedForward
        # two nn.LayerNorm layers (one for each sub-block)

    def forward(self, x):
        # apply attention with residual connection
        x = x+self.multihead(self.layernorm1(x))
        # apply feedforward with residual connection
        x = x+self.feedforward(self.layernorm2(x))
        return x

class CharTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_heads) for _ in range(4)])
        self.layernorm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        # nn.Embedding for token embeddings (vocab_size, n_embd)
        # nn.Embedding for positional embeddings (block_size, n_embd)
        # nn.Sequential of N TransformerBlocks (use n_layers = 4)
        # final nn.LayerNorm
        # final nn.Linear head (n_embd -> vocab_size)

    def forward(self, x, targets=None):
        # x shape: (batch, time)
        # 1. token embeddings + positional embeddings
        # 2. pass through transformer blocks
        # 3. apply final layernorm and linear head → logits
        # 4. if targets provided, compute cross entropy loss and return (logits, loss)
        # otherwise just return (logits, None)
        B, T = x.shape
        tok_emb = self.token_emb(x)                        # (B, T, n_embd)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))            # (T, n_embd)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.layernorm(x)
        logits = self.head(x)
        if targets is not None:
            # cross_entropy expects (batch, vocab_size, time) or flattened
            loss = F.cross_entropy(logits.view(B*T, vocab_size), targets.view(B*T))
            return logits, loss
        return logits, None

# training loop

model = CharTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
losses = []

for step in range(10000):
    x, y = get_batch('train')
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f'step {step} | loss {loss.item()}')
        losses.append(loss.item())
    # 1. forward pass — get logits and loss
    # 2. zero gradients
    # 3. backward pass
    # 4. optimizer step
    # 5. print loss every 500 steps

@torch.no_grad()
def generate(model, start_text, max_new_tokens=200):
    encoded = encode(start_text)
    x = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        x_cropped = x[:, -block_size:]          # crop to block_size
        logits, _ = model(x_cropped)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
    return decode(x[0].tolist())
print(generate(model, start_text="Good morning"))