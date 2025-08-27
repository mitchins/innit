import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

PAD = 256

def to_bytes(text, max_len=2048):
    """Convert text to byte tensor with padding"""
    b = (text or "").encode("utf-8", "ignore")[:max_len]
    x = torch.full((max_len,), PAD, dtype=torch.long)
    if b:
        x[:len(b)] = torch.tensor(list(b), dtype=torch.long)
    return x

class TinyByteCNN_EN(nn.Module):
    """Tiny byte-level CNN for English detection"""
    
    def __init__(self, emb=64, blocks=4):
        super().__init__()
        self.emb = nn.Embedding(257, emb)
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(emb, emb, 9, padding=4, groups=emb),
                nn.Conv1d(emb, emb, 1),
                nn.GELU(),
                nn.Dropout(0.1)
            ))
            self.norms.append(nn.LayerNorm(emb))
        
        self.fc = nn.Linear(emb * 2, 2)

    def forward(self, x):
        """Forward pass: x shape (B, L)"""
        h = self.emb(x).transpose(1, 2)  # (B, E, L)
        
        for block, ln in zip(self.blocks, self.norms):
            y = block(h)
            h = (h + y).transpose(1, 2)  # (B, L, E)
            h = ln(h).transpose(1, 2)    # (B, E, L)
        
        # Global pooling
        mean = h.mean(2)  # (B, E)
        mmax = h.amax(2)  # (B, E)
        
        return self.fc(torch.cat([mean, mmax], 1))  # (B, 2)

def hf_stream_text(dataset_id, split, text_key="text", lang_key=None, lang_value=None, max_items=50000):
    """Stream text from HuggingFace dataset"""
    try:
        ds = load_dataset(dataset_id, split=split, streaming=True)
        count = 0
        for r in ds:
            if count >= max_items:
                break
            if lang_key and lang_value is not None and r.get(lang_key) != lang_value:
                continue
            t = r.get(text_key) or r.get("content") or r.get("raw") or ""
            if t and len(t) >= 50:
                yield t
                count += 1
    except Exception as e:
        print(f"Error loading {dataset_id}: {e}")

def english_iter(max_items=50000):
    """Generate English text samples"""
    print("Loading English samples...")
    count = 0
    
    # Try language identification dataset first (most reliable)
    try:
        print("  Trying language identification dataset...")
        ds = load_dataset("papluca/language-identification", streaming=True)
        for r in ds["train"]:
            if count >= max_items:
                break
            if r.get("labels") == 0:  # English is label 0
                yield r["text"]
                count += 1
                if count % 5000 == 0:
                    print(f"    Loaded {count} English samples")
    except Exception as e:
        print(f"  Failed to load language identification dataset: {e}")
    
    # Add synthetic English samples for diversity
    english_samples = [
        "The quick brown fox jumps over the lazy dog. This is a test of English language detection.",
        "To be or not to be, that is the question. Shakespeare wrote many famous plays in English.",
        "In the beginning was the Word, and the Word was with God. This is from the Bible.",
        "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty.",
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
        "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse.",
        "Happy families are all alike; every unhappy family is unhappy in its own way.",
        "All children, except one, grow up. They soon know that they will grow up, and the way Wendy knew was this.",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms.",
        "Mr. and Mrs. Dursley of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much.",
    ] * 2000
    
    print(f"  Adding {len(english_samples)} synthetic English samples...")
    for sample in english_samples:
        if count >= max_items:
            break
        yield sample
        count += 1

def nonenglish_iter(max_items=50000):
    """Generate non-English text samples"""
    print("Loading non-English samples...")
    
    # Try to load from language identification dataset
    try:
        ds = load_dataset("papluca/language-identification", streaming=True)
        count = 0
        for r in ds["train"]:
            if count >= max_items:
                break
            if r.get("labels") != 0:  # Non-English
                yield r["text"]
                count += 1
    except Exception as e:
        print(f"Failed to load language identification dataset: {e}")
        
        # Fallback: synthetic non-English samples
        non_english_samples = [
            "Bonjour, comment allez-vous aujourd'hui?",
            "Hola, ¿cómo estás? Me llamo María.",
            "Guten Tag, ich heiße Hans und komme aus Deutschland.",
            "Ciao, come stai? Io sto bene, grazie.",
            "Привет, как дела? Меня зовут Иван.",
            "こんにちは、元気ですか？私の名前は田中です。",
            "안녕하세요, 어떻게 지내세요? 제 이름은 김철수입니다.",
            "你好，你好吗？我的名字是李明。",
        ] * 2000
        
        for sample in non_english_samples:
            yield sample

class BookBin(torch.utils.data.Dataset):
    """Binary dataset for English vs non-English"""
    
    def __init__(self, pos_iter, neg_iter, max_len=2048, short_prob=0.3):
        self.buf = []
        
        print("Loading positive samples...")
        for i, t in enumerate(pos_iter):
            if i >= 25000:  # Limit samples to avoid memory issues
                break
            self.buf.append((t, 1))
            if i % 5000 == 0:
                print(f"  Loaded {i} positive samples")
        
        print("Loading negative samples...")
        for i, t in enumerate(neg_iter):
            if i >= 25000:
                break
            self.buf.append((t, 0))
            if i % 5000 == 0:
                print(f"  Loaded {i} negative samples")
        
        random.shuffle(self.buf)
        self.max_len = max_len
        self.short_prob = short_prob
        print(f"Dataset created with {len(self.buf)} samples")

    def __len__(self):
        return len(self.buf)

    def __getitem__(self, idx):
        t, y = self.buf[idx]
        
        # Always use fixed max_len for consistent tensor sizes
        if len(t) <= self.max_len:
            crop = t
        else:
            start = random.randrange(0, len(t) - self.max_len + 1)
            crop = t[start:start + self.max_len]
        
        return to_bytes(crop, max_len=self.max_len), torch.tensor(y, dtype=torch.long)

def train():
    """Main training function"""
    print("Starting training...")
    
    # Use MPS if available, otherwise CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("Using CPU")
    model = TinyByteCNN_EN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Create dataset
    ds = BookBin(english_iter(), nonenglish_iter())
    dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    
    print("Training started...")
    model.train()
    
    for step, (x, y) in enumerate(dl, 1):
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % 100 == 0:
            with torch.no_grad():
                p = F.softmax(logits, -1)[:, 1].mean().item()
                acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"Step {step:4d}  Loss {loss.item():.4f}  Acc {acc:.3f}  Mean P(EN) {p:.3f}")
        
        if step >= 2000:  # Quick training for demo
            break
    
    # Save model
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/innit.pt")
    print("Model saved to artifacts/innit.pt")

if __name__ == "__main__":
    train()