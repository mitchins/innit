import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

PAD = 256

def to_tokens_bytes(byts: bytes, L=2048):
    """Fast byte tokenization using NumPy"""
    x = np.full((L,), PAD, dtype=np.int64)
    n = min(L, len(byts))
    if n:
        x[:n] = np.frombuffer(byts[:n], dtype=np.uint8)
    return torch.from_numpy(x)

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

def simple_english_iter(max_items=10000):
    """Simple English text generator using papluca dataset"""
    print("Loading English samples from papluca/language-identification...")
    
    try:
        ds = load_dataset("papluca/language-identification", streaming=True)
        count = 0
        for r in ds["train"]:
            if count >= max_items:
                break
            if r.get("labels") == 0:  # English is label 0
                text = r.get("text", "")
                if len(text) >= 20:  # Minimum length filter
                    yield text
                    count += 1
                    if count % 1000 == 0:
                        print(f"  Loaded {count} English samples")
    except Exception as e:
        print(f"Error loading English data: {e}")
    
    # Synthetic English fallback
    print("Adding synthetic English samples...")
    synthetic_en = [
        "The quick brown fox jumps over the lazy dog every morning in the beautiful countryside.",
        "Shakespeare wrote many famous plays including Hamlet, Romeo and Juliet, and Macbeth in English.",
        "The United States Declaration of Independence was written in English by Thomas Jefferson.",
        "Charles Darwin published his theory of evolution in a book called The Origin of Species.",
        "The Internet has revolutionized communication and information sharing around the world.",
        "Machine learning and artificial intelligence are transforming many industries today.",
        "Climate change is one of the most pressing challenges facing humanity in the 21st century.",
        "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
        "DNA contains the genetic instructions for the development and function of all living things.",
        "The scientific method involves making observations, forming hypotheses, and testing predictions.",
    ] * 1500
    
    for text in synthetic_en:
        yield text

def simple_nonenglish_iter(max_items=10000):
    """Simple non-English text generator"""
    print("Loading non-English samples from papluca/language-identification...")
    
    try:
        ds = load_dataset("papluca/language-identification", streaming=True)
        count = 0
        for r in ds["train"]:
            if count >= max_items:
                break
            if r.get("labels") != 0:  # Non-English
                text = r.get("text", "")
                if len(text) >= 20:
                    yield text
                    count += 1
                    if count % 1000 == 0:
                        print(f"  Loaded {count} non-English samples")
    except Exception as e:
        print(f"Error loading non-English data: {e}")
    
    # Synthetic non-English samples
    print("Adding synthetic non-English samples...")
    synthetic_non_en = [
        "Bonjour, comment allez-vous aujourd'hui? J'espère que vous passez une excellente journée en France.",
        "Hola, ¿cómo estás? Espero que tengas un buen día. El español es un idioma hermoso hablado por millones.",
        "Guten Tag, wie geht es Ihnen? Deutschland ist ein Land mit einer reichen Geschichte und Kultur.",
        "Ciao, come stai oggi? L'Italia è famosa per la sua cucina, arte e architettura meravigliosa.",
        "Привет, как дела? Россия - самая большая страна в мире с богатой литературной традицией.",
        "こんにちは、元気ですか？日本は美しい国で、技術と伝統文化で有名です。桜の季節が特に美しいです。",
        "안녕하세요, 어떻게 지내세요? 한국은 K-pop과 한국 드라마로 세계적으로 유명해졌습니다.",
        "你好，你好吗？中国有着五千年的悠久历史和灿烂的文化。长城是世界著名的建筑奇迹。",
        "مرحبا، كيف حالك اليوم؟ العالم العربي له تاريخ غني وثقافة متنوعة. الأدب العربي عريق وجميل.",
        "नमस्ते, आप कैसे हैं? भारत एक विविधताओं से भरा देश है जहां कई भाषाएं और संस्कृतियां फलती-फूलती हैं।",
        "Olá, como você está? O Brasil é um país lindo com praias incríveis e uma cultura vibrante.",
        "Hej, hur mår du? Sverige är känt för sina vackra skogar och det moderna välfärdssystemet.",
        "Hallo, hoe gaat het? Nederland is beroemd om zijn tulpen, windmolens en fietscultuur.",
        "Cześć, jak się masz? Polska ma bogatą historię i piękne średniowieczne miasta.",
        "Merhaba, nasılsınız? Türkiye Avrupa ve Asya arasında köprü görevi gören güzel bir ülkedir."
    ] * 800
    
    for text in synthetic_non_en:
        yield text

class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset with byte preprocessing"""
    
    def __init__(self, english_iter, nonenglish_iter, max_len=2048):
        print("Building simple dataset...")
        self.data = []
        self.max_len = max_len
        
        # Collect English samples
        en_count = 0
        for text in english_iter:
            self.data.append((text, 1))
            en_count += 1
        
        # Collect non-English samples  
        non_en_count = 0
        for text in nonenglish_iter:
            self.data.append((text, 0))
            non_en_count += 1
        
        # Shuffle
        random.shuffle(self.data)
        
        print(f"Dataset created: {len(self.data)} samples ({en_count} EN, {non_en_count} non-EN)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        
        # Convert to bytes and crop
        text_bytes = text.encode("utf-8", "ignore")
        if len(text_bytes) > self.max_len:
            start = random.randrange(0, len(text_bytes) - self.max_len + 1)
            text_bytes = text_bytes[start:start + self.max_len]
        
        # Convert to tensor
        x = to_tokens_bytes(text_bytes, self.max_len)
        y = torch.tensor(label, dtype=torch.long)
        
        return x, y

def train():
    """Simple training function"""
    print("Starting simple training...")
    
    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = "cpu" 
        print("Using CPU")
    
    # Create model
    model = TinyByteCNN_EN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Create dataset
    ds = SimpleDataset(simple_english_iter(), simple_nonenglish_iter())
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
        
        if step % 50 == 0:
            with torch.no_grad():
                probs = F.softmax(logits, -1)
                pred_prob_en = probs[:, 1].mean().item()
                acc = (logits.argmax(-1) == y).float().mean().item()
            
            print(f"Step {step:4d}  Loss {loss.item():.4f}  Acc {acc:.3f}  P(EN) {pred_prob_en:.3f}")
        
        if step >= 1000:  # Quick training
            print("Training completed!")
            break
    
    # Save model
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/innit.pt")
    print("Model saved to artifacts/innit.pt")
    
    return model

if __name__ == "__main__":
    model = train()