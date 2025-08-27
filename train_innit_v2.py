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

def collate_fixed_len(batch, L=2048):
    """Fixed-length collation for consistent batching"""
    xs, ys = [], []
    for text_bytes, y in batch:
        # text_bytes is already bytes from __getitem__
        xs.append(to_tokens_bytes(text_bytes, L))
        ys.append(y)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)

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

def stream_cc100(lang, max_items):
    """Stream from CC-100 dataset for specific language"""
    try:
        ds = load_dataset("statmt/cc100", lang, split="train", streaming=True)
        count = 0
        for r in ds:
            if count >= max_items:
                break
            t = r.get("text", "")
            if len(t) >= 50:
                yield t
                count += 1
    except Exception as e:
        print(f"  Failed to load CC-100 {lang}: {e}")

def stream_wikipedia_en(max_items):
    """Stream English Wikipedia as clean English source"""
    try:
        ds = load_dataset("wikipedia", "20231101.en", split="train", streaming=True)
        count = 0
        for r in ds:
            if count >= max_items:
                break
            t = r.get("text", "")
            if len(t) >= 100:  # Longer threshold for Wikipedia
                yield t
                count += 1
    except Exception as e:
        print(f"  Failed to load Wikipedia EN: {e}")

def english_iter(max_items=25000):
    """Generate clean English text samples"""
    print("Loading English samples...")
    count = 0
    
    # Try papluca language identification first (small but clean)
    try:
        print("  Loading from language identification dataset...")
        ds = load_dataset("papluca/language-identification", streaming=True)
        for r in ds["train"]:
            if count >= max_items // 2:  # Use half quota
                break
            if r.get("labels") == 0:  # English is label 0
                yield r["text"]
                count += 1
                if count % 2000 == 0:
                    print(f"    Loaded {count} samples from papluca")
    except Exception as e:
        print(f"  Failed language identification: {e}")
    
    # Add Wikipedia English for diversity
    try:
        print("  Loading from Wikipedia English...")
        wiki_count = 0
        for text in stream_wikipedia_en(max_items - count):
            yield text
            count += 1
            wiki_count += 1
            if wiki_count % 1000 == 0:
                print(f"    Loaded {wiki_count} Wikipedia articles")
            if count >= max_items:
                break
    except Exception as e:
        print(f"  Failed Wikipedia: {e}")
    
    # Fallback synthetic samples if needed
    if count < max_items // 4:
        print("  Adding synthetic English samples...")
        english_samples = [
            "The quick brown fox jumps over the lazy dog. This is a test of English language detection systems.",
            "To be or not to be, that is the question. Shakespeare wrote many famous plays in the English language.",
            "In the beginning was the Word, and the Word was with God. This biblical text is in English.",
            "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty and dedicated to the proposition that all men are created equal.",
            "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness. Charles Dickens wrote this opening in English.",
            "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.",
            "Happy families are all alike; every unhappy family is unhappy in its own way. This famous opening line is from a Russian novel translated into English.",
            "All children, except one, grow up. They soon know that they will grow up, and the way Wendy knew was this. Peter Pan is a classic English children's story.",
            "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat.",
            "Mr. and Mrs. Dursley of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious.",
        ] * 500
        
        for sample in english_samples:
            if count >= max_items:
                break
            yield sample
            count += 1
    
    print(f"  Total English samples loaded: {count}")

def nonenglish_iter(max_items=25000):
    """Generate clean non-English text samples"""
    print("Loading non-English samples...")
    count = 0
    
    # Primary non-English languages from CC-100
    non_en_langs = ["fr", "de", "es", "it", "ru", "zh", "ar", "hi", "ja", "pt", "pl", "tr", "nl", "sv", "ko"]
    
    # Add romanized versions for hard negatives  
    romanized_langs = ["ar_rom", "ru_rom", "hi_rom"]
    
    per_lang = max(1, max_items // (len(non_en_langs) + len(romanized_langs)))
    
    # Load from each non-English language
    for lang in non_en_langs:
        print(f"  Loading {lang} samples...")
        lang_count = 0
        for text in stream_cc100(lang, per_lang):
            yield text
            count += 1
            lang_count += 1
            if lang_count % 500 == 0:
                print(f"    {lang}: {lang_count} samples")
            if count >= max_items:
                break
        if count >= max_items:
            break
    
    # Add romanized hard negatives
    if count < max_items:
        for lang in romanized_langs:
            print(f"  Loading {lang} samples...")
            lang_count = 0
            for text in stream_cc100(lang, min(per_lang, max_items - count)):
                yield text
                count += 1
                lang_count += 1
                if count >= max_items:
                    break
            if count >= max_items:
                break
    
    # Try papluca for additional non-English if needed
    if count < max_items // 2:
        try:
            print("  Adding from papluca non-English...")
            ds = load_dataset("papluca/language-identification", streaming=True)
            for r in ds["train"]:
                if count >= max_items:
                    break
                if r.get("labels") != 0:  # Non-English
                    yield r["text"]
                    count += 1
        except Exception as e:
            print(f"  Failed papluca non-English: {e}")
    
    # Synthetic fallback
    if count < max_items // 4:
        print("  Adding synthetic non-English samples...")
        non_english_samples = [
            "Bonjour, comment allez-vous aujourd'hui? C'est un beau jour pour apprendre le français.",
            "Hola, ¿cómo estás? Me llamo María y vivo en España. El español es un idioma muy bonito.",
            "Guten Tag, ich heiße Hans und komme aus Deutschland. Deutsch ist meine Muttersprache.",
            "Ciao, come stai? Io sto bene, grazie. L'italiano è una lingua molto musicale e bella.",
            "Привет, как дела? Меня зовут Иван, и я говорю по-русски. Русский язык очень богатый.",
            "こんにちは、元気ですか？私の名前は田中です。日本語は美しい言語だと思います。",
            "안녕하세요, 어떻게 지내세요? 제 이름은 김철수입니다. 한국어는 흥미로운 언어입니다.",
            "你好，你好吗？我的名字是李明。中文是世界上使用人数最多的语言之一。",
            "مرحبا، كيف حالك؟ اسمي أحمد وأنا أتكلم العربية. العربية لغة جميلة جدا.",
            "नमस्ते, आप कैसे हैं? मेरा नाम राम है और मैं हिंदी बोलता हूं।",
        ] * 300
        
        for sample in non_english_samples:
            if count >= max_items:
                break
            yield sample
            count += 1
    
    print(f"  Total non-English samples loaded: {count}")

class CleanDataset(torch.utils.data.Dataset):
    """Clean dataset with fixed-size tensors"""
    
    def __init__(self, pos_iter, neg_iter, max_len=2048):
        print("Building dataset...")
        self.buf = []
        self.max_len = max_len
        
        # Load positive samples (English)
        for text in pos_iter:
            self.buf.append((text, 1))
        
        # Load negative samples (non-English)
        for text in neg_iter:
            self.buf.append((text, 0))
        
        # Shuffle for good training
        random.shuffle(self.buf)
        
        pos_count = sum(1 for _, y in self.buf if y == 1)
        neg_count = len(self.buf) - pos_count
        
        print(f"Dataset created: {len(self.buf)} samples ({pos_count} EN, {neg_count} non-EN)")

    def __len__(self):
        return len(self.buf)

    def __getitem__(self, idx):
        text, y = self.buf[idx]
        
        # Random crop to max_len
        text_bytes = text.encode("utf-8", "ignore")
        if len(text_bytes) <= self.max_len:
            crop_bytes = text_bytes
        else:
            start = random.randrange(0, len(text_bytes) - self.max_len + 1)
            crop_bytes = text_bytes[start:start + self.max_len]
        
        return crop_bytes, y

def train():
    """Main training function with clean pipeline"""
    print("Starting training with clean data pipeline...")
    
    # Use MPS if available, otherwise CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Create model
    model = TinyByteCNN_EN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Create clean dataset
    print("\n" + "="*50)
    ds = CleanDataset(english_iter(), nonenglish_iter())
    print("="*50 + "\n")
    
    # Use custom collation for fixed tensor sizes
    dl = DataLoader(
        ds, 
        batch_size=32, 
        shuffle=True, 
        drop_last=True, 
        collate_fn=lambda b: collate_fixed_len(b, L=2048),
        num_workers=0
    )
    
    print("Training started...")
    model.train()
    
    # Optional: fastText distillation (license-safe)
    ft_model = None
    try:
        import fasttext
        print("Loading fastText lid.176 for distillation...")
        ft_model = fasttext.load_model("lid.176.bin")  # CC-BY-SA licensed
        print("FastText loaded - will use distillation")
    except:
        print("FastText not available - training without distillation")
    
    for step, (x, y) in enumerate(dl, 1):
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        ce_loss = F.cross_entropy(logits, y)
        loss = ce_loss
        
        # Optional knowledge distillation from fastText
        if ft_model is not None and step % 5 == 0:  # Every 5 steps for efficiency
            with torch.no_grad():
                t_probs = []
                for i in range(x.size(0)):
                    raw_bytes = bytes(int(v) for v in x[i].tolist() if v != PAD)
                    text_str = raw_bytes.decode("utf-8", "ignore")
                    if text_str.strip():
                        labels, probs = ft_model.predict(text_str, k=1)
                        is_en = labels and labels[0] == "__label__en"
                        t_probs.append(float(probs[0] if is_en else 0.0))
                    else:
                        t_probs.append(0.0)
                
                if t_probs:
                    t = torch.tensor(t_probs, dtype=torch.float32, device=x.device)
                    s = torch.softmax(logits, -1)[:, 1]
                    kd_loss = F.binary_cross_entropy(s, t)
                    loss = 0.7 * ce_loss + 0.3 * kd_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % 50 == 0:
            with torch.no_grad():
                pred_probs = F.softmax(logits, -1)[:, 1].mean().item()
                acc = (logits.argmax(-1) == y).float().mean().item()
                
            print(f"Step {step:4d}  Loss {loss.item():.4f}  CE {ce_loss.item():.4f}  Acc {acc:.3f}  P(EN) {pred_probs:.3f}")
        
        if step >= 1500:  # Quick training for demo
            print("Training completed!")
            break
    
    # Save model
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/innit.pt")
    print("Model saved to artifacts/innit.pt")
    
    return model

if __name__ == "__main__":
    model = train()