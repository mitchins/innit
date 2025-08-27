import torch
import torch.nn.functional as F
from train_innit import TinyByteCNN_EN, to_bytes, PAD
from pathlib import Path
import sys

def load_model(path="artifacts/innit.pt"):
    """Load trained model from checkpoint"""
    model = TinyByteCNN_EN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def windows(text, win=2048, stride=2048):
    """Generate byte windows from text"""
    b = text.encode("utf-8", "ignore")
    for i in range(0, max(1, len(b)), stride):
        yield b[i:i+win]

def score_book(model, text, win=2048):
    """Score entire book using windowed voting"""
    ps = []
    
    with torch.no_grad():
        for w in windows(text, win, win):
            # Create padded input tensor
            x = torch.full((1, win), PAD, dtype=torch.long)
            if w:
                w_len = min(len(w), win)
                x[0, :w_len] = torch.tensor(list(w[:w_len]), dtype=torch.long)
            
            # Get probability of English
            logits = model(x)
            p_en = F.softmax(logits, -1)[0, 1].item()
            ps.append(p_en)
    
    if not ps:
        return {"mean_pEN": 0.0, "hi>=0.99": 0.0, "label": "UNCERTAIN", "n_windows": 0}
    
    # Aggregate predictions
    mean_p = sum(ps) / len(ps)
    frac_hi = sum(1 for p in ps if p >= 0.99) / len(ps)
    
    # Determine label with conservative thresholds
    if mean_p >= 0.995 and frac_hi >= 0.90:
        label = "ENGLISH"
    elif mean_p <= 0.01:
        label = "NOT-EN"
    else:
        label = "UNCERTAIN"
    
    return {
        "mean_pEN": mean_p,
        "hi>=0.99": frac_hi,
        "label": label,
        "n_windows": len(ps)
    }

def main():
    """CLI for evaluating text files"""
    if len(sys.argv) < 2:
        print("Usage: python eval_innit.py <text_file>")
        sys.exit(1)
    
    text_path = Path(sys.argv[1])
    if not text_path.exists():
        print(f"File not found: {text_path}")
        sys.exit(1)
    
    # Load text
    try:
        text = text_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Load model
    try:
        model = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure to train the model first with: python train_innit.py")
        sys.exit(1)
    
    # Score the book
    result = score_book(model, text)
    
    # Print results
    print(f"File: {text_path}")
    print(f"Language: {result['label']}")
    print(f"Mean P(English): {result['mean_pEN']:.4f}")
    print(f"High confidence windows (>=99%): {result['hi>=0.99']:.2%}")
    print(f"Number of windows: {result['n_windows']}")

if __name__ == "__main__":
    main()