---
language: multilingual
license: mit
library_name: pytorch
tags:
- text-classification
- language-detection
- byte-level
- multilingual
- english-detection
- cnn
pipeline_tag: text-classification
datasets:
- custom
metrics:
- accuracy
model-index:
- name: innit
  results:
  - task:
      type: text-classification
      name: English vs Non-English Detection
    metrics:
    - type: accuracy
      value: 99.94
      name: Validation Accuracy
    - type: accuracy  
      value: 100.0
      name: Challenge Set Accuracy
---

# innit: Fast English vs Non-English Text Detection

A lightweight byte-level CNN for fast binary language detection (English vs Non-English).

## Model Details

- **Model Type**: Byte-level Convolutional Neural Network
- **Task**: Binary text classification (English vs Non-English)
- **Architecture**: TinyByteCNN_EN with 6 convolutional blocks
- **Parameters**: 156,642
- **Input**: Raw UTF-8 bytes (max 256 bytes)
- **Output**: Binary classification (0=Non-English, 1=English)

## Performance

- **Validation Accuracy**: 99.94%
- **Challenge Set Accuracy**: 100% (14/14 test cases)
- **Inference Speed**: Sub-millisecond on modern CPUs
- **Model Size**: ~600KB

## Supported Languages

Trained to distinguish English from 52+ languages across diverse scripts:
- **Latin scripts**: Spanish, French, German, Italian, Dutch, Portuguese, etc.
- **CJK scripts**: Chinese (Simplified/Traditional), Japanese, Korean
- **Cyrillic scripts**: Russian, Ukrainian, Bulgarian, Serbian
- **Other scripts**: Arabic, Hindi, Bengali, Thai, Hebrew, etc.

## Architecture

```
TinyByteCNN_EN:
├── Embedding: 257 → 80 dimensions (256 bytes + padding)
├── 6x Convolutional Blocks:
│   ├── Conv1D (kernel=3, residual connections)
│   ├── GELU activation
│   ├── BatchNorm1D  
│   └── Dropout (0.15)
├── Enhanced Pooling: mean + max + std
└── Classification Head: 240 → 80 → 2
```

## Training Data

- **Total samples**: 17,543 balanced samples
- **English**: 8,772 samples from diverse sources
- **Non-English**: 8,771 samples across 52+ languages
- **Text lengths**: 3-276 characters (optimized for short texts)
- **Special coverage**: Emoji handling, mathematical formulas, scientific notation

## Quick Start

### Option 1: ONNX Runtime (Recommended)
```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("model.onnx")

def predict(text):
    # Prepare input
    bytes_data = text.encode('utf-8', errors='ignore')[:256]
    padded = np.zeros(256, dtype=np.int64)
    padded[:len(bytes_data)] = list(bytes_data)
    
    # Run inference
    outputs = session.run(['logits'], {'input_bytes': padded.reshape(1, -1)})
    logits = outputs[0][0]
    
    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return probs[1]  # English probability

# Examples
print(predict("Hello world!"))           # ~1.0 (English)
print(predict("Bonjour le monde"))       # ~0.0 (French)
print(predict("Check our sale! 🎉"))     # ~1.0 (English with emoji)
```

### Option 2: Python Package
```bash
# Install the utility package
pip install innit-detector

# CLI usage
innit "Hello world!"                    # → English (confidence: 0.974)
innit --download                        # Download model first
innit "Hello" "Bonjour" "你好"          # Multiple texts

# Library usage
from innit_detector import InnitDetector
detector = InnitDetector()
result = detector.predict("Hello world!")
print(result['is_english'])  # True
```

### Option 3: PyTorch (Advanced)
```python
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import numpy as np

# Load model (requires TinyByteCNN_EN class definition)
state_dict = load_file("model.safetensors")
model = TinyByteCNN_EN(emb=80, blocks=6, dropout=0.15)
model.load_state_dict(state_dict)
model.eval()

def predict(text):
    bytes_data = text.encode('utf-8', errors='ignore')[:256]
    padded = np.zeros(256, dtype=np.long)
    padded[:len(bytes_data)] = list(bytes_data)
    
    with torch.no_grad():
        logits = model(torch.tensor(padded).unsqueeze(0))
        probs = F.softmax(logits, dim=1)
        return probs[0][1].item()
```

## ONNX Support

ONNX version available for cross-platform deployment:
- `model.onnx` - Full precision (FP32) for maximum compatibility

## Challenge Set Results

Perfect 100% accuracy on comprehensive test cases:
- Ultra-short texts: "Good morning!" ✅
- Emoji handling: "Check out our sale! 🎉" ✅  
- Mathematical formulas: "x = (-b ± √(b²-4ac))/2a" ✅
- Scientific notation: "CO₂ + H₂O → C₆H₁₂O₆" ✅
- Diverse scripts: Arabic, CJK, Cyrillic, Devanagari ✅
- English-like languages: Dutch, German ✅

## Limitations

- Binary classification only (English vs Non-English)
- Optimized for texts up to 256 UTF-8 bytes
- May have reduced accuracy on very rare languages not in training data
- Not suitable for multilingual text (mixed languages in single input)

## License

MIT License - free for commercial use.