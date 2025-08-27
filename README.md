# innit — Fast English vs Non‑English Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/innit-detector.svg)](https://pypi.org/project/innit-detector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight Python package for fast binary language detection (English vs Non‑English) with two backends: fast ONNX Runtime or minimal TinyGrad.

## Features

- Fast: sub‑millisecond with ONNX CPU on short texts
- Small: ~600 KB model
- Accurate: 99.94% validation accuracy (challenge set: 100%)
- Simple: CLI and Python API; automatic model download and caching
- Flexible: ONNX for speed, TinyGrad for minimal dependencies

## Quick Start

### Installation

- Default (TinyGrad backend included): `pip install innit-detector`
- Fastest CPU backend (ONNX): `pip install innit-detector[onnx]`
- Development (with tools): `pip install innit-detector[onnx,dev]`

### CLI

```bash
# Basic classification
innit "Hello world!"

# Download and cache model assets to ~/.innit
innit --download

# Multiple texts
innit "Hello" "Bonjour" "你好" "Привет"

# Long text (sample first/last 10%)
innit --chunk-strategy ends --ends-pct 0.1 "Very long paragraph ..."

# JSON output
innit --json "Hello world!"
```

Models
- Assets are cached in `~/.innit/`.
- If missing, run: `innit --download` (downloads ONNX + TinyGrad assets).
- Choose assets: `innit --download --download-backend onnx|tinygrad|both`.

### Python API

```python
from innit import InnitClient, InnitClientConfig

# Initialize (auto-detects installed backend)
client = InnitClient()
# Prefer ONNX explicitly:
# client = InnitClient(InnitClientConfig(backend='onnx'))

# Single prediction
result = client.classify("Hello world!")
print(result['is_english'])    # True
print(result['confidence'])    # 0.974

# Batch prediction
results = client.classify_snippets(["Hello", "Bonjour", "你好"])
for r in results:
    print(f"{r['is_english']} ({r['confidence']:.3f})")

# Document classification with heuristics
result = client.classify_document(long_text)                     # auto: may use 'ends' for long docs
result = client.classify_document(long_text, strategy='chunk')   # force chunking
result = client.classify_document(long_text, strategy='ends', ends_pct=0.1)
```

## Supported Languages

Trained to distinguish English from 52+ languages including:

- **Latin scripts**: Spanish, French, German, Italian, Dutch, Portuguese, etc.
- **CJK scripts**: Chinese (Simplified/Traditional), Japanese, Korean  
- **Cyrillic scripts**: Russian, Ukrainian, Bulgarian, Serbian
- **Other scripts**: Arabic, Hindi, Bengali, Thai, Hebrew, Tamil, etc.

## Performance

- Validation accuracy: 99.94%
- Challenge set accuracy: 100% (14/14)
- Model size: ~600 KB
- Inference speed: ONNX CPU ~0.6 ms; TinyGrad ~50–60 ms (CPU, batch 1)
- Memory usage: ~10 MB

### Challenge Set

Full marks across diverse cases, including ultra‑short text, emoji, formulas, scientific notation, and major scripts (Arabic, CJK, Cyrillic, Devanagari).

## Architecture

- Model: byte‑level CNN
- Input: raw UTF‑8 bytes (up to 256 bytes per chunk)
- Output: binary (English vs Non‑English)
- Parameters: ~156k (~600 KB)
- Runtime: ONNX or TinyGrad

## Text Length Handling

| Text Length | Handling | Performance |
|-------------|----------|-------------|
| 0-256 bytes | Single chunk | Optimal |
| 256+ bytes | Auto-chunking | Excellent |
| 2KB+ text | Multiple chunks + voting | Very good |

For texts longer than 256 bytes, innit can:
1. Split into overlapping chunks
2. Process each chunk independently
3. Combine results using weighted averaging and majority voting
4. Report an aggregated confidence score

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Code quality
ruff check .
black innit innit_*.py
mypy innit innit_detector.py innit_tinygrad.py innit_tinygrad_fixed.py innit_client.py

# Run tests
pytest
```

## Model Details

Models and weights are hosted on Hugging Face: https://huggingface.co/Mitchins/innit-language-detection

- PyTorch weights (SafeTensors)
- ONNX model
- Training notes and documentation

## Contributing

Issues and pull requests welcome! See the [GitHub repository](https://github.com/Mitchins/innit-language-detection) for details.

## License

MIT License - see [LICENSE](LICENSE) file for details.
