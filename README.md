# innit - Fast English vs Non-English Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/innit-detector.svg)](https://pypi.org/project/innit-detector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight Python package for fast binary language detection (English vs Non-English) with two backends: ultra-fast ONNX Runtime or tiny, hackable TinyGrad.

## Features

- ⚡ **Ultra-fast**: Sub-millisecond (ONNX CPU); TinyGrad for minimal deps
- 🪶 **Lightweight**: ~600KB model  
- 🎯 **Accurate**: 99.94% validation accuracy, 100% on comprehensive challenge set
- 🔧 **Easy**: Simple CLI and Python API with automatic model download
- 🌍 **Universal**: Trained on 52+ languages across diverse scripts
- 📝 **Smart chunking**: Handles texts of any length automatically
- 🚀 **Flexible**: Choose ONNX (speed) or TinyGrad (minimal deps)

## Quick Start

### Installation

- Default (TinyGrad backend included): `pip install innit-detector`
- Fastest CPU backend (ONNX): `pip install innit-detector[onnx]`
- Dev (ONNX + tools): `pip install innit-detector[onnx,dev]`

### CLI Usage

```bash
# Basic usage
innit "Hello world!"                    # → English (confidence: 0.974)

# Download model first (recommended)
innit --download

# Multiple texts
innit "Hello" "Bonjour" "你好" "Привет"  # → EN, NOT-EN, NOT-EN, NOT-EN

# Long text
innit --chunk-strategy ends --ends-pct 0.1 "Very long paragraph ..."
# → Samples the first/last 10% for a quick decision

# JSON output
innit --json "Hello world!"
# → {"language": "en", "is_english": true, "confidence": 0.974, ...}
```

Models
- Assets are cached in `~/.innit/`.
- If missing, run: `innit --download` (downloads ONNX + TinyGrad assets).
- Choose assets: `innit --download --download-backend onnx|tinygrad|both`.

### Python API (quick)

```python
from innit_client import InnitClient, InnitClientConfig

# Initialize (auto-detects installed backend; base install ships TinyGrad)
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

# Document classification with hidden heuristics
result = client.classify_document(long_text)           # auto: picks 'ends' for long docs
result = client.classify_document(long_text, strategy='chunk')  # force chunking
result = client.classify_document(long_text, strategy='ends', ends_pct=0.1)
```

## Supported Languages

Trained to distinguish English from 52+ languages including:

- **Latin scripts**: Spanish, French, German, Italian, Dutch, Portuguese, etc.
- **CJK scripts**: Chinese (Simplified/Traditional), Japanese, Korean  
- **Cyrillic scripts**: Russian, Ukrainian, Bulgarian, Serbian
- **Other scripts**: Arabic, Hindi, Bengali, Thai, Hebrew, Tamil, etc.

## Performance

- **Validation accuracy**: 99.94%
- **Challenge set accuracy**: 100% (14/14 comprehensive test cases)
- **Model size**: ~600KB
- **Inference speed**: ONNX CPU ~0.6ms; TinyGrad ~50–60ms (CPU, batch 1)
- **Memory usage**: Minimal (~10MB RAM)

### Challenge Set Results

Perfect accuracy on diverse test cases:
- ✅ Ultra-short texts: "Good morning!"
- ✅ Emoji handling: "Check our sale! 🎉"  
- ✅ Mathematical formulas: "x = (-b ± √(b²-4ac))/2a"
- ✅ Scientific notation: "CO₂ + H₂O → C₆H₁₂O₆"
- ✅ All major scripts: Arabic, CJK, Cyrillic, Devanagari
- ✅ Tricky cases: Dutch vs English, technical text

## Architecture

- **Model type**: Byte-level Convolutional Neural Network
- **Input**: Raw UTF-8 bytes (up to 256 bytes per chunk)
- **Output**: Binary classification (English vs Non-English)
- **Parameters**: 156,642 (~600KB)
- **Framework**: ONNX Runtime (cross-platform)

## Text Length Handling

| Text Length | Handling | Performance |
|-------------|----------|-------------|
| 0-256 bytes | Single chunk | Optimal |
| 256+ bytes | Auto-chunking | Excellent |
| 2KB+ text | Multiple chunks + voting | Very good |

For texts longer than 256 bytes, innit automatically:
1. Splits text into overlapping chunks
2. Processes each chunk independently  
3. Combines results using weighted averaging and majority voting
4. Reports aggregated confidence score

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Code quality
ruff check innit_detector.py
black innit_detector.py
mypy innit_detector.py

# Run tests
pytest --cov=innit_detector
```

## Model Details

The model is available on [🤗 HuggingFace Hub](https://huggingface.co/Mitchins/innit-language-detection) with:
- PyTorch weights (SafeTensors format)
- ONNX runtime version  
- Complete training details
- Comprehensive documentation

## Contributing

Issues and pull requests welcome! See the [GitHub repository](https://github.com/Mitchins/innit-language-detection) for details.

## License

MIT License - see [LICENSE](LICENSE) file for details.
