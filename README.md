# innit - Fast English vs Non-English Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/innit-detector.svg)](https://pypi.org/project/innit-detector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight Python package for fast binary language detection (English vs Non-English) using ONNX runtime.

## Features

- ⚡ **Ultra-fast**: Sub-millisecond inference on modern CPUs
- 🪶 **Lightweight**: ~600KB model, minimal dependencies  
- 🎯 **Accurate**: 99.94% validation accuracy, 100% on comprehensive challenge set
- 🔧 **Easy**: Simple CLI and Python API with automatic model download
- 🌍 **Universal**: Trained on 52+ languages across diverse scripts
- 📝 **Smart chunking**: Handles texts of any length automatically
- 🚀 **Production-ready**: ONNX deployment, no PyTorch dependency

## Quick Start

### Installation

```bash
pip install innit-detector
```

### CLI Usage

```bash
# Basic usage
innit "Hello world!"                    # → English (confidence: 0.974)

# Download model first (recommended)
innit --download

# Multiple texts
innit "Hello" "Bonjour" "你好" "Привет"  # → EN, NOT-EN, NOT-EN, NOT-EN

# Long text with chunking
innit "Very long paragraph that exceeds the 256-byte model limit..."
# → English (confidence: 0.990), chunked, 5 chunks

# JSON output
innit --json "Hello world!"
# → {"language": "en", "is_english": true, "confidence": 0.974, ...}
```

### Python API

```python
from innit_detector import InnitDetector

# Initialize (downloads model automatically if needed)
detector = InnitDetector()

# Single prediction
result = detector.predict("Hello world!")
print(result['is_english'])    # True
print(result['confidence'])    # 0.974

# Batch prediction
results = detector.predict_batch(["Hello", "Bonjour", "你好"])
for r in results:
    print(f"{r['is_english']} ({r['confidence']:.3f})")

# Control chunking for long texts
result = detector.predict(long_text, chunk_strategy='auto')  # Default
result = detector.predict(long_text, chunk_strategy='truncate')  # Classic
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
- **Model size**: ~600KB (ONNX format)
- **Inference speed**: Sub-millisecond on modern CPUs
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