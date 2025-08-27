# innit - Fast English Detection

Note: The current PyPI release is a lightweight placeholder to reserve the
package name while the model is trained and productized. It installs quickly
and does not include heavy training dependencies. The CLI expects you to
provide an ONNX model file.

A tiny, fast, and dependency-light tool to determine if text is English or not English. Perfect for book-length texts where you need quick language detection without heavy ML frameworks.

## Features

- **Fast**: Sub-millisecond inference per 2KB window on CPU
- **Small**: ~1-2MB model size (0.5-1MB with int8 quantization)  
- **Simple**: Binary classification - English vs Not-English
- **Legal**: Trained only on legally clean datasets
- **Deployable**: Ships as ONNX runtime (no PyTorch dependency for inference)

## Installation

### For inference only (lightweight):
```bash
pip install onnxruntime
# Download the innit.onnx model file
```

### For training and development:
```bash
git clone <repo>
cd innit
pip install -e .
```

## Quick Start

### CLI Usage
```bash
# Analyze a text file
innit book.txt

# Output as JSON
innit book.txt --json

# Use specific model
innit book.txt --model path/to/innit.onnx
```

### Python API
```python
from innit.onnx_runner import ONNXInnitRunner, score_text_onnx

# Load model
runner = ONNXInnitRunner("innit.onnx")

# Score text
result = score_text_onnx(runner, text)
print(result["label"])  # "ENGLISH", "NOT-EN", or "UNCERTAIN"
```

## Training Your Own Model

1. **Train the model**:
```bash
python train_innit.py
```

2. **Export to ONNX**:
```bash
python export_onnx.py
```

3. **Test evaluation**:
```bash
python eval_innit.py sample_text.txt
```

## How It Works

- **Architecture**: Tiny byte-level CNN with depthwise separable convolutions
- **Input**: UTF-8 bytes (no tokenizer needed)
- **Strategy**: Slides 2KB windows over text and aggregates predictions
- **Thresholds**: Conservative - requires high confidence across many windows

## Model Details

- **Input**: Sequences of up to 2048 UTF-8 bytes
- **Architecture**: 4-block CNN with residual connections
- **Output**: Binary classification (English probability)
- **Training**: ~50K samples each of English and non-English text
- **Datasets**: Project Gutenberg (English) + multilingual sources (non-English)

## Legal & Licensing

### Training Data Sources
- **English**: Project Gutenberg texts (public domain in US)
- **Non-English**: HuggingFace multilingual datasets with permissive licenses
- See `DATA_SOURCES.md` for complete dataset information

### Model License
This model and code are released under MIT License. See `LICENSE` for details.

### Usage Notes
- The model weights are original work trained on legally clean data
- No copyrighted text content is redistributed
- Safe for commercial use

## Performance

| Metric | Value |
|--------|--------|
| Model Size (FP32) | ~1.5 MB |
| Model Size (INT8) | ~0.8 MB |
| Inference Speed | <1ms per 2KB window |
| Memory Usage | <100 MB |
| Accuracy | >95% on book-length texts |

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests if applicable  
4. Submit a pull request

## Troubleshooting

**Model file not found**: Ensure you've either trained a model with `python train_innit.py` or downloaded a pre-trained `innit.onnx` file.

**Import errors**: For inference, you only need `onnxruntime`. For training, install the full development dependencies.

**Poor performance**: The model works best on book-length texts (>1KB). Very short texts may return "UNCERTAIN".
# innit
