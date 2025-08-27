"""ONNX runtime utilities for lightweight deployment"""

import json
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

PAD = 256

def to_bytes_numpy(text_bytes, max_len=2048):
    """Convert bytes to padded numpy array for ONNX"""
    x = np.full((1, max_len), PAD, dtype=np.int64)
    if text_bytes:
        length = min(len(text_bytes), max_len)
        x[0, :length] = list(text_bytes[:length])
    return x

class ONNXInnitRunner:
    """ONNX runtime wrapper for innit model"""
    
    def __init__(self, model_path):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is required for ONNX inference. Install with: pip install onnxruntime")
        
        self.session = ort.InferenceSession(
            str(model_path), 
            providers=["CPUExecutionProvider"]
        )
        
    def predict_proba(self, text_bytes, max_len=2048):
        """Get English probability for byte sequence"""
        x = to_bytes_numpy(text_bytes, max_len)
        logits = self.session.run(None, {"tokens": x})[0]
        
        # Softmax to get probabilities
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        return float(probs[0, 1])  # P(English)

def score_text_onnx(runner, text, window_size=2048, stride=2048):
    """Score text using ONNX model with windowing"""
    text_bytes = text.encode("utf-8", "ignore")
    
    if not text_bytes:
        return {
            "label": "UNCERTAIN",
            "mean_pEN": 0.0,
            "hi>=0.99": 0.0,
            "windows": 0
        }
    
    # Process in windows
    probabilities = []
    for i in range(0, max(1, len(text_bytes)), stride):
        window_bytes = text_bytes[i:i + window_size]
        p_en = runner.predict_proba(window_bytes, window_size)
        probabilities.append(p_en)
    
    # Aggregate results
    mean_p = sum(probabilities) / len(probabilities)
    frac_hi = sum(1 for p in probabilities if p >= 0.99) / len(probabilities)
    
    # Determine label
    if mean_p >= 0.995 and frac_hi >= 0.90:
        label = "ENGLISH"
    elif mean_p <= 0.01:
        label = "NOT-EN"
    else:
        label = "UNCERTAIN"
    
    return {
        "label": label,
        "mean_pEN": mean_p,
        "hi>=0.99": frac_hi,
        "windows": len(probabilities)
    }