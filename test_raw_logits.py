#!/usr/bin/env python3
"""
Test raw logits output from tinygrad vs ONNX
"""

import numpy as np
import onnxruntime as ort
from innit_tinygrad import InnitTinygrad

def compare_raw_logits():
    """Compare raw logits before softmax"""
    print("🔍 Comparing raw logits")
    print("=" * 30)
    
    test_text = "Bonjour le monde!"
    
    # Prepare input
    bytes_data = test_text.encode("utf-8", errors="ignore")[:256]
    padded = np.zeros(256, dtype=np.int64)
    padded[:len(bytes_data)] = list(bytes_data)
    input_array = padded.reshape(1, -1)
    
    print(f"Test text: '{test_text}'")
    print(f"Input shape: {input_array.shape}")
    print(f"Input sample: {input_array[0, :10]}")
    
    # ONNX prediction
    onnx_session = ort.InferenceSession(
        "/Users/mitchellcurrie/.innit/model.onnx", 
        providers=["CPUExecutionProvider"]
    )
    onnx_logits = onnx_session.run(["logits"], {"input_bytes": input_array})[0][0]
    
    print(f"\nONNX raw logits: {onnx_logits}")
    
    # Apply softmax manually
    onnx_exp = np.exp(onnx_logits - np.max(onnx_logits))
    onnx_probs = onnx_exp / np.sum(onnx_exp)
    print(f"ONNX probs: {onnx_probs}")
    print(f"ONNX prediction: {'EN' if onnx_probs[1] > 0.5 else 'NON-EN'}")
    
    # Tinygrad prediction
    tg_detector = InnitTinygrad()
    
    # Get raw logits from tinygrad
    from tinygrad import Tensor
    input_tensor = Tensor(input_array)
    tg_logits_tensor = tg_detector.model(input_tensor)
    tg_logits = tg_logits_tensor.numpy()[0]
    
    print(f"\nTinygrad raw logits: {tg_logits}")
    
    # Apply softmax manually
    tg_exp = np.exp(tg_logits - np.max(tg_logits))
    tg_probs = tg_exp / np.sum(tg_exp)
    print(f"Tinygrad probs: {tg_probs}")
    print(f"Tinygrad prediction: {'EN' if tg_probs[1] > 0.5 else 'NON-EN'}")
    
    print(f"\nLogit difference: {abs(onnx_logits[0] - tg_logits[0]):.6f}, {abs(onnx_logits[1] - tg_logits[1]):.6f}")
    print(f"Prob difference: {abs(onnx_probs[0] - tg_probs[0]):.6f}, {abs(onnx_probs[1] - tg_probs[1]):.6f}")

if __name__ == "__main__":
    compare_raw_logits()