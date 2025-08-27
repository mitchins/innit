#!/usr/bin/env python3
"""
Compare the fixed implementation with ONNX
"""

import numpy as np
import onnxruntime as ort
from innit_tinygrad_fixed import InnitTinygradFixed

def compare_logits():
    """Compare raw logits between ONNX and fixed tinygrad"""
    print("🔍 Comparing ONNX vs Fixed Tinygrad")
    print("=" * 40)
    
    # Test cases
    test_cases = [
        "Hello world!",
        "Bonjour le monde!",
        "你好世界！"
    ]
    
    # Initialize detectors
    onnx_session = ort.InferenceSession(
        "/Users/mitchellcurrie/.innit/model.onnx", 
        providers=["CPUExecutionProvider"]
    )
    tg_detector = InnitTinygradFixed()
    
    print(f"{'Text':<20} {'ONNX Logits':<25} {'TG Logits':<25} {'Difference'}")
    print("-" * 80)
    
    for text in test_cases:
        # Prepare input
        bytes_data = text.encode("utf-8", errors="ignore")[:256]
        padded = np.zeros(256, dtype=np.int64)
        padded[:len(bytes_data)] = list(bytes_data)
        input_array = padded.reshape(1, -1)
        
        # ONNX prediction
        onnx_logits = onnx_session.run(["logits"], {"input_bytes": input_array})[0][0]
        
        # Tinygrad prediction
        tg_result = tg_detector.predict(text)
        tg_logits = np.array(tg_result["raw_logits"])
        
        # Calculate difference
        diff = np.abs(onnx_logits - tg_logits)
        
        display_text = text if len(text) <= 15 else text[:12] + "..."
        onnx_str = f"[{onnx_logits[0]:.3f}, {onnx_logits[1]:.3f}]"
        tg_str = f"[{tg_logits[0]:.3f}, {tg_logits[1]:.3f}]"
        diff_str = f"[{diff[0]:.3f}, {diff[1]:.3f}]"
        
        print(f"{display_text:<20} {onnx_str:<25} {tg_str:<25} {diff_str}")
        
        # Apply softmax to both for predictions
        onnx_exp = np.exp(onnx_logits - np.max(onnx_logits))
        onnx_probs = onnx_exp / np.sum(onnx_exp)
        onnx_pred = "EN" if onnx_probs[1] > 0.5 else "NON-EN"
        
        tg_exp = np.exp(tg_logits - np.max(tg_logits))
        tg_probs = tg_exp / np.sum(tg_exp)
        tg_pred = "EN" if tg_probs[1] > 0.5 else "NON-EN"
        
        print(f"{'Predictions:':<20} {onnx_pred:<25} {tg_pred:<25}")
        print()

if __name__ == "__main__":
    compare_logits()