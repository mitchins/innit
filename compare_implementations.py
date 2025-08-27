#!/usr/bin/env python3
"""
Compare ONNX vs Tinygrad implementations of innit
"""

import time
import numpy as np
from innit_detector import InnitDetector
from innit_tinygrad import InnitTinygrad

def compare_implementations():
    """Compare ONNX and tinygrad implementations"""
    print("🔄 Comparing ONNX vs Tinygrad implementations")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ("Hello world!", True),
        ("Bonjour le monde!", False),
        ("你好世界！", False),
        ("This is English text.", True),
        ("Este es español.", False),
    ]
    
    # Initialize both detectors
    print("Loading ONNX detector...")
    onnx_detector = InnitDetector()
    
    print("Loading Tinygrad detector...")
    tg_detector = InnitTinygrad()
    
    print("\nComparison Results:")
    print("-" * 50)
    print(f"{'Text':<30} {'Expected':<10} {'ONNX':<15} {'Tinygrad':<15} {'Match':<8}")
    print("-" * 50)
    
    matches = 0
    total_onnx_time = 0
    total_tg_time = 0
    
    for text, expected_english in test_cases:
        # ONNX prediction
        start = time.time()
        onnx_result = onnx_detector.predict(text)
        onnx_time = (time.time() - start) * 1000
        total_onnx_time += onnx_time
        
        # Tinygrad prediction
        start = time.time()
        tg_result = tg_detector.predict(text)
        tg_time = (time.time() - start) * 1000
        total_tg_time += tg_time
        
        # Compare results
        onnx_pred = "EN" if onnx_result["is_english"] else "NON-EN"
        tg_pred = "EN" if tg_result["is_english"] else "NON-EN"
        expected_str = "EN" if expected_english else "NON-EN"
        
        match = "✅" if onnx_result["is_english"] == tg_result["is_english"] else "❌"
        if onnx_result["is_english"] == tg_result["is_english"]:
            matches += 1
        
        # Format for display
        display_text = text if len(text) <= 25 else text[:22] + "..."
        onnx_display = f"{onnx_pred} ({onnx_result['confidence']:.3f})"
        tg_display = f"{tg_pred} ({tg_result['confidence']:.3f})"
        
        print(f"{display_text:<30} {expected_str:<10} {onnx_display:<15} {tg_display:<15} {match:<8}")
    
    print("-" * 50)
    print(f"Agreement: {matches}/{len(test_cases)} ({matches/len(test_cases)*100:.1f}%)")
    print(f"ONNX avg time: {total_onnx_time/len(test_cases):.2f}ms")
    print(f"Tinygrad avg time: {total_tg_time/len(test_cases):.2f}ms")
    print(f"Speed ratio: {total_tg_time/total_onnx_time:.1f}x slower")

def debug_single_prediction():
    """Debug a single prediction in detail"""
    text = "Bonjour le monde!"
    print(f"\n🔍 Debugging prediction for: '{text}'")
    print("=" * 50)
    
    # ONNX prediction with details
    onnx_detector = InnitDetector()
    onnx_result = onnx_detector.predict(text)
    
    print("ONNX Result:")
    print(f"  Language: {onnx_result['language']}")
    print(f"  Is English: {onnx_result['is_english']}")
    print(f"  Confidence: {onnx_result['confidence']:.6f}")
    print(f"  English prob: {onnx_result['probabilities']['english']:.6f}")
    print(f"  Non-English prob: {onnx_result['probabilities']['non_english']:.6f}")
    
    # Tinygrad prediction with details
    tg_detector = InnitTinygrad()
    tg_result = tg_detector.predict(text)
    
    print("\nTinygrad Result:")
    print(f"  Language: {tg_result['language']}")
    print(f"  Is English: {tg_result['is_english']}")
    print(f"  Confidence: {tg_result['confidence']:.6f}")
    print(f"  English prob: {tg_result['probabilities']['english']:.6f}")
    print(f"  Non-English prob: {tg_result['probabilities']['non_english']:.6f}")
    
    # Check input preprocessing
    print(f"\nInput preprocessing check:")
    bytes_data = text.encode("utf-8", errors="ignore")[:256]
    padded = np.zeros(256, dtype=np.int64)
    padded[:len(bytes_data)] = list(bytes_data)
    print(f"  Text bytes: {bytes_data}")
    print(f"  Padded shape: {padded.shape}")
    print(f"  First 10 values: {padded[:10]}")

if __name__ == "__main__":
    compare_implementations()
    debug_single_prediction()