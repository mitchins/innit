#!/usr/bin/env python3
"""
Test individual conv layer behavior
"""

import numpy as np
import torch
import torch.nn.functional as F
from tinygrad import Tensor, nn
import safetensors.torch

def test_conv_layer():
    """Test if conv layer behaves the same"""
    print("🔍 Testing Conv1D layer")
    print("=" * 30)
    
    # Load weights
    pt_weights = safetensors.torch.load_file('innit-model/model.safetensors')
    
    # Test input
    test_input = np.random.randn(1, 80, 256).astype(np.float32)
    
    # PyTorch conv
    torch_conv = torch.nn.Conv1d(80, 80, 3, padding=1)
    torch_conv.weight.data = pt_weights["blocks.0.0.weight"]
    torch_conv.bias.data = pt_weights["blocks.0.0.bias"]
    
    with torch.no_grad():
        torch_input = torch.from_numpy(test_input)
        torch_output = torch_conv(torch_input)
    
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"PyTorch output sample: {torch_output.numpy()[0, :5, 0]}")
    
    # Tinygrad conv
    tinygrad_conv = nn.Conv1d(80, 80, 3, padding=1)
    tinygrad_conv.weight = Tensor(pt_weights["blocks.0.0.weight"].numpy())
    tinygrad_conv.bias = Tensor(pt_weights["blocks.0.0.bias"].numpy())
    
    tinygrad_input = Tensor(test_input)
    tinygrad_output = tinygrad_conv(tinygrad_input)
    
    print(f"Tinygrad output shape: {tinygrad_output.shape}")
    print(f"Tinygrad output sample: {tinygrad_output.numpy()[0, :5, 0]}")
    
    # Compare
    diff = np.abs(torch_output.numpy() - tinygrad_output.numpy()).mean()
    print(f"\nMean absolute difference: {diff:.8f}")
    
    if diff < 1e-5:
        print("✅ Conv layers match!")
    else:
        print("❌ Conv layers differ significantly")

def test_embedding():
    """Test embedding layer"""
    print("\n🔍 Testing Embedding layer")
    print("=" * 30)
    
    # Load weights
    pt_weights = safetensors.torch.load_file('innit-model/model.safetensors')
    
    # Test input
    test_indices = np.array([[66, 111, 110, 106, 111]], dtype=np.int64)  # "Bonjo"
    
    # PyTorch embedding
    torch_emb = torch.nn.Embedding(257, 80)
    torch_emb.weight.data = pt_weights["emb.weight"]
    
    with torch.no_grad():
        torch_input = torch.from_numpy(test_indices)
        torch_output = torch_emb(torch_input)
    
    print(f"PyTorch embedding output shape: {torch_output.shape}")
    print(f"PyTorch embedding sample: {torch_output.numpy()[0, 0, :5]}")
    
    # Tinygrad embedding
    tinygrad_emb = nn.Embedding(257, 80)
    tinygrad_emb.weight = Tensor(pt_weights["emb.weight"].numpy())
    
    tinygrad_input = Tensor(test_indices)
    tinygrad_output = tinygrad_emb(tinygrad_input)
    
    print(f"Tinygrad embedding output shape: {tinygrad_output.shape}")
    print(f"Tinygrad embedding sample: {tinygrad_output.numpy()[0, 0, :5]}")
    
    # Compare
    diff = np.abs(torch_output.numpy() - tinygrad_output.numpy()).mean()
    print(f"\nMean absolute difference: {diff:.8f}")
    
    if diff < 1e-5:
        print("✅ Embeddings match!")
    else:
        print("❌ Embeddings differ significantly")

if __name__ == "__main__":
    test_embedding()
    test_conv_layer()