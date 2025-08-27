#!/usr/bin/env python3
"""
Test Linear layer behavior
"""

import numpy as np
import torch
from tinygrad import Tensor, nn
import safetensors.torch

def test_linear_layers():
    """Test if linear layers behave the same"""
    print("🔍 Testing Linear layers")
    print("=" * 30)
    
    # Load weights
    pt_weights = safetensors.torch.load_file('innit-model/model.safetensors')
    
    # Test input (240 dims - 3 x 80 pooled features)
    test_input = np.random.randn(1, 240).astype(np.float32)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"FC weights shapes:")
    print(f"  fc.0.weight: {pt_weights['fc.0.weight'].shape}")
    print(f"  fc.0.bias: {pt_weights['fc.0.bias'].shape}")
    print(f"  fc.3.weight: {pt_weights['fc.3.weight'].shape}")
    print(f"  fc.3.bias: {pt_weights['fc.3.bias'].shape}")
    
    # PyTorch layers
    torch_fc1 = torch.nn.Linear(240, 80)
    torch_fc1.weight.data = pt_weights["fc.0.weight"]
    torch_fc1.bias.data = pt_weights["fc.0.bias"]
    
    torch_fc2 = torch.nn.Linear(80, 2)
    torch_fc2.weight.data = pt_weights["fc.3.weight"]
    torch_fc2.bias.data = pt_weights["fc.3.bias"]
    
    with torch.no_grad():
        torch_input = torch.from_numpy(test_input)
        torch_x1 = torch_fc1(torch_input)
        torch_x1_relu = torch.relu(torch_x1)
        torch_output = torch_fc2(torch_x1_relu)
    
    print(f"\nPyTorch FC1 output sample: {torch_x1.numpy()[0, :5]}")
    print(f"PyTorch FC1+ReLU output sample: {torch_x1_relu.numpy()[0, :5]}")
    print(f"PyTorch final output: {torch_output.numpy()[0]}")
    
    # Tinygrad layers
    tinygrad_fc1 = nn.Linear(240, 80)
    tinygrad_fc1.weight = Tensor(pt_weights["fc.0.weight"].numpy())
    tinygrad_fc1.bias = Tensor(pt_weights["fc.0.bias"].numpy())
    
    tinygrad_fc2 = nn.Linear(80, 2)
    tinygrad_fc2.weight = Tensor(pt_weights["fc.3.weight"].numpy())
    tinygrad_fc2.bias = Tensor(pt_weights["fc.3.bias"].numpy())
    
    tinygrad_input = Tensor(test_input)
    tinygrad_x1 = tinygrad_fc1(tinygrad_input)
    tinygrad_x1_relu = tinygrad_x1.relu()
    tinygrad_output = tinygrad_fc2(tinygrad_x1_relu)
    
    print(f"\nTinygrad FC1 output sample: {tinygrad_x1.numpy()[0, :5]}")
    print(f"Tinygrad FC1+ReLU output sample: {tinygrad_x1_relu.numpy()[0, :5]}")
    print(f"Tinygrad final output: {tinygrad_output.numpy()[0]}")
    
    # Compare
    diff1 = np.abs(torch_x1.numpy() - tinygrad_x1.numpy()).mean()
    diff_relu = np.abs(torch_x1_relu.numpy() - tinygrad_x1_relu.numpy()).mean()
    diff_final = np.abs(torch_output.numpy() - tinygrad_output.numpy()).mean()
    
    print(f"\nFC1 difference: {diff1:.8f}")
    print(f"ReLU difference: {diff_relu:.8f}")
    print(f"Final difference: {diff_final:.8f}")
    
    if diff_final < 1e-5:
        print("✅ Linear layers match!")
    else:
        print("❌ Linear layers differ")

if __name__ == "__main__":
    test_linear_layers()