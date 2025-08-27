#!/usr/bin/env python3
"""
Test BatchNorm layer behavior
"""

import numpy as np
import torch
import torch.nn.functional as F
from tinygrad import Tensor, nn
import safetensors.torch

def test_batchnorm():
    """Test if batchnorm behaves the same"""
    print("🔍 Testing BatchNorm layer")
    print("=" * 30)
    
    # Load weights
    pt_weights = safetensors.torch.load_file('innit-model/model.safetensors')
    
    # Test input (after conv + relu)
    test_input = np.random.randn(1, 80, 256).astype(np.float32)
    test_input = np.maximum(test_input, 0)  # ReLU
    
    # PyTorch batchnorm
    torch_bn = torch.nn.BatchNorm1d(80)
    torch_bn.weight.data = pt_weights["blocks.0.2.weight"]
    torch_bn.bias.data = pt_weights["blocks.0.2.bias"]
    torch_bn.running_mean.data = pt_weights["blocks.0.2.running_mean"]
    torch_bn.running_var.data = pt_weights["blocks.0.2.running_var"]
    torch_bn.eval()  # Important: set to eval mode!
    
    with torch.no_grad():
        torch_input = torch.from_numpy(test_input)
        torch_output = torch_bn(torch_input)
    
    print(f"PyTorch BN output shape: {torch_output.shape}")
    print(f"PyTorch BN output sample: {torch_output.numpy()[0, :5, 0]}")
    
    # Tinygrad batchnorm
    tinygrad_bn = nn.BatchNorm(80)
    tinygrad_bn.weight = Tensor(pt_weights["blocks.0.2.weight"].numpy())
    tinygrad_bn.bias = Tensor(pt_weights["blocks.0.2.bias"].numpy())
    tinygrad_bn.running_mean = Tensor(pt_weights["blocks.0.2.running_mean"].numpy())
    tinygrad_bn.running_var = Tensor(pt_weights["blocks.0.2.running_var"].numpy())
    
    tinygrad_input = Tensor(test_input)
    tinygrad_output = tinygrad_bn(tinygrad_input)
    
    print(f"Tinygrad BN output shape: {tinygrad_output.shape}")
    print(f"Tinygrad BN output sample: {tinygrad_output.numpy()[0, :5, 0]}")
    
    # Compare
    diff = np.abs(torch_output.numpy() - tinygrad_output.numpy()).mean()
    print(f"\nMean absolute difference: {diff:.8f}")
    
    if diff < 1e-4:
        print("✅ BatchNorm layers match!")
    else:
        print("❌ BatchNorm layers differ significantly")
        print(f"Max difference: {np.abs(torch_output.numpy() - tinygrad_output.numpy()).max():.8f}")

def test_full_block():
    """Test full conv block: Conv1D -> ReLU -> BatchNorm"""
    print("\n🔍 Testing full ConvBlock")
    print("=" * 30)
    
    # Load weights  
    pt_weights = safetensors.torch.load_file('innit-model/model.safetensors')
    
    # Test input
    test_input = np.random.randn(1, 80, 256).astype(np.float32)
    
    # PyTorch block
    torch_conv = torch.nn.Conv1d(80, 80, 3, padding=1)
    torch_conv.weight.data = pt_weights["blocks.0.0.weight"]
    torch_conv.bias.data = pt_weights["blocks.0.0.bias"]
    
    torch_bn = torch.nn.BatchNorm1d(80)
    torch_bn.weight.data = pt_weights["blocks.0.2.weight"]
    torch_bn.bias.data = pt_weights["blocks.0.2.bias"]
    torch_bn.running_mean.data = pt_weights["blocks.0.2.running_mean"]
    torch_bn.running_var.data = pt_weights["blocks.0.2.running_var"]
    torch_bn.eval()
    
    with torch.no_grad():
        torch_input = torch.from_numpy(test_input)
        torch_x = torch_conv(torch_input)
        torch_x = F.relu(torch_x)
        torch_output = torch_bn(torch_x)
    
    print(f"PyTorch block output sample: {torch_output.numpy()[0, :5, 0]}")
    
    # Tinygrad block
    from innit_tinygrad import ConvBlock
    tinygrad_block = ConvBlock(80, 80, 3)
    
    # Load weights manually
    tinygrad_block.conv.weight = Tensor(pt_weights["blocks.0.0.weight"].numpy())
    tinygrad_block.conv.bias = Tensor(pt_weights["blocks.0.0.bias"].numpy())
    tinygrad_block.bn.weight = Tensor(pt_weights["blocks.0.2.weight"].numpy())
    tinygrad_block.bn.bias = Tensor(pt_weights["blocks.0.2.bias"].numpy())
    tinygrad_block.bn.running_mean = Tensor(pt_weights["blocks.0.2.running_mean"].numpy())
    tinygrad_block.bn.running_var = Tensor(pt_weights["blocks.0.2.running_var"].numpy())
    
    tinygrad_input = Tensor(test_input)
    tinygrad_output = tinygrad_block(tinygrad_input)
    
    print(f"Tinygrad block output sample: {tinygrad_output.numpy()[0, :5, 0]}")
    
    # Compare
    diff = np.abs(torch_output.numpy() - tinygrad_output.numpy()).mean()
    print(f"\nMean absolute difference: {diff:.8f}")
    
    if diff < 1e-4:
        print("✅ Full blocks match!")
    else:
        print("❌ Full blocks differ significantly")

if __name__ == "__main__":
    test_batchnorm()
    test_full_block()