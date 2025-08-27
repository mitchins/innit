#!/usr/bin/env python3
"""
Debug step by step to find where the implementation diverges
"""

import numpy as np
import torch
import onnxruntime as ort
import safetensors.torch
from tinygrad import Tensor
from innit_tinygrad_fixed import TinyByteCNN_Pure

def debug_step_by_step():
    """Debug each step of the forward pass"""
    print("🔍 Step-by-step debugging")
    print("=" * 35)
    
    # Test input
    text = "Hello world!"
    bytes_data = text.encode("utf-8", errors="ignore")[:256]
    padded = np.zeros(256, dtype=np.int64)
    padded[:len(bytes_data)] = list(bytes_data)
    input_array = padded.reshape(1, -1)
    
    print(f"Input text: '{text}'")
    print(f"Input shape: {input_array.shape}")
    print(f"Input sample: {input_array[0, :10]}")
    
    # Load PyTorch weights
    pt_weights = safetensors.torch.load_file('innit-model/model.safetensors')
    
    # Create PyTorch model for comparison
    emb = torch.nn.Embedding(257, 80)
    emb.weight.data = pt_weights["emb.weight"]
    
    # Create tinygrad model
    with open('innit-model/config.json') as f:
        import json
        config = json.load(f)
    
    tg_model = TinyByteCNN_Pure(config)
    tg_model.load_weights(pt_weights)
    
    # Step 1: Embedding
    print(f"\n--- Step 1: Embedding ---")
    torch_input = torch.from_numpy(input_array)
    torch_emb = emb(torch_input)
    print(f"PyTorch embedding shape: {torch_emb.shape}")
    print(f"PyTorch embedding sample: {torch_emb.detach().numpy()[0, 0, :5]}")
    
    tg_input = Tensor(input_array)
    tg_emb = tg_model.embedding(tg_input)
    print(f"Tinygrad embedding shape: {tg_emb.shape}")
    print(f"Tinygrad embedding sample: {tg_emb.numpy()[0, 0, :5]}")
    
    # Check if embeddings match
    emb_diff = np.abs(torch_emb.detach().numpy() - tg_emb.numpy()).mean()
    print(f"Embedding difference: {emb_diff:.8f}")
    
    # Step 2: Transpose
    print(f"\n--- Step 2: Transpose ---")
    torch_transposed = torch_emb.transpose(-1, -2)
    tg_transposed = tg_emb.transpose(-1, -2)
    print(f"PyTorch transposed shape: {torch_transposed.shape}")
    print(f"Tinygrad transposed shape: {tg_transposed.shape}")
    
    trans_diff = np.abs(torch_transposed.detach().numpy() - tg_transposed.numpy()).mean()
    print(f"Transpose difference: {trans_diff:.8f}")
    
    # Step 3: First conv block
    print(f"\n--- Step 3: First Conv Block ---")
    
    # PyTorch conv block
    conv0 = torch.nn.Conv1d(80, 80, 3, padding=1)
    conv0.weight.data = pt_weights["blocks.0.0.weight"]
    conv0.bias.data = pt_weights["blocks.0.0.bias"]
    
    bn0 = torch.nn.BatchNorm1d(80)
    bn0.weight.data = pt_weights["blocks.0.2.weight"]
    bn0.bias.data = pt_weights["blocks.0.2.bias"]
    bn0.running_mean.data = pt_weights["blocks.0.2.running_mean"]
    bn0.running_var.data = pt_weights["blocks.0.2.running_var"]
    bn0.eval()
    
    with torch.no_grad():
        torch_conv_out = conv0(torch_transposed)
        torch_relu_out = torch.relu(torch_conv_out)
        torch_bn_out = bn0(torch_relu_out)
    
    print(f"PyTorch conv block output shape: {torch_bn_out.shape}")
    print(f"PyTorch conv block sample: {torch_bn_out.detach().numpy()[0, :5, 0]}")
    
    # Tinygrad conv block
    tg_conv_out = tg_model.conv_block(tg_transposed, 0)
    print(f"Tinygrad conv block output shape: {tg_conv_out.shape}")
    print(f"Tinygrad conv block sample: {tg_conv_out.numpy()[0, :5, 0]}")
    
    conv_diff = np.abs(torch_bn_out.detach().numpy() - tg_conv_out.numpy()).mean()
    print(f"Conv block difference: {conv_diff:.8f}")
    
    if conv_diff > 1e-4:
        print("🚨 Significant difference in conv block!")
        
        # Debug the conv1d operation specifically
        print("\n--- Debugging Conv1D ---")
        
        # Manual conv1d in PyTorch
        torch_conv_manual = torch.nn.functional.conv1d(
            torch_transposed, 
            pt_weights["blocks.0.0.weight"], 
            pt_weights["blocks.0.0.bias"],
            padding=1
        )
        
        # Tinygrad conv1d
        tg_conv_manual = tg_model.conv1d(
            tg_transposed,
            tg_model.conv_weights[0],
            tg_model.conv_biases[0],
            padding=1
        )
        
        print(f"PyTorch conv1d output shape: {torch_conv_manual.shape}")
        print(f"Tinygrad conv1d output shape: {tg_conv_manual.shape}")
        print(f"PyTorch conv1d sample: {torch_conv_manual.detach().numpy()[0, :3, 0]}")
        print(f"Tinygrad conv1d sample: {tg_conv_manual.numpy()[0, :3, 0]}")
        
        conv1d_diff = np.abs(torch_conv_manual.detach().numpy() - tg_conv_manual.numpy()).mean()
        print(f"Conv1d difference: {conv1d_diff:.8f}")

if __name__ == "__main__":
    debug_step_by_step()