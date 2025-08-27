#!/usr/bin/env python3
"""
Debug the tinygrad implementation by comparing intermediate outputs
"""

import json
import numpy as np
from tinygrad import Tensor
import safetensors.torch

# Load the model weights and config
with open('innit-model/config.json') as f:
    config = json.load(f)

pt_weights = safetensors.torch.load_file('innit-model/model.safetensors')

def debug_forward_pass():
    """Debug the forward pass step by step"""
    print("🔍 Debugging tinygrad forward pass")
    print("=" * 40)
    
    # Test input
    text = "Bonjour"
    bytes_data = text.encode("utf-8", errors="ignore")[:256]
    padded = np.zeros(256, dtype=np.int64)
    padded[:len(bytes_data)] = list(bytes_data)
    input_tensor = Tensor(padded.reshape(1, -1))
    
    print(f"Input text: '{text}'")
    print(f"Input bytes: {bytes_data}")
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input range: {padded.min()} - {padded.max()}")
    
    # Step 1: Embedding
    emb_weight = Tensor(pt_weights["emb.weight"].numpy())
    print(f"\nEmbedding weight shape: {emb_weight.shape}")
    
    # Manual embedding lookup
    embedded = emb_weight[input_tensor]
    print(f"Embedded shape: {embedded.shape}")
    print(f"Embedded sample: {embedded.numpy()[0, 0, :5]}")
    
    # Step 2: Transpose for conv1d
    embedded_t = embedded.transpose(-1, -2)
    print(f"After transpose: {embedded_t.shape}")
    
    # Step 3: Check first conv block
    conv0_weight = Tensor(pt_weights["blocks.0.0.weight"].numpy())
    conv0_bias = Tensor(pt_weights["blocks.0.0.bias"].numpy())
    bn0_weight = Tensor(pt_weights["blocks.0.2.weight"].numpy())
    bn0_bias = Tensor(pt_weights["blocks.0.2.bias"].numpy())
    bn0_mean = Tensor(pt_weights["blocks.0.2.running_mean"].numpy())
    bn0_var = Tensor(pt_weights["blocks.0.2.running_var"].numpy())
    
    print(f"\nConv0 weight shape: {conv0_weight.shape}")
    print(f"Conv0 bias shape: {conv0_bias.shape}")
    
    # Let's check if there's an issue with the tensor concatenation
    print(f"\n--- Testing pooling operations ---")
    dummy_conv_output = embedded_t  # Just use embedded as dummy conv output
    print(f"Dummy conv output shape: {dummy_conv_output.shape}")
    
    # Test pooling operations
    try:
        max_pool = dummy_conv_output.max(axis=2)
        print(f"Max pool shape: {max_pool.shape}")
    except Exception as e:
        print(f"Max pool error: {e}")
    
    try:
        mean_pool = dummy_conv_output.mean(axis=2)
        print(f"Mean pool shape: {mean_pool.shape}")
    except Exception as e:
        print(f"Mean pool error: {e}")
    
    try:
        last_pool = dummy_conv_output[:, :, -1]
        print(f"Last pool shape: {last_pool.shape}")
    except Exception as e:
        print(f"Last pool error: {e}")
    
    # Test concatenation
    try:
        pooled = Tensor.cat(max_pool, mean_pool, last_pool, dim=1)
        print(f"Concatenated shape: {pooled.shape}")
    except Exception as e:
        print(f"Concatenation error: {e}")
        # Try alternative concatenation
        try:
            pooled = max_pool.cat(mean_pool, dim=1).cat(last_pool, dim=1)
            print(f"Alternative concat shape: {pooled.shape}")
        except Exception as e2:
            print(f"Alternative concat error: {e2}")

if __name__ == "__main__":
    debug_forward_pass()