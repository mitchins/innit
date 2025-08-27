#!/usr/bin/env python3
"""
innit - Fast English vs Non-English Text Detection (Fixed Tinygrad Implementation)

A pure tinygrad implementation without nn helpers that might have subtle differences.
"""

import json
import time
from pathlib import Path

import numpy as np
import safetensors.torch
from tinygrad import Tensor

# Model configuration (prefer cache in ~/.innit)
MODEL_PATH = Path.home() / ".innit" / "model.safetensors"
CONFIG_PATH = Path.home() / ".innit" / "config.json"


class TinyByteCNN_Pure:
    """Pure tinygrad implementation of TinyByteCNN"""

    def __init__(self, config: dict):
        self.config = config

        # Weights will be loaded from SafeTensors
        self.emb_weight = None
        self.conv_weights = []
        self.conv_biases = []
        self.bn_weights = []
        self.bn_biases = []
        self.bn_means = []
        self.bn_vars = []
        self.fc1_weight = None
        self.fc1_bias = None
        self.fc2_weight = None
        self.fc2_bias = None

    def load_weights(self, pt_weights):
        """Load all weights from PyTorch state dict"""
        # Embedding
        self.emb_weight = Tensor(pt_weights["emb.weight"].numpy())

        # Conv blocks (6 blocks)
        for i in range(6):
            # Conv weights and bias
            self.conv_weights.append(Tensor(pt_weights[f"blocks.{i}.0.weight"].numpy()))
            self.conv_biases.append(Tensor(pt_weights[f"blocks.{i}.0.bias"].numpy()))

            # BatchNorm weights, bias, running stats
            self.bn_weights.append(Tensor(pt_weights[f"blocks.{i}.2.weight"].numpy()))
            self.bn_biases.append(Tensor(pt_weights[f"blocks.{i}.2.bias"].numpy()))
            self.bn_means.append(Tensor(pt_weights[f"blocks.{i}.2.running_mean"].numpy()))
            self.bn_vars.append(Tensor(pt_weights[f"blocks.{i}.2.running_var"].numpy()))

        # Fully connected layers
        self.fc1_weight = Tensor(pt_weights["fc.0.weight"].numpy())
        self.fc1_bias = Tensor(pt_weights["fc.0.bias"].numpy())
        self.fc2_weight = Tensor(pt_weights["fc.3.weight"].numpy())
        self.fc2_bias = Tensor(pt_weights["fc.3.bias"].numpy())

    def embedding(self, x):
        """Manual embedding lookup: x[i] -> emb_weight[x[i]]"""
        return self.emb_weight[x]

    def conv1d(self, x, weight, bias, padding=1):
        """Manual 1D convolution using conv2d"""
        # x: (batch, in_channels, seq_len) -> (batch, in_channels, 1, seq_len)
        # weight: (out_channels, in_channels, kernel_size) -> (out_channels, in_channels, 1, kernel_size)

        batch_size, in_channels, seq_len = x.shape
        out_channels, _, kernel_size = weight.shape

        # Reshape for conv2d
        x_2d = x.reshape(batch_size, in_channels, 1, seq_len)
        weight_2d = weight.reshape(out_channels, in_channels, 1, kernel_size)

        # Apply conv2d with padding
        if padding > 0:
            x_2d = x_2d.pad(((0, 0), (0, 0), (0, 0), (padding, padding)))

        # Conv2d
        result = x_2d.conv2d(weight_2d)

        # Reshape back to 1D: (batch, out_channels, 1, new_seq_len) -> (batch, out_channels, new_seq_len)
        result = result.reshape(batch_size, out_channels, -1)

        # Add bias
        return result + bias.reshape(1, -1, 1)

    def batchnorm1d(self, x, weight, bias, running_mean, running_var, eps=1e-5):
        """Manual batch normalization in eval mode"""
        # x: (batch, channels, seq_len)
        # Reshape running stats for broadcasting: (1, channels, 1)
        mean = running_mean.reshape(1, -1, 1)
        var = running_var.reshape(1, -1, 1)
        weight = weight.reshape(1, -1, 1)
        bias = bias.reshape(1, -1, 1)

        # Normalize: (x - mean) / sqrt(var + eps) * weight + bias
        normalized = (x - mean) / (var + eps).sqrt()
        return normalized * weight + bias

    def conv_block(self, x, i):
        """Conv1D -> GELU -> BatchNorm -> Residual add"""
        residual = x
        x = self.conv1d(x, self.conv_weights[i], self.conv_biases[i], padding=1)
        x = x.gelu()
        x = self.batchnorm1d(
            x, self.bn_weights[i], self.bn_biases[i], self.bn_means[i], self.bn_vars[i]
        )
        return x + residual

    def global_pool(self, x):
        """Global pooling: mean, max, std (matches ONNX)"""
        # x: (batch, channels, seq_len)
        mean_pool = x.mean(axis=2)  # (batch, channels)
        max_pool = x.max(axis=2)  # (batch, channels)
        std_pool = (x.var(axis=2) + 1e-5).sqrt()  # (batch, channels)
        return Tensor.cat(mean_pool, max_pool, std_pool, dim=1)

    def linear(self, x, weight, bias):
        """Manual linear layer: x @ weight.T + bias"""
        return x.dot(weight.T) + bias

    def __call__(self, x):
        """Forward pass"""
        # x: (batch, seq_len) - integers 0-256

        # Embedding: (batch, seq_len) -> (batch, seq_len, emb_dim)
        x = self.embedding(x)

        # Transpose for conv1d: (batch, seq_len, emb_dim) -> (batch, emb_dim, seq_len)
        x = x.transpose(-1, -2)

        # Apply 6 conv blocks
        for i in range(6):
            x = self.conv_block(x, i)

        # Global pooling
        pooled = self.global_pool(x)  # (batch, 240)

        # First linear layer + GELU
        x = self.linear(pooled, self.fc1_weight, self.fc1_bias)
        x = x.gelu()

        # Output layer
        x = self.linear(x, self.fc2_weight, self.fc2_bias)

        return x


class InnitTinygradFixed:
    """Fixed tinygrad-based innit detector"""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.config_path = CONFIG_PATH

        # Load config and model
        with open(self.config_path) as f:
            self.config = json.load(f)

        self.model = TinyByteCNN_Pure(self.config)
        self._load_weights()
        # Warm-up: JIT compile kernels with a dummy forward pass
        try:
            dummy = Tensor(np.zeros((1, self.config.get("max_length", 256)), dtype=np.int32))
            _ = self.model(dummy).numpy()
        except Exception:
            pass

    def _load_weights(self):
        """Load weights from SafeTensors file"""
        print("Loading weights from SafeTensors...")

        # Load PyTorch weights
        pt_weights = safetensors.torch.load_file(self.model_path)
        self.model.load_weights(pt_weights)

        print("✅ Weights loaded successfully!")

    def predict(self, text: str) -> dict[str, float | str | bool]:
        """Predict if text is English or not"""
        start_time = time.time()

        # Prepare input (same as original)
        bytes_data = text.encode("utf-8", errors="ignore")[:256]
        padded = np.zeros(256, dtype=np.int32)
        padded[: len(bytes_data)] = list(bytes_data)

        # Convert to tinygrad tensor
        input_tensor = Tensor(padded.reshape(1, -1))

        # Forward pass
        logits = self.model(input_tensor)

        # Apply softmax
        logits_np = logits.numpy()[0]
        exp_logits = np.exp(logits_np - np.max(logits_np))
        probs = exp_logits / np.sum(exp_logits)

        inference_time = time.time() - start_time

        # Format results
        is_english = probs[1] > 0.5
        confidence = max(probs[0], probs[1])

        return {
            "language": "en" if is_english else "other",
            "is_english": is_english,
            "confidence": float(confidence),
            "probabilities": {"english": float(probs[1]), "non_english": float(probs[0])},
            "inference_time_ms": inference_time * 1000,
            "framework": "tinygrad_fixed",
            "raw_logits": logits_np.tolist(),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict multiple texts at once"""
        return self.predict_batch_fast(texts)

    def predict_batch_fast(self, texts: list[str]) -> list[dict]:
        """Vectorized batch prediction with a single forward pass"""
        start_time = time.time()
        max_len = 256
        batch = np.zeros((len(texts), max_len), dtype=np.int32)
        for i, t in enumerate(texts):
            b = t.encode("utf-8", errors="ignore")[:max_len]
            batch[i, : len(b)] = list(b)
        logits = self.model(Tensor(batch)).numpy()
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        results = []
        for i, t in enumerate(texts):
            is_en = probs[i, 1] > 0.5
            conf = float(max(probs[i, 0], probs[i, 1]))
            results.append(
                {
                    "language": "en" if is_en else "other",
                    "is_english": bool(is_en),
                    "confidence": conf,
                    "probabilities": {
                        "english": float(probs[i, 1]),
                        "non_english": float(probs[i, 0]),
                    },
                    "inference_time_ms": (time.time() - start_time) * 1000,
                    "framework": "tinygrad_fixed",
                    "raw_logits": logits[i].tolist(),
                }
            )
        return results


def test_fixed_implementation():
    """Test the fixed implementation"""
    print("🔥 Testing Fixed Tinygrad Implementation")
    print("=" * 45)

    # Initialize detector
    detector = InnitTinygradFixed()

    # Test cases
    test_cases = [
        ("Hello world!", True),
        ("Bonjour le monde!", False),
        ("你好世界！", False),
        ("This is English text.", True),
        ("Este es español.", False),
    ]

    print("\nResults:")
    print("-" * 45)
    print(f"{'Text':<25} {'Expected':<10} {'Predicted':<10} {'Confidence':<12} {'Logits'}")
    print("-" * 45)

    for text, expected_english in test_cases:
        result = detector.predict(text)

        display_text = text if len(text) <= 20 else text[:17] + "..."
        expected_str = "EN" if expected_english else "NON-EN"
        predicted_str = "EN" if result["is_english"] else "NON-EN"
        confidence = result["confidence"]
        logits = result["raw_logits"]

        print(
            f"{display_text:<25} {expected_str:<10} {predicted_str:<10} {confidence:<12.3f} {logits}"
        )


if __name__ == "__main__":
    test_fixed_implementation()
