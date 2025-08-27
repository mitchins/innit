#!/usr/bin/env python3
"""
innit - Fast English vs Non-English Text Detection (Tinygrad Implementation)

A tinygrad-based implementation of the innit language detection model.
Converts the original PyTorch/ONNX model to run on tinygrad.
"""

import json
import time
from pathlib import Path

import numpy as np
import safetensors.torch
from tinygrad import Tensor, nn

# Model configuration (prefer cache in ~/.innit, fallback to repo paths)
CACHE_MODEL_PATH = Path.home() / ".innit" / "model.safetensors"
CACHE_CONFIG_PATH = Path.home() / ".innit" / "config.json"
REPO_MODEL_PATH = Path("innit-model/model.safetensors")
REPO_CONFIG_PATH = Path("innit-model/config.json")


class ConvBlock:
    """Conv1D -> GELU -> BatchNorm -> Residual add"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv(x)
        x = x.gelu()
        x = self.bn(x)
        # Residual connection (in/out channels are equal for all blocks)
        return x + residual


class TinyByteCNN_TG:
    """TinyByteCNN implementation in tinygrad"""

    def __init__(self, config: dict):
        self.config = config

        # Embedding layer (vocab_size=257, emb_dim=80)
        self.emb = nn.Embedding(config["vocab_size"], config["emb_dim"])

        # 6 convolutional blocks (all 80->80 channels, kernel=3)
        self.blocks = []
        for i in range(config["num_blocks"]):
            block = ConvBlock(config["emb_dim"], config["emb_dim"], kernel_size=3)
            self.blocks.append(block)

        # Final classifier: Linear(240, 80) -> ReLU -> Linear(80, 2)
        # 240 = 80 channels * 3 (max, mean, last pooling)
        self.fc1 = nn.Linear(240, 80)
        self.fc2 = nn.Linear(80, config["num_classes"])

    def __call__(self, x: Tensor) -> Tensor:
        # x shape: (batch, seq_len) - integers 0-256

        # Embedding: (batch, seq_len) -> (batch, seq_len, emb_dim)
        x = self.emb(x)

        # Transpose for conv1d: (batch, seq_len, emb_dim) -> (batch, emb_dim, seq_len)
        x = x.transpose(-1, -2)

        # Apply conv blocks
        for block in self.blocks:
            x = block(x)

        # Global pooling: mean, max, std (to match ONNX model)
        # x shape: (batch, 80, seq_len)
        mean_pool = x.mean(axis=2)  # (batch, 80)
        max_pool = x.max(axis=2)  # (batch, 80)
        std_pool = (x.var(axis=2) + 1e-5).sqrt()  # (batch, 80)

        # Concatenate pooled features in order: mean, max, std -> (batch, 240)
        pooled = Tensor.cat(mean_pool, max_pool, std_pool, dim=1)

        # Apply final classifier
        x = self.fc1(pooled)
        x = x.relu()
        x = self.fc2(x)

        return x


class InnitTinygrad:
    """Tinygrad-based innit detector"""

    def __init__(self, model_path: str | Path | None = None):
        # Resolve model/config paths
        self.model_path, self.config_path = self._resolve_paths(model_path)

        # Load config and model
        with open(self.config_path) as f:
            self.config = json.load(f)

        self.model = TinyByteCNN_TG(self.config)
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

        # Convert to tinygrad tensors
        state_dict = {}
        for name, tensor in pt_weights.items():
            if "num_batches_tracked" in name:
                continue  # Skip batch norm tracking statistics
            state_dict[name] = Tensor(tensor.numpy())

        # Map weights to tinygrad model
        self._load_embedding_weights(state_dict)
        self._load_conv_block_weights(state_dict)
        self._load_classifier_weights(state_dict)

        # Set model to eval mode (disable batch norm training)
        self._set_eval_mode()

        print("✅ Weights loaded successfully!")

    def _resolve_paths(self, model_path: str | Path | None) -> tuple[Path, Path]:
        # Choose provided path if given
        if model_path:
            mp = Path(model_path)
            cp = CACHE_CONFIG_PATH if CACHE_CONFIG_PATH.exists() else REPO_CONFIG_PATH
            return mp, cp
        # Prefer cache
        mp = CACHE_MODEL_PATH if CACHE_MODEL_PATH.exists() else REPO_MODEL_PATH
        cp = CACHE_CONFIG_PATH if CACHE_CONFIG_PATH.exists() else REPO_CONFIG_PATH
        # Attempt auto-download into cache if neither exists
        if not mp.exists() or not cp.exists():
            try:
                from .detector import download_model

                download_model(force=False, backend="tinygrad")
                mp, cp = CACHE_MODEL_PATH, CACHE_CONFIG_PATH
            except Exception:
                pass
        return mp, cp

    def _set_eval_mode(self):
        """Set batch norm layers to eval mode"""
        for block in self.model.blocks:
            if hasattr(block.bn, "training"):
                block.bn.training = False

    def _load_embedding_weights(self, state_dict):
        """Load embedding layer weights"""
        self.model.emb.weight = state_dict["emb.weight"]

    def _load_conv_block_weights(self, state_dict):
        """Load convolutional block weights"""
        for i, block in enumerate(self.model.blocks):
            # Conv1d weights and bias
            block.conv.weight = state_dict[f"blocks.{i}.0.weight"]
            block.conv.bias = state_dict[f"blocks.{i}.0.bias"]

            # BatchNorm weights, bias, running stats
            block.bn.weight = state_dict[f"blocks.{i}.2.weight"]
            block.bn.bias = state_dict[f"blocks.{i}.2.bias"]
            block.bn.running_mean = state_dict[f"blocks.{i}.2.running_mean"]
            block.bn.running_var = state_dict[f"blocks.{i}.2.running_var"]

    def _load_classifier_weights(self, state_dict):
        """Load final classifier weights"""
        # First linear layer
        self.model.fc1.weight = state_dict["fc.0.weight"]
        self.model.fc1.bias = state_dict["fc.0.bias"]

        # Output layer
        self.model.fc2.weight = state_dict["fc.3.weight"]
        self.model.fc2.bias = state_dict["fc.3.bias"]

    def predict(self, text: str) -> dict[str, float | str | bool]:
        """Predict if text is English or not"""
        start_time = time.time()

        # Prepare input (same as original)
        bytes_data = text.encode("utf-8", errors="ignore")[:256]
        padded = np.zeros(256, dtype=np.int32)
        padded[: len(bytes_data)] = list(bytes_data)

        # Convert to tinygrad tensor
        input_tensor = Tensor(padded.reshape(1, -1))

        # Forward pass (tinygrad doesn't need no_grad context)
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
            "framework": "tinygrad",
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
        # Softmax per row
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
                    "framework": "tinygrad",
                }
            )
        return results


def benchmark_tinygrad():
    """Simple benchmark comparing tinygrad vs ONNX"""
    print("🔥 Tinygrad innit benchmark")
    print("=" * 40)

    # Initialize tinygrad detector
    tg_detector = InnitTinygrad()

    # Test cases
    test_cases = [
        "Hello world!",
        "Bonjour le monde!",
        "你好世界！",
        "This is a longer English text that should be detected correctly by the model.",
        "Este es un texto en español que debería ser detectado como no inglés.",
    ]

    print("\nTinygrad Results:")
    print("-" * 40)

    total_time = 0
    for i, text in enumerate(test_cases):
        result = tg_detector.predict(text)
        total_time += result["inference_time_ms"]

        lang = "EN" if result["is_english"] else "NON-EN"
        print(
            f"{i+1}. {text[:30]:<30} → {lang:<6} ({result['confidence']:.3f}) [{result['inference_time_ms']:.2f}ms]"
        )

    avg_time = total_time / len(test_cases)
    print(f"\nAverage inference time: {avg_time:.2f}ms")
    print(f"Total time: {total_time:.2f}ms")


if __name__ == "__main__":
    benchmark_tinygrad()
