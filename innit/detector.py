#!/usr/bin/env python3
"""
innit - Fast English vs Non-English Text Detection

A lightweight utility for binary language detection with multiple backend support.
Can be used as a CLI tool or imported as a Python library.

Usage:
    # CLI
    python innit_detector.py "Hello world!"
    python innit_detector.py --download  # Download model first
    python innit_detector.py --backend tinygrad "Hello world!"

    # Library
from innit.detector import InnitDetector
    detector = InnitDetector()  # Default: ONNX backend
    detector = InnitDetector(backend="tinygrad")  # Lightweight backend
    result = detector.predict("Hello world!")
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np


class TinyByteCNN_TG:
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
        """Load weights dict (NumPy arrays or torch tensors)."""
        from tinygrad import Tensor

        def as_np(x):
            return x.numpy() if hasattr(x, "numpy") else x

        # Embedding
        self.emb_weight = Tensor(as_np(pt_weights["emb.weight"]))

        # Conv blocks (6 blocks)
        for i in range(6):
            self.conv_weights.append(Tensor(as_np(pt_weights[f"blocks.{i}.0.weight"])))
            self.conv_biases.append(Tensor(as_np(pt_weights[f"blocks.{i}.0.bias"])))
            self.bn_weights.append(Tensor(as_np(pt_weights[f"blocks.{i}.2.weight"])))
            self.bn_biases.append(Tensor(as_np(pt_weights[f"blocks.{i}.2.bias"])))
            self.bn_means.append(Tensor(as_np(pt_weights[f"blocks.{i}.2.running_mean"])))
            self.bn_vars.append(Tensor(as_np(pt_weights[f"blocks.{i}.2.running_var"])))

        # Fully connected layers
        self.fc1_weight = Tensor(as_np(pt_weights["fc.0.weight"]))
        self.fc1_bias = Tensor(as_np(pt_weights["fc.0.bias"]))
        self.fc2_weight = Tensor(as_np(pt_weights["fc.3.weight"]))
        self.fc2_bias = Tensor(as_np(pt_weights["fc.3.bias"]))

    def conv1d(self, x, weight, bias, padding=1):
        """Conv1D using conv2d"""
        batch_size, in_channels, seq_len = x.shape
        out_channels, _, kernel_size = weight.shape

        # Reshape for conv2d
        x_2d = x.reshape(batch_size, in_channels, 1, seq_len)
        weight_2d = weight.reshape(out_channels, in_channels, 1, kernel_size)

        # Apply conv2d with padding
        if padding > 0:
            x_2d = x_2d.pad(((0, 0), (0, 0), (0, 0), (padding, padding)))

        result = x_2d.conv2d(weight_2d).reshape(batch_size, out_channels, -1)
        return result + bias.reshape(1, -1, 1)

    def batchnorm1d(self, x, weight, bias, running_mean, running_var, eps=1e-5):
        """Batch normalization in eval mode"""
        mean = running_mean.reshape(1, -1, 1)
        var = running_var.reshape(1, -1, 1)
        weight = weight.reshape(1, -1, 1)
        bias = bias.reshape(1, -1, 1)

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
        from tinygrad import Tensor

        mean_pool = x.mean(axis=2)
        max_pool = x.max(axis=2)
        std_pool = (x.var(axis=2) + 1e-5).sqrt()
        return Tensor.cat(mean_pool, max_pool, std_pool, dim=1)

    def __call__(self, x):
        """Forward pass"""
        # Embedding
        x = self.emb_weight[x]

        # Transpose for conv1d
        x = x.transpose(-1, -2)

        # Apply 6 conv blocks
        for i in range(6):
            x = self.conv_block(x, i)

        # Global pooling
        pooled = self.global_pool(x)

        # Classifier
        x = pooled.dot(self.fc1_weight.T) + self.fc1_bias
        x = x.gelu()
        x = x.dot(self.fc2_weight.T) + self.fc2_bias

        return x


# Model configuration
MODEL_URL = "https://huggingface.co/Mitchins/innit-language-detection/resolve/main/model.onnx"
MODEL_PATH = Path.home() / ".innit" / "model.onnx"
MODEL_SIZE_MB = 0.6  # Approximate size for progress

# Tinygrad asset locations (SafeTensors + config)
TINY_MODEL_URL = (
    "https://huggingface.co/Mitchins/innit-language-detection/resolve/main/model.safetensors"
)
TINY_CONFIG_URL = (
    "https://huggingface.co/Mitchins/innit-language-detection/resolve/main/config.json"
)
TINY_MODEL_PATH = Path.home() / ".innit" / "model.safetensors"
TINY_CONFIG_PATH = Path.home() / ".innit" / "config.json"


class InnitDetector:
    """Fast English vs Non-English text detection with multiple backend support."""

    def __init__(self, model_path: str | Path | None = None, backend: str = "onnx"):
        """
        Initialize the detector.

        Args:
            model_path: Path to model file. If None, uses default cached location.
            backend: Backend to use ("onnx", "tinygrad"). Default: "onnx"
        """
        self.backend = backend.lower()
        self.model_path = Path(model_path) if model_path else MODEL_PATH

        if self.backend == "onnx":
            self.session = None
            self._load_model()
        elif self.backend == "tinygrad":
            self._load_tinygrad_model()
        else:
            raise ValueError(f"Unsupported backend: {backend}. Choose from: onnx, tinygrad")

    def _load_model(self):
        """Load ONNX model session."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Run with --download flag to download the model."
            )

        try:
            import onnxruntime as ort
        except ImportError as err:
            raise ImportError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            ) from err

        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )

    def _load_tinygrad_model(self):
        """Load tinygrad model."""
        try:
            from tinygrad import Tensor

            from .assets import load_safetensors_numpy
        except ImportError as err:
            raise ImportError(
                "tinygrad and safetensors are required for tinygrad backend. "
                "Install with: pip install tinygrad safetensors"
            ) from err

        # Prefer cached paths in ~/.innit, fall back to repo, else attempt download
        safetensors_path = (
            TINY_MODEL_PATH if TINY_MODEL_PATH.exists() else Path("innit-model/model.safetensors")
        )
        config_path = (
            TINY_CONFIG_PATH if TINY_CONFIG_PATH.exists() else Path("innit-model/config.json")
        )

        # Attempt auto-download into cache if missing
        if not safetensors_path.exists() or not config_path.exists():
            try:
                download_model(force=False, backend="tinygrad")
                safetensors_path = TINY_MODEL_PATH
                config_path = TINY_CONFIG_PATH
            except Exception:
                pass
        # Validate presence
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"SafeTensors model not found. Expected at {TINY_MODEL_PATH}."
                " Run with --download or install assets."
            )
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found. Expected at {TINY_CONFIG_PATH}."
                " Run with --download or install assets."
            )

        # Load config and create tinygrad model
        with open(config_path) as f:
            import json

            self.config = json.load(f)

        self.tg_model = TinyByteCNN_TG(self.config)

        # Load weights from SafeTensors (NumPy)
        pt_weights = load_safetensors_numpy(safetensors_path)
        self.tg_model.load_weights(pt_weights)
        # Warm-up to JIT compile kernels
        try:
            dummy = Tensor(np.zeros((1, self.config.get("max_length", 256)), dtype=np.int32))
            _ = self.tg_model(dummy).numpy()
        except Exception:
            pass

    def _predict_chunk(self, text: str) -> dict[str, float]:
        """
        Predict single chunk (internal method).

        Args:
            text: Input text chunk (<= 256 bytes)

        Returns:
            Dict with raw probabilities
        """
        # Prepare input (same as training)
        bytes_data = text.encode("utf-8", errors="ignore")[:256]
        # ONNX expects int64; TinyGrad can use int32
        dtype = np.int64 if self.backend == "onnx" else np.int32
        padded = np.zeros(256, dtype=dtype)
        padded[: len(bytes_data)] = list(bytes_data)

        if self.backend == "onnx":
            if not self.session:
                raise RuntimeError("ONNX model not loaded")

            # Run ONNX inference
            outputs = self.session.run(["logits"], {"input_bytes": padded.reshape(1, -1)})
            logits = outputs[0][0]

        elif self.backend == "tinygrad":
            if not self.tg_model:
                raise RuntimeError("Tinygrad model not loaded")

            # Run tinygrad inference
            from tinygrad import Tensor

            input_tensor = Tensor(padded.reshape(1, -1))
            logits_tensor = self.tg_model(input_tensor)
            logits = logits_tensor.numpy()[0]

        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        return {"english": float(probs[1]), "non_english": float(probs[0])}

    def predict(
        self, text: str, chunk_strategy: str = "auto", ends_pct: float = 0.1
    ) -> dict[str, str | float]:
        """
        Predict if text is English or not with automatic chunking for long texts.

        Args:
            text: Input text to classify
            chunk_strategy: 'truncate', 'chunk', or 'auto' (default)
                          'auto' uses chunking for texts > 256 bytes

        Returns:
            Dict with 'language', 'confidence', and 'probabilities'
        """
        text_bytes = text.encode("utf-8", errors="ignore")

        # Single chunk if short enough or truncation requested
        if len(text_bytes) <= 256 or chunk_strategy == "truncate":
            probs = self._predict_chunk(text)
            is_english = probs["english"] > 0.5
            confidence = max(probs["english"], probs["non_english"])

            return {
                "language": "en" if is_english else "other",
                "is_english": is_english,
                "confidence": confidence,
                "probabilities": probs,
                "chunks_processed": 1,
                "method": "single_chunk",
            }

        # Multi-chunk processing for long texts
        if chunk_strategy == "ends":
            chunks = self._create_ends_chunks(text, ends_pct=ends_pct)
        else:
            chunks = self._create_chunks(text)
        if len(chunks) == 1:
            # Fallback to single chunk
            probs = self._predict_chunk(chunks[0])
            is_english = probs["english"] > 0.5
            confidence = max(probs["english"], probs["non_english"])
        else:
            # Process all chunks and aggregate
            chunk_predictions = [self._predict_chunk(chunk) for chunk in chunks]

            # Weighted average (longer chunks get more weight)
            chunk_weights = [len(chunk.encode("utf-8")) for chunk in chunks]
            total_weight = sum(chunk_weights)

            avg_en_prob = (
                sum(
                    pred["english"] * weight
                    for pred, weight in zip(chunk_predictions, chunk_weights, strict=False)
                )
                / total_weight
            )

            avg_non_en_prob = 1.0 - avg_en_prob

            # Majority vote for additional confidence
            en_votes = sum(1 for pred in chunk_predictions if pred["english"] > 0.5)
            vote_ratio = en_votes / len(chunk_predictions)

            # Combine weighted average with vote confidence
            is_english = avg_en_prob > 0.5
            base_confidence = max(avg_en_prob, avg_non_en_prob)

            # Boost confidence if votes are unanimous or nearly so
            if vote_ratio >= 0.8 or vote_ratio <= 0.2:
                confidence = min(0.99, base_confidence * 1.1)
            else:
                confidence = base_confidence * 0.9  # Reduce confidence for mixed votes

            probs = {"english": avg_en_prob, "non_english": avg_non_en_prob}

        return {
            "language": "en" if is_english else "other",
            "is_english": is_english,
            "confidence": confidence,
            "probabilities": probs,
            "chunks_processed": len(chunks),
            "method": "chunked" if len(chunks) > 1 else "single_chunk",
        }

    def _create_chunks(
        self, text: str, max_chunk_bytes: int = 256, overlap_chars: int = 20
    ) -> list:
        """
        Create overlapping chunks that respect UTF-8 byte boundaries.

        Args:
            text: Input text to chunk
            max_chunk_bytes: Maximum bytes per chunk
            overlap_chars: Character overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= overlap_chars * 2:
            return [text]  # Too short to chunk meaningfully

        chunks = []
        start = 0

        while start < len(text):
            # Try to get a chunk of reasonable character length
            # (estimate ~1.5 bytes per char to stay under byte limit)
            estimated_chars = max_chunk_bytes // 2
            end = min(start + estimated_chars, len(text))

            # Adjust to stay under byte limit
            chunk = text[start:end]
            while len(chunk.encode("utf-8")) > max_chunk_bytes and len(chunk) > 10:
                end -= 10
                chunk = text[start:end]

            if len(chunk.strip()) < 10:  # Skip tiny chunks
                break

            chunks.append(chunk)

            # Move start position with overlap
            if end >= len(text):
                break
            start = max(start + len(chunk) - overlap_chars, start + 1)

        return chunks if chunks else [text[:100]]  # Fallback

    def _create_ends_chunks(
        self, text: str, ends_pct: float = 0.1, max_chunk_bytes: int = 256
    ) -> list:
        """Create two chunks: the first and last percentage of the document.

        ends_pct is fraction of the document length (in bytes approx), each end capped at max_chunk_bytes.
        """
        tb = text.encode("utf-8", errors="ignore")
        if len(tb) <= max_chunk_bytes:
            return [text]
        # Determine byte counts for ends
        end_bytes = max(1, int(len(tb) * ends_pct))
        # Build first and last chunks by bytes, then decode back safely
        first_b = tb[: min(end_bytes, max_chunk_bytes)]
        last_b = tb[-min(end_bytes, max_chunk_bytes) :]
        try:
            first = first_b.decode("utf-8", errors="ignore")
            last = last_b.decode("utf-8", errors="ignore")
        except Exception:
            first, last = text[: max_chunk_bytes // 2], text[-max_chunk_bytes // 2 :]
        # Avoid duplicating if document is tiny
        chunks = [first]
        if last and last != first:
            chunks.append(last)
        return chunks

    def predict_batch(self, texts: list) -> list:
        """
        Predict multiple texts at once.

        Args:
            texts: List of texts to classify

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]

    def predict_batch_fast(self, texts: list) -> list:
        """Vectorized batch prediction when possible (tinygrad and ONNX)."""
        max_len = 256
        dtype = np.int64 if self.backend == "onnx" else np.int32
        batch = np.zeros((len(texts), max_len), dtype=dtype)
        for i, _t in enumerate(texts):
            b = _t.encode("utf-8", errors="ignore")[:max_len]
            batch[i, : len(b)] = list(b)
        if self.backend == "onnx":
            outputs = self.session.run(["logits"], {"input_bytes": batch})
            logits = outputs[0]
        elif self.backend == "tinygrad":
            from tinygrad import Tensor

            logits = self.tg_model(Tensor(batch)).numpy()
        else:
            raise ValueError("Unsupported backend")
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        results = []
        for i, _t in enumerate(texts):
            is_en = probs[i, 1] > 0.5
            confidence = float(max(probs[i, 0], probs[i, 1]))
            results.append(
                {
                    "language": "en" if is_en else "other",
                    "is_english": bool(is_en),
                    "confidence": confidence,
                    "probabilities": {
                        "english": float(probs[i, 1]),
                        "non_english": float(probs[i, 0]),
                    },
                    "chunks_processed": 1,
                    "method": "batch",
                }
            )
        return results


def download_model(force: bool = False, backend: str = "both") -> bool:
    """
    Download model assets from HuggingFace.

    Args:
        force: Whether to overwrite existing model
        backend: 'onnx', 'tinygrad', or 'both'

    Returns:
        True if download successful
    """

    def _download(url: str, dest: Path, size_hint_mb: float | None = None) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)

        class ProgressBar:
            def __init__(self, total_size):
                self.total_size = total_size
                self.downloaded = 0

            def update(self, chunk_size):
                self.downloaded += chunk_size
                if self.total_size > 0:
                    percent = (self.downloaded / self.total_size) * 100
                    bar_length = 40
                    filled_length = int(bar_length * percent // 100)
                    bar = "█" * filled_length + "-" * (bar_length - filled_length)
                    print(
                        f"\r[{bar}] {percent:.1f}% ({self.downloaded}/{self.total_size} bytes)",
                        end="",
                    )
                else:
                    print(f"\rDownloaded: {self.downloaded} bytes", end="")

        try:
            print(f"Downloading: {url}\nTo: {dest}")
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                progress = ProgressBar(total_size)
                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        progress.update(len(chunk))
            print(f"\n✅ Downloaded to {dest}")
            return True
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            if dest.exists():
                dest.unlink()
            return False

    ok = True
    b = backend.lower()
    if b in ("onnx", "both"):
        if not MODEL_PATH.exists() or force:
            ok &= _download(MODEL_URL, MODEL_PATH, size_hint_mb=MODEL_SIZE_MB)
        else:
            print(f"Model already exists at {MODEL_PATH}")
    if b in ("tinygrad", "both"):
        if not TINY_MODEL_PATH.exists() or force:
            ok &= _download(TINY_MODEL_URL, TINY_MODEL_PATH)
        else:
            print(f"TinyGrad model already exists at {TINY_MODEL_PATH}")
        if not TINY_CONFIG_PATH.exists() or force:
            ok &= _download(TINY_CONFIG_URL, TINY_CONFIG_PATH)
        else:
            print(f"TinyGrad config already exists at {TINY_CONFIG_PATH}")
    return bool(ok)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="innit - Fast English vs Non-English Text Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello world!"                    # Detect language
  %(prog)s --download                        # Download model first
  %(prog)s "Bonjour" "Hello" "你好"           # Multiple texts
  %(prog)s --json "Hello world!"             # JSON output
  %(prog)s --backend tinygrad "Hello!"       # Use tinygrad backend
  %(prog)s --model /path/to/model.onnx "Hi"  # Custom model path
        """,
    )

    parser.add_argument("texts", nargs="*", help="Text(s) to classify")

    parser.add_argument("--download", action="store_true", help="Download model assets to ~/.innit")
    parser.add_argument(
        "--download-backend",
        choices=["onnx", "tinygrad", "both"],
        default="both",
        help="Which assets to download when using --download",
    )

    parser.add_argument("--model", type=str, help="Path to model file")

    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "onnx", "tinygrad"],
        default="auto",
        help="Backend: auto (prefer onnx), onnx (fastest CPU), or tinygrad (lightweight)",
    )

    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    parser.add_argument(
        "--chunk-strategy",
        choices=["auto", "truncate", "chunk", "ends"],
        default="auto",
        help="How to handle long texts: auto (chunk if >256 bytes), truncate, chunk, or ends (sample ends)",
    )
    parser.add_argument(
        "--ends-pct",
        type=float,
        default=0.10,
        help="For --chunk-strategy ends: percentage from each end (0-0.5)",
    )

    parser.add_argument("--version", action="version", version="innit 1.0")

    args = parser.parse_args()

    # Download model if requested
    if args.download:
        success = download_model(force=True, backend=args.download_backend)
        if not success:
            sys.exit(1)
        if not args.texts:
            print("Model downloaded. You can now run predictions.")
            return

    # Check if we have texts to process
    if not args.texts:
        # Read from stdin if available
        if not sys.stdin.isatty():
            texts = [line.strip() for line in sys.stdin if line.strip()]
        else:
            parser.print_help()
            sys.exit(1)
    else:
        texts = args.texts

    # Initialize detector
    # Resolve backend if auto
    backend = args.backend
    if backend == "auto":
        try:
            import onnxruntime  # noqa: F401

            backend = "onnx"
        except Exception:
            backend = "tinygrad"

    try:
        detector = InnitDetector(model_path=args.model, backend=backend)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Run with --download to get the model:")
        print(f"   python {sys.argv[0]} --download")
        sys.exit(1)
    except ImportError as e:
        print(f"❌ {e}")
        if args.backend == "auto":
            print(
                "💡 Install a backend: 'pip install innit-detector[onnx]' for ONNX, or TinyGrad is already bundled."
            )
        sys.exit(1)

    # Process texts
    results = []
    for text in texts:
        try:
            result = detector.predict(
                text, chunk_strategy=args.chunk_strategy, ends_pct=args.ends_pct
            )
            results.append({"text": text, **result})
        except Exception as e:
            print(f"❌ Error processing '{text}': {e}")
            continue

    # Output results
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for result in results:
            text = result["text"]
            lang = "English" if result["is_english"] else "Non-English"
            conf = result["confidence"]
            en_prob = result["probabilities"]["english"]
            chunks = result.get("chunks_processed", 1)
            method = result.get("method", "single_chunk")

            # Truncate long text for display
            display_text = text if len(text) <= 50 else text[:47] + "..."

            if len(texts) > 1:
                chunk_info = f", {chunks} chunks" if chunks > 1 else ""
                print(
                    f"'{display_text}' → {lang} (confidence: {conf:.3f}, en_prob: {en_prob:.3f}{chunk_info})"
                )
            else:
                print(f"Text: {display_text}")
                print(f"Language: {lang}")
                print(f"Confidence: {conf:.3f}")
                print(f"English probability: {en_prob:.3f}")
                if chunks > 1:
                    print(f"Processing: {method}, {chunks} chunks")


if __name__ == "__main__":
    main()
