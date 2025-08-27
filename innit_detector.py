#!/usr/bin/env python3
"""
innit - Fast English vs Non-English Text Detection

A lightweight utility for binary language detection using ONNX runtime.
Can be used as a CLI tool or imported as a Python library.

Usage:
    # CLI
    python innit_detector.py "Hello world!"
    python innit_detector.py --download  # Download model first

    # Library
    from innit_detector import InnitDetector
    detector = InnitDetector()
    result = detector.predict("Hello world!")
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

# Model configuration
MODEL_URL = "https://huggingface.co/Mitchins/innit-language-detection/resolve/main/model.onnx"
MODEL_PATH = Path.home() / ".innit" / "model.onnx"
MODEL_SIZE_MB = 0.6  # Approximate size for progress


class InnitDetector:
    """Fast English vs Non-English text detection using ONNX runtime."""

    def __init__(self, model_path: str | Path | None = None):
        """
        Initialize the detector.

        Args:
            model_path: Path to ONNX model. If None, uses default cached location.
        """
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.session = None
        self._load_model()

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
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime") from err

        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )

    def _predict_chunk(self, text: str) -> dict[str, float]:
        """
        Predict single chunk (internal method).

        Args:
            text: Input text chunk (<= 256 bytes)

        Returns:
            Dict with raw probabilities
        """
        if not self.session:
            raise RuntimeError("Model not loaded")

        # Prepare input (same as training)
        bytes_data = text.encode("utf-8", errors="ignore")[:256]
        padded = np.zeros(256, dtype=np.int64)
        padded[: len(bytes_data)] = list(bytes_data)

        # Run inference
        outputs = self.session.run(["logits"], {"input_bytes": padded.reshape(1, -1)})
        logits = outputs[0][0]

        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        return {"english": float(probs[1]), "non_english": float(probs[0])}

    def predict(self, text: str, chunk_strategy: str = "auto") -> dict[str, str | float]:
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

    def predict_batch(self, texts: list) -> list:
        """
        Predict multiple texts at once.

        Args:
            texts: List of texts to classify

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]


def download_model(force: bool = False) -> bool:
    """
    Download the ONNX model from HuggingFace.

    Args:
        force: Whether to overwrite existing model

    Returns:
        True if download successful
    """
    if MODEL_PATH.exists() and not force:
        print(f"Model already exists at {MODEL_PATH}")
        return True

    print(f"Downloading innit model ({MODEL_SIZE_MB}MB)...")
    print(f"From: {MODEL_URL}")
    print(f"To: {MODEL_PATH}")

    # Create directory
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

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
                    f"\r[{bar}] {percent:.1f}% ({self.downloaded}/{self.total_size} bytes)", end=""
                )
            else:
                print(f"\rDownloaded: {self.downloaded} bytes", end="")

    try:
        # Get file size
        with urllib.request.urlopen(MODEL_URL) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            progress = ProgressBar(total_size)

            with open(MODEL_PATH, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(len(chunk))

        print(f"\n✅ Model downloaded successfully to {MODEL_PATH}")
        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()  # Clean up partial download
        return False


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
  %(prog)s --model /path/to/model.onnx "Hi"  # Custom model path
        """,
    )

    parser.add_argument("texts", nargs="*", help="Text(s) to classify")

    parser.add_argument("--download", action="store_true", help="Download the ONNX model")

    parser.add_argument("--model", type=str, help="Path to ONNX model file")

    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    parser.add_argument(
        "--chunk-strategy",
        choices=["auto", "truncate", "chunk"],
        default="auto",
        help="How to handle long texts: auto (chunk if >256 bytes), truncate, or chunk",
    )

    parser.add_argument("--version", action="version", version="innit 1.0")

    args = parser.parse_args()

    # Download model if requested
    if args.download:
        success = download_model(force=True)
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
    try:
        detector = InnitDetector(model_path=args.model)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Run with --download to get the model:")
        print(f"   python {sys.argv[0]} --download")
        sys.exit(1)
    except ImportError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Process texts
    results = []
    for text in texts:
        try:
            result = detector.predict(text, chunk_strategy=args.chunk_strategy)
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
