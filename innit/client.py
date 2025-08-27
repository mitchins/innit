#!/usr/bin/env python3
"""
Client-friendly helper for innit detector.

Provides simple methods that hide batching and chunking heuristics:
 - classify_document(text, strategy='auto'|'ends'|'chunk'|'truncate', ends_pct=0.1)
 - classify_snippets(texts): vectorized batch classification

Backed by InnitDetector in innit_detector.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from .detector import InnitDetector


@dataclass
class InnitClientConfig:
    backend: str = "auto"  # 'auto' | 'onnx' | 'tinygrad'
    model_path: str | None = None
    # For document classification when strategy='auto'
    ends_pct: float = 0.10
    long_doc_bytes_threshold: int = 1024  # switch to 'ends' if longer than this


class InnitClient:
    """High-level client API around InnitDetector."""

    def __init__(self, config: InnitClientConfig | None = None):
        self.config = config or InnitClientConfig()
        backend = self.config.backend
        if backend == "auto":
            # Prefer ONNX if available, else TinyGrad
            try:
                import onnxruntime  # noqa: F401

                backend = "onnx"
            except Exception:
                try:
                    import safetensors  # noqa: F401
                    import tinygrad  # noqa: F401

                    backend = "tinygrad"
                except Exception:
                    raise ImportError(
                        "No backend found. Install one: pip install 'innit-detector[onnx]' or 'innit-detector[tinygrad]'"
                    )
        self.detector = InnitDetector(model_path=self.config.model_path, backend=backend)

    # --- Single snippet ---
    def classify(self, text: str, *, strategy: str = "auto", ends_pct: float | None = None) -> dict:
        """Classify a single snippet.

        strategy: 'auto'|'truncate'|'chunk'|'ends'
        ends_pct: overrides default ends percentage when strategy='ends' or auto chooses ends
        """
        ep = self.config.ends_pct if ends_pct is None else float(ends_pct)

        # Direct pass-through; InnitDetector handles chunking internally
        return self.detector.predict(text, chunk_strategy=strategy, ends_pct=ep)

    # --- Whole document ---
    def classify_document(
        self, text: str, *, strategy: str = "auto", ends_pct: float | None = None
    ) -> dict:
        """Classify a full document with hidden heuristics.

        - If strategy == 'auto': use 'ends' for long docs, else single-chunk.
        - If strategy is provided explicitly, defer to detector.
        """
        ep = self.config.ends_pct if ends_pct is None else float(ends_pct)
        if strategy == "auto":
            tb = text.encode("utf-8", errors="ignore")
            chosen = "ends" if len(tb) >= self.config.long_doc_bytes_threshold else "auto"
            return self.detector.predict(text, chunk_strategy=chosen, ends_pct=ep)
        return self.detector.predict(text, chunk_strategy=strategy, ends_pct=ep)

    # JS-style alias if desired
    def classifyDocument(self, text: str, **kwargs) -> dict:
        return self.classify_document(text, **kwargs)

    # --- Bulk snippets ---
    def classify_snippets(self, texts: list[str]) -> list[dict]:
        """Vectorized classification for multiple short snippets."""
        return self.detector.predict_batch_fast(texts)
