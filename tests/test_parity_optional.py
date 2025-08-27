#!/usr/bin/env python3
"""Optional parity tests with ONNX.

Skips gracefully if onnxruntime or ONNX model is not available.
"""

import os
import math
from pathlib import Path

import numpy as np

from innit import InnitClient, InnitClientConfig


def _has_onnx():
    try:
        import onnxruntime as _  # noqa: F401
    except Exception:
        return False
    return (Path.home() / ".innit" / "model.onnx").exists()


def test_onnx_tinygrad_parity_small_suite():
    if not _has_onnx():
        return  # skip silently

    tg = InnitClient(InnitClientConfig(backend="tinygrad"))
    onnx = InnitClient(InnitClientConfig(backend="onnx"))

    texts = [
        "Hello world!",
        "Bonjour le monde!",
        "你好世界！",
        "This is English text.",
        "Este es español.",
    ]
    tg_res = tg.classify_snippets(texts)
    onnx_res = onnx.classify_snippets(texts)
    for a, b in zip(tg_res, onnx_res, strict=False):
        assert a["is_english"] == b["is_english"]


if __name__ == "__main__":
    test_onnx_tinygrad_parity_small_suite()
    print("Parity test completed (skipped if ONNX unavailable)")

