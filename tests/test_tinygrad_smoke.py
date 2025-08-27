#!/usr/bin/env python3
"""TinyGrad smoke tests.

Run directly or via pytest. These tests avoid network and use local weights
from the repo (or ~/.innit cache if present).
"""

import math

from innit import InnitClient, InnitClientConfig


def test_tinygrad_basic_predictions():
    client = InnitClient(InnitClientConfig(backend="tinygrad"))
    samples = [
        ("Hello world!", True),
        ("Bonjour le monde!", False),
        ("你好世界！", False),
        ("This is English text.", True),
        ("Este es español.", False),
    ]
    for text, expected in samples:
        res = client.classify(text)
        assert isinstance(res["is_english"], bool)
        assert res["is_english"] == expected
        p_en = res["probabilities"]["english"]
        p_other = res["probabilities"]["non_english"]
        assert 0.0 <= p_en <= 1.0 and 0.0 <= p_other <= 1.0
        assert math.isclose(p_en + p_other, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_batch_fast_matches_elementwise():
    client = InnitClient(InnitClientConfig(backend="tinygrad"))
    texts = [
        "Hello world!",
        "Bonjour le monde!",
        "你好世界！",
        "This is English text.",
        "Este es español.",
    ]
    fast = client.classify_snippets(texts)
    slow = [client.classify(t) for t in texts]
    for a, b in zip(fast, slow, strict=False):
        assert a["is_english"] == b["is_english"]


if __name__ == "__main__":
    # Allow running as a script for quick smoke
    test_tinygrad_basic_predictions()
    test_batch_fast_matches_elementwise()
    print("TinyGrad smoke tests passed")
