"""Model asset helpers (loader, paths).

Provides SafeTensors loading without requiring torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def load_safetensors_numpy(path: str | Path) -> Dict[str, "np.ndarray"]:
    """Load a SafeTensors file into a dict of NumPy arrays.

    This avoids importing torch at runtime.
    """
    from safetensors.numpy import load_file  # lazy import

    return load_file(str(path))

