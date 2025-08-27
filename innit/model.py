"""Model definitions and utilities for innit"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 256

class TinyByteCNN_EN(nn.Module):
    """Tiny byte-level CNN for English detection"""
    
    def __init__(self, emb=64, blocks=4):
        super().__init__()
        self.emb = nn.Embedding(257, emb)
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(emb, emb, 9, padding=4, groups=emb),
                nn.Conv1d(emb, emb, 1),
                nn.GELU(),
                nn.Dropout(0.1)
            ))
            self.norms.append(nn.LayerNorm(emb))
        
        self.fc = nn.Linear(emb * 2, 2)

    def forward(self, x):
        """Forward pass: x shape (B, L)"""
        h = self.emb(x).transpose(1, 2)  # (B, E, L)
        
        for block, ln in zip(self.blocks, self.norms):
            y = block(h)
            h = (h + y).transpose(1, 2)  # (B, L, E)
            h = ln(h).transpose(1, 2)    # (B, E, L)
        
        # Global pooling
        mean = h.mean(2)  # (B, E)
        mmax = h.amax(2)  # (B, E)
        
        return self.fc(torch.cat([mean, mmax], 1))  # (B, 2)

def to_bytes_tensor(text_bytes, max_len=2048):
    """Convert bytes to padded tensor"""
    x = torch.full((max_len,), PAD, dtype=torch.long)
    if text_bytes:
        length = min(len(text_bytes), max_len)
        x[:length] = torch.tensor(list(text_bytes[:length]), dtype=torch.long)
    return x