# qwen_mem/memory_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryPlus(nn.Module):
    """
    Simple Memory+ layer.
    This is a direct, small-scale version of the paper's memory block.

    Args:
        d_model: model hidden size (1024 for Qwen2.5-0.5B)
        n_mem:   number of memory slots (start small like 10_000)
        d_key:   key dimension
        d_value: value dimension (usually = d_model)
        k:       top-k memories per token
    """

    def __init__(self, d_model: int, n_mem: int = 10_000,
                 d_key: int = 256, d_value: int = None, k: int = 32):
        super().__init__()
        if d_value is None:
            d_value = d_model

        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.n_mem = n_mem
        self.k = k

        # Trainable memory keys/values
        self.keys = nn.Parameter(torch.randn(n_mem, d_key) * 0.02)
        self.values = nn.Parameter(torch.randn(n_mem, d_value) * 0.02)

        # Query and gating / output projections (Memory+ style)
        self.q_proj = nn.Linear(d_model, d_key, bias=False)
        self.gate_proj = nn.Linear(d_model, d_value, bias=False)
        self.out_proj = nn.Linear(d_value, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, S, d_model]
        returns: [B, S, d_model]
        """
        B, S, D = x.shape

        # Project tokens to queries
        q = self.q_proj(x) # [B, S, d_key]

        # Cosine similarity to all keys
        q_norm = F.normalize(q, dim=-1) # [B, S, d_key]
        k_norm = F.normalize(self.keys, dim=-1) # [n_mem, d_key]
        # [B, S, n_mem]
        sims = torch.matmul(q_norm, k_norm.t())

        # Top-k nearest memory slots
        topk_vals, topk_idx = sims.topk(self.k, dim=-1) # [B, S, k]
        weights = F.softmax(topk_vals, dim=-1) # [B, S, k]

        # Gather values and do weighted sum
        # values[topk_idx]: [B, S, k, d_value]
        mem_vals = self.values[topk_idx]
        # [B, S, d_value]
        mem_out = (weights.unsqueeze(-1) * mem_vals).sum(dim=-2)

        # Memory+ gating and projection back to d_model
        gate = F.silu(self.gate_proj(x)) # [B, S, d_value]
        mem_out = self.out_proj(mem_out * gate) # [B, S, d_model]

        return mem_out
