"""Small executable architecture fixtures, not pretrained language models.

GPT structure reference: github.com/karpathy/nanoGPT/blob/master/model.py
LLaMA structure reference: github.com/meta-llama/llama/blob/main/llama/model.py
Implementations below are independent, native PyTorch, with deliberately small dimensions.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab: int = 32
    width: int = 16
    heads: int = 4
    kv_heads: int = 2
    layers: int = 2
    hidden: int = 32
    context: int = 32

    def __post_init__(self):
        if any(not isinstance(v,int) or v<=0 for v in vars(self).values()):
            raise ValueError('All model dimensions must be positive integers.')
        if self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError('width must be divisible by heads; heads must be divisible by kv_heads.')


class CausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.dim = cfg.heads, cfg.width // cfg.heads
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.projection = nn.Linear(cfg.width, cfg.width)

    def forward(self, x):
        b, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (a.reshape(b, length, self.heads, self.dim).transpose(1, 2) for a in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.projection(y.transpose(1, 2).contiguous().reshape(b, length, width))


class GPTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention_norm = nn.LayerNorm(cfg.width)
        self.attention = CausalAttention(cfg)
        self.ffn_norm = nn.LayerNorm(cfg.width)
        self.ffn = nn.Sequential(nn.Linear(cfg.width, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, cfg.width))

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or Config()
        self.token_embedding = nn.Embedding(cfg.vocab, cfg.width)
        self.position_embedding = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.ModuleList(GPTBlock(cfg) for _ in range(cfg.layers))
        self.final_norm = nn.LayerNorm(cfg.width)
        self.lm_head = nn.Linear(cfg.width, cfg.vocab, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


class RotaryEmbedding(nn.Module):
    """Adjacent pair rotation, equivalent to Meta's complex pair convention."""
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.width // cfg.heads
        if dim % 2:
            raise ValueError('RoPE requires an even head dimension.')
        inverse = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        angles = torch.arange(cfg.context).float()[:, None] * inverse[None, :]
        self.register_buffer('cosine', angles.cos())
        self.register_buffer('sine', angles.sin())

    def forward(self, x):
        length = x.shape[-2]
        cos, sin = self.cosine[:length], self.sine[:length]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.kv_heads, self.dim = cfg.heads, cfg.kv_heads, cfg.width // cfg.heads
        self.q = nn.Linear(cfg.width, cfg.width, bias=False)
        self.k = nn.Linear(cfg.width, cfg.kv_heads * self.dim, bias=False)
        self.v = nn.Linear(cfg.width, cfg.kv_heads * self.dim, bias=False)
        self.rope = RotaryEmbedding(cfg)
        self.projection = nn.Linear(cfg.width, cfg.width, bias=False)

    def forward(self, x):
        b, length, width = x.shape
        q = self.rope(self.q(x).reshape(b, length, self.heads, self.dim).transpose(1, 2))
        k = self.rope(self.k(x).reshape(b, length, self.kv_heads, self.dim).transpose(1, 2))
        v = self.v(x).reshape(b, length, self.kv_heads, self.dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.projection(y.transpose(1, 2).contiguous().reshape(b, length, width))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.width, cfg.hidden, bias=False)
        self.up = nn.Linear(cfg.width, cfg.hidden, bias=False)
        self.down = nn.Linear(cfg.hidden, cfg.width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LlamaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention_norm = nn.RMSNorm(cfg.width, eps=1e-5)
        self.attention = GroupedQueryAttention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.width, eps=1e-5)
        self.ffn = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class LlamaStyle(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or Config()
        self.token_embedding = nn.Embedding(cfg.vocab, cfg.width)
        self.blocks = nn.ModuleList(LlamaBlock(cfg) for _ in range(cfg.layers))
        self.final_norm = nn.RMSNorm(cfg.width, eps=1e-5)
        self.lm_head = nn.Linear(cfg.width, cfg.vocab, bias=False)

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def build_tinygpt():
    torch.manual_seed(42)
    return {'model': TinyGPT(), 'args': (torch.tensor([[1, 7, 3, 12, 5, 9]]),), 'options': {'backend': 'export'}}


def build_llama():
    torch.manual_seed(42)
    return {'model': LlamaStyle(), 'args': (torch.tensor([[1, 7, 3, 12, 5, 9]]),), 'options': {'backend': 'export'}}


build_model = build_tinygpt
