from __future__ import annotations

from typing import Any


def _resolve_device(torch: Any, requested: str) -> str:
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("device must be one of: auto, cpu, mps")
    mps_available = bool(torch.backends.mps.is_available())
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS device is not available on this machine")
    return {"cpu": "cpu", "mps": "mps", "auto": "mps" if mps_available else "cpu"}[requested]


def _configure_execution(torch: Any, execution_device: str) -> None:
    if execution_device == "cpu":
        torch.set_num_threads(1)


def load_predictor(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[Any, dict[int, str], int]:  # pragma: no cover - optional model extra
    """Load the official CommonLingua checkpoint architecture."""
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("Install model support with `uv sync --extra models`") from error

    execution_device = _resolve_device(torch, device)
    _configure_execution(torch, execution_device)

    class ByteNgramEmbed(nn.Module):
        def __init__(self, num_buckets: int, embed_dim: int, n: int = 3) -> None:
            super().__init__()
            self.n = n
            self.num_buckets = num_buckets
            self.embed = nn.Embedding(num_buckets, embed_dim)

        def forward(self, byte_ids):
            _, length = byte_ids.shape
            clamped = byte_ids.clamp(max=255)
            padded = functional.pad(clamped, (0, self.n - 1), value=0)
            hashed = torch.zeros_like(byte_ids)
            for index in range(self.n):
                hashed = hashed * 257 + padded[:, index : index + length]
            return self.embed(hashed % self.num_buckets)

    class ByteConvBlock(nn.Module):
        def __init__(self, d_model: int, kernel_size: int, expand: int) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.padding = kernel_size - 1
            self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model)
            self.norm2 = nn.LayerNorm(d_model)
            feedforward = d_model * expand
            self.ffn_gate = nn.Linear(d_model, feedforward, bias=False)
            self.ffn_up = nn.Linear(d_model, feedforward, bias=False)
            self.ffn_down = nn.Linear(feedforward, d_model, bias=False)

        def forward(self, inputs):
            residual = inputs
            inputs = self.norm1(inputs).transpose(1, 2)
            inputs = functional.pad(inputs, (self.padding, 0))
            inputs = functional.silu(self.conv(inputs)).transpose(1, 2)
            inputs = residual + inputs
            residual = inputs
            inputs = self.norm2(inputs)
            inputs = self.ffn_down(functional.silu(self.ffn_gate(inputs)) * self.ffn_up(inputs))
            return residual + inputs

    class ByteAttentionBlock(nn.Module):
        def __init__(self, d_model: int, heads: int, expand: int) -> None:
            super().__init__()
            self.heads = heads
            self.head_dim = d_model // heads
            self.norm1 = nn.LayerNorm(d_model)
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.norm2 = nn.LayerNorm(d_model)
            feedforward = d_model * expand
            self.ffn_gate = nn.Linear(d_model, feedforward, bias=False)
            self.ffn_up = nn.Linear(d_model, feedforward, bias=False)
            self.ffn_down = nn.Linear(feedforward, d_model, bias=False)

        def forward(self, inputs):
            batch, length, dimension = inputs.shape
            residual = inputs
            hidden = self.norm1(inputs)
            qkv = self.qkv(hidden).reshape(batch, length, 3, self.heads, self.head_dim)
            query, key, value = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
            attention = (query @ key.transpose(-2, -1)) / (self.head_dim**0.5)
            attention = attention.softmax(dim=-1)
            hidden = (attention @ value).transpose(1, 2).contiguous().view(batch, length, dimension)
            inputs = residual + self.out_proj(hidden)
            residual = inputs
            hidden = self.norm2(inputs)
            hidden = self.ffn_down(functional.silu(self.ffn_gate(hidden)) * self.ffn_up(hidden))
            return residual + hidden

    class ByteHybrid(nn.Module):
        def __init__(self, num_classes: int, max_len: int, config: dict[str, int]) -> None:
            super().__init__()
            d_model = config["d_model"]
            self.embed = nn.Embedding(257, d_model, padding_idx=256)
            self.ngram_embed = ByteNgramEmbed(config["ngram_buckets"], config["ngram_dim"])
            self.ngram_proj = nn.Linear(config["ngram_dim"], d_model, bias=False)
            self.conv_layers = nn.ModuleList(
                ByteConvBlock(d_model, config["conv_kernel"], config["ffn_expand"])
                for _ in range(config["n_conv"])
            )
            self.attn_layers = nn.ModuleList(
                ByteAttentionBlock(d_model, config["n_heads"], config["ffn_expand"])
                for _ in range(config["n_attn"])
            )
            self.final_norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, num_classes),
            )
            self.max_len = max_len

        def forward(self, byte_ids):
            mask = byte_ids != 256
            hidden = self.embed(byte_ids) + self.ngram_proj(self.ngram_embed(byte_ids))
            for layer in self.conv_layers:
                hidden = layer(hidden)
            for layer in self.attn_layers:
                hidden = layer(hidden)
            hidden = self.final_norm(hidden)
            mask = mask.unsqueeze(-1).to(hidden.dtype)
            hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return self.head(hidden)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = {
        "d_model": 256,
        "n_conv": 3,
        "n_attn": 1,
        "n_heads": 4,
        "ffn_expand": 2,
        "conv_kernel": 15,
        "ngram_buckets": 4096,
        "ngram_dim": 64,
    }
    model = ByteHybrid(checkpoint["num_classes"], checkpoint["max_len"], config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(execution_device)
    model.eval()
    index_to_language = {index: language for language, index in checkpoint["lang2idx"].items()}
    max_len = int(checkpoint["max_len"])

    def predict_many(texts):
        text_items = tuple(texts)
        encoded = np.full((len(text_items), max_len), 256, dtype=np.int64)
        for row, text in enumerate(text_items):
            raw = text.encode("utf-8", errors="replace")[:max_len]
            encoded[row, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        with torch.no_grad():
            inputs = torch.from_numpy(encoded).to(execution_device)
            probabilities = torch.softmax(model(inputs).float(), dim=-1)
        confidence, index = probabilities.max(dim=1)
        return tuple(
            (index_to_language[int(predicted_index)], float(predicted_confidence))
            for predicted_index, predicted_confidence in zip(index, confidence, strict=True)
        )

    class Predictor:
        def __call__(self, text: str) -> tuple[str, float]:
            return predict_many((text,))[0]

        def predict_many(self, texts):
            return predict_many(texts)

    return Predictor(), index_to_language, max_len
