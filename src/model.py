#!/usr/bin/env python3
"""
Student Model Architecture for Multi-Teacher Snapshot Distillation.
LLaMA-style transformer with modern components (RMSNorm, RoPE, SwiGLU).
Designed for ~250M parameters.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class StudentConfig:
    """Configuration for the student model."""
    
    vocab_size: int = 128256
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More efficient than LayerNorm, used in LLaMA/Gemma models.
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(x.dtype)
    
    def reset_parameters(self):
        """FSDP-friendly initialization."""
        with torch.no_grad():
            self.weight.fill_(1.0)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    Applies rotation to queries and keys to encode position information.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin embeddings."""
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys."""
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim]
    
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """
    SwiGLU activation function (Swish-Gated Linear Unit).
    Combines SiLU activation with Gated Linear Unit.
    Formula: SwiGLU(x) = SiLU(gate(x)) ⊙ up(x)
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        gate = F.silu(gate)
        up = self.up_proj(x)
        intermediate = gate * up
        output = self.down_proj(intermediate)
        return output
    
    def reset_parameters(self):
        """FSDP-friendly initialization."""
        for module in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class Attention(nn.Module):
    """Multi-Head Self-Attention with RoPE position embeddings."""
    
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = config.attention_dropout
        
        assert self.hidden_size % self.num_heads == 0, \
            "hidden_size must be divisible by num_heads"
        
        # Q, K, V projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # RoPE embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        # Dropout
        self.dropout = nn.Dropout(self.attention_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, seq_len, head_dim]
        
        # Apply RoPE
        cos, sin = self.rotary_emb(q, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output
    
    def reset_parameters(self):
        """FSDP-friendly initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-normalization.
    Architecture: Input → RMSNorm → Attention → Add → RMSNorm → SwiGLU → Add → Output
    """
    
    def __init__(self, config: StudentConfig):
        super().__init__()
        
        # Pre-normalization layers
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Attention and feedforward
        self.self_attn = Attention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Feedforward with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def reset_parameters(self):
        """FSDP-friendly initialization."""
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.reset_parameters()
        self.mlp.reset_parameters()


class StudentModel(nn.Module):
    """
    Complete student model for multi-teacher distillation.
    LLaMA-style architecture with ~250M parameters.
    """
    
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Language modeling head (tied with embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
            labels: [batch_size, seq_length] for loss computation
            
        Returns:
            Dictionary with 'logits' and optional 'loss'
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare causal attention mask
        batch_size, seq_length = input_ids.shape
        
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length),
                dtype=torch.long,
                device=input_ids.device
            )
        
        # Create causal mask
        causal_mask = torch.tril(
            torch.ones(
                (seq_length, seq_length),
                dtype=torch.bool,
                device=input_ids.device
            )
        ).view(1, 1, seq_length, seq_length)
        
        # Combine with padding mask
        attention_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        attention_mask = attention_mask & causal_mask
        
        # Convert to attention scores format (0 for attend, -inf for mask)
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'logits': logits,
            'loss': loss,
        }
    
    def num_parameters(self, only_trainable: bool = False) -> int:
        """Count model parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def reset_parameters(self):
        """FSDP-friendly initialization."""
        self.apply(self._init_weights)
        for layer in self.layers:
            layer.reset_parameters()
        self.norm.reset_parameters()


def create_student_model(
    vocab_size: int = 128256,
    hidden_size: int = 768,
    num_layers: int = 12
) -> StudentModel:
    """
    Convenience function to create a student model.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        
    Returns:
        StudentModel instance
    """
    config = StudentConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=hidden_size // 64,
        intermediate_size=hidden_size * 8 // 3,
    )
    return StudentModel(config)
