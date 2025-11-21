#!/usr/bin/env python3
"""
Loss functions for knowledge distillation.
Combines soft targets (KL divergence) with hard targets (cross-entropy).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Combined loss for multi-teacher knowledge distillation.
    
    Components:
    - Soft targets: KL divergence with averaged teacher logits
    - Hard targets: Cross-entropy with ground truth labels
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        vocab_size: int = 128256
    ):
        """
        Args:
            temperature: Softening temperature for distillation (higher = softer)
            alpha: Weight for distillation loss (1-alpha for hard label loss)
            vocab_size: Vocabulary size for decompressing logits
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def decompress_logits(
        self,
        values: torch.Tensor,
        indices: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Decompress Top-K logits to full vocabulary size.
        
        Args:
            values: [batch, seq, top_k] - top-k logit values
            indices: [batch, seq, top_k] - top-k vocabulary indices
            device: target device
            
        Returns:
            full_logits: [batch, seq, vocab_size]
        """
        batch_size, seq_len, top_k = values.shape
        
        # Initialize full logits with very negative values
        full_logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            fill_value=-1e4,
            dtype=torch.float32,
            device=device
        )
        
        # Scatter top-k values into full logits
        # Optimized vectorized version
        batch_indices = torch.arange(batch_size, device=device)[:, None, None].expand_as(indices)
        seq_indices = torch.arange(seq_len, device=device)[None, :, None].expand_as(indices)
        
        full_logits[batch_indices, seq_indices, indices] = values.float()
        
        return full_logits
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher1_values: torch.Tensor,
        teacher1_indices: torch.Tensor,
        teacher2_values: torch.Tensor,
        teacher2_indices: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """
        Calculate combined distillation loss.
        
        Args:
            student_logits: [batch, seq, vocab] - student model output
            teacher1_values: [batch, seq, top_k] - first teacher compressed logits
            teacher1_indices: [batch, seq, top_k] - first teacher indices
            teacher2_values: [batch, seq, top_k] - second teacher compressed logits
            teacher2_indices: [batch, seq, top_k] - second teacher indices
            labels: [batch, seq] - ground truth token IDs
            
        Returns:
            dict with total_loss, distill_loss, hard_loss
        """
        device = student_logits.device
        
        # Decompress teacher logits
        teacher1_logits = self.decompress_logits(teacher1_values, teacher1_indices, device)
        teacher2_logits = self.decompress_logits(teacher2_values, teacher2_indices, device)
        
        # Average teacher logits
        teacher_logits = (teacher1_logits + teacher2_logits) / 2.0
        
        # Shift for next-token prediction
        student_logits_shifted = student_logits[:, :-1, :].contiguous()
        teacher_logits_shifted = teacher_logits[:, :-1, :].contiguous()
        labels_shifted = labels[:, 1:].contiguous()
        
        # Flatten batch and sequence dimensions
        batch_size, seq_len, vocab_size = student_logits_shifted.shape
        student_logits_flat = student_logits_shifted.view(-1, vocab_size)
        teacher_logits_flat = teacher_logits_shifted.view(-1, vocab_size)
        labels_flat = labels_shifted.view(-1)
        
        # Soft target loss (KL divergence)
        student_log_probs = F.log_softmax(
            student_logits_flat / self.temperature,
            dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits_flat / self.temperature,
            dim=-1
        )
        distill_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Hard target loss (cross-entropy)
        hard_loss = self.ce_loss(student_logits_flat, labels_flat)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return {
            'total_loss': total_loss,
            'distill_loss': distill_loss.item(),
            'hard_loss': hard_loss.item(),
        }


def create_distillation_loss(
    temperature: float = 2.0,
    alpha: float = 0.5,
    vocab_size: int = 128256
) -> DistillationLoss:
    """
    Convenience function to create distillation loss.
    
    Args:
        temperature: Softening temperature (default: 2.0)
        alpha: Weight for distillation vs hard loss (default: 0.5)
        vocab_size: Vocabulary size
        
    Returns:
        DistillationLoss instance
    """
    return DistillationLoss(
        temperature=temperature,
        alpha=alpha,
        vocab_size=vocab_size
    )
