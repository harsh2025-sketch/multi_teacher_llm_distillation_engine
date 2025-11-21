#!/usr/bin/env python3
"""
Automated Multi-Teacher LLM Distillation Engine.

Complete end-to-end automation for knowledge distillation from multiple teacher models.
Supports HuggingFace models, custom datasets, and full training pipeline automation.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from torch.cuda.amp import GradScaler, autocast
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from config import DistillationConfig
from model_manager import ModelManager
from dataset_manager import DatasetManager
from student_architecture import StudentLLM, StudentArchitectureConfig
from utils import (
    setup_logging, set_seed, get_device, format_time,
    format_number, Timer, ProgressTracker, print_banner,
    save_dict_to_json, print_gpu_memory
)

logger = logging.getLogger(__name__)


class MultiTeacherDistillationLoss(nn.Module):
    """Loss function for multi-teacher knowledge distillation."""
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.7,
        teacher_weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_weights = teacher_weights or []
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: List[torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: [batch, seq_len, vocab_size]
            teacher_logits_list: List of teacher logits
            labels: [batch, seq_len]
        
        Returns:
            Dictionary with loss components
        """
        # Average teacher logits
        if self.teacher_weights:
            teacher_logits = sum(
                w * logits for w, logits in zip(self.teacher_weights, teacher_logits_list)
            )
        else:
            teacher_logits = sum(teacher_logits_list) / len(teacher_logits_list)
        
        # Shift for next-token prediction
        shift_student = student_logits[:, :-1, :].contiguous()
        shift_teacher = teacher_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Flatten
        shift_student = shift_student.view(-1, student_logits.size(-1))
        shift_teacher = shift_teacher.view(-1, teacher_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Create mask for valid positions
        mask = shift_labels != -100
        
        # Distillation loss (soft targets)
        student_log_probs = F.log_softmax(shift_student / self.temperature, dim=-1)
        teacher_probs = F.softmax(shift_teacher / self.temperature, dim=-1)
        
        distill_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='none'
        ).sum(dim=-1)
        
        distill_loss = (distill_loss * mask.float()).sum() / mask.float().sum()
        distill_loss = distill_loss * (self.temperature ** 2)
        
        # Hard label loss
        hard_loss = F.cross_entropy(
            shift_student,
            shift_labels,
            ignore_index=-100
        )
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return {
            'total_loss': total_loss,
            'distill_loss': distill_loss.detach(),
            'hard_loss': hard_loss.detach()
        }


class AutomatedDistillationEngine:
    """
    Automated engine for multi-teacher LLM distillation.
    
    Handles complete workflow:
    1. Load configuration
    2. Download/load teacher models
    3. Load/preprocess dataset
    4. Initialize student model
    5. Train with progress tracking
    6. Save checkpoints and final model
    """
    
    def __init__(self, config: DistillationConfig):
        """
        Initialize distillation engine.
        
        Args:
            config: Complete distillation configuration
        """
        self.config = config
        
        # Setup logging
        log_file = Path(config.output_dir) / "training.log"
        setup_logging(log_file=str(log_file))
        
        # Set seed
        set_seed(config.seed)
        
        # Get device
        self.device = get_device()
        
        # Initialize managers
        self.model_manager = ModelManager(cache_dir=config.cache_dir, device=self.device)
        self.dataset_manager = DatasetManager(cache_dir=config.cache_dir)
        
        # Placeholders
        self.teacher_models = None
        self.teacher_tokenizers = None
        self.student_model = None
        self.student_tokenizer = None
        self.train_dataloader = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.loss_fn = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Metrics
        self.metrics = {
            'steps': [],
            'epochs': [],
            'total_loss': [],
            'distill_loss': [],
            'hard_loss': [],
            'learning_rate': []
        }
        
        logger.info("\n" + "="*80)
        logger.info("Automated Multi-Teacher LLM Distillation Engine")
        logger.info("="*80)
        logger.info(f"Project: {config.project_name}")
        logger.info(f"Output: {config.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info("="*80 + "\n")
    
    def setup(self):
        """Complete setup: load models, data, initialize training."""
        logger.info("Starting setup...")
        
        # 1. Load teacher models
        logger.info("\n[Step 1/5] Loading teacher models...")
        with Timer("Teacher loading"):
            self.teacher_models, self.teacher_tokenizers = self.model_manager.load_teachers(
                self.config.teachers
            )
            
            # Check compatibility
            self.model_manager.check_model_compatibility(self.teacher_models)
        
        # 2. Select primary tokenizer
        logger.info("\n[Step 2/5] Setting up tokenizer...")
        self.student_tokenizer = self.model_manager.align_tokenizers(
            self.teacher_tokenizers,
            target_vocab_size=self.config.student.vocab_size
        )
        
        # 3. Load and prepare dataset
        logger.info("\n[Step 3/5] Loading dataset...")
        with Timer("Dataset loading"):
            texts = self.dataset_manager.load_dataset(self.config.dataset)
            texts = self.dataset_manager.validate_dataset(texts, min_length=10)
            
            # Create dataloader
            self.train_dataloader = self.dataset_manager.create_dataloader(
                texts=texts,
                tokenizer=self.student_tokenizer,
                batch_size=self.config.training.batch_size,
                max_length=self.config.training.max_length,
                num_workers=self.config.training.dataloader_num_workers,
                pin_memory=self.config.training.dataloader_pin_memory
            )
        
        # 4. Initialize student model
        logger.info("\n[Step 4/5] Initializing student model...")
        with Timer("Student initialization"):
            student_config = StudentArchitectureConfig(**self.config.student.__dict__)
            self.student_model = StudentLLM(student_config).to(self.device)
            
            num_params = self.student_model.num_parameters()
            logger.info(f"Student parameters: {format_number(num_params)} ({num_params/1e6:.1f}M)")
        
        # 5. Setup training components
        logger.info("\n[Step 5/5] Setting up training components...")
        self._setup_training()
        
        logger.info("\n" + "="*80)
        logger.info("Setup completed successfully!")
        logger.info("="*80 + "\n")
        
        if torch.cuda.is_available():
            print_gpu_memory()
    
    def _setup_training(self):
        """Setup optimizer, scheduler, loss function."""
        config = self.config.training
        
        # Loss function
        teacher_weights = [t.weight for t in self.config.teachers]
        self.loss_fn = MultiTeacherDistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha,
            teacher_weights=teacher_weights
        )
        
        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight', 'norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.student_model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': config.weight_decay,
            },
            {
                'params': [
                    p for n, p in self.student_model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon
        )
        
        # Scheduler
        num_training_steps = len(self.train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        
        self.scheduler = get_scheduler(
            name=config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Mixed precision scaler
        if config.use_fp16 or config.use_bf16:
            self.scaler = GradScaler() if config.use_fp16 else None
        
        logger.info(f"  Optimizer: AdamW")
        logger.info(f"  Learning rate: {config.learning_rate:.2e}")
        logger.info(f"  Scheduler: {config.lr_scheduler_type}")
        logger.info(f"  Warmup steps: {num_warmup_steps}")
        logger.info(f"  Total steps: {num_training_steps}")
        logger.info(f"  Mixed precision: {'FP16' if config.use_fp16 else 'BF16' if config.use_bf16 else 'None'}")
    
    @torch.no_grad()
    def get_teacher_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[torch.Tensor]:
        """Get logits from all teachers."""
        teacher_logits_list = []
        
        for teacher in self.teacher_models:
            outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            teacher_logits_list.append(logits)
        
        return teacher_logits_list
    
    def train(self):
        """Execute complete training loop."""
        config = self.config.training
        
        print_banner("STARTING TRAINING")
        
        logger.info(f"Configuration:")
        logger.info(f"  Epochs: {config.num_epochs}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        logger.info(f"  Teachers: {len(self.teacher_models)}")
        logger.info(f"  Dataset size: {len(self.train_dataloader.dataset)}")
        logger.info(f"  Batches per epoch: {len(self.train_dataloader)}")
        
        total_steps = len(self.train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
        progress_tracker = ProgressTracker(total_steps=total_steps, log_interval=config.logging_steps)
        
        start_time = time.time()
        self.student_model.train()
        
        # Training loop
        for epoch in range(config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            epoch_steps = 0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{config.num_epochs}",
                total=len(self.train_dataloader)
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = input_ids.clone()
                
                # Get teacher logits
                teacher_logits_list = self.get_teacher_logits(input_ids, attention_mask)
                
                # Forward pass
                use_amp = config.use_fp16 or config.use_bf16
                dtype = torch.float16 if config.use_fp16 else torch.bfloat16 if config.use_bf16 else torch.float32
                
                if use_amp:
                    with autocast(dtype=dtype):
                        outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask)
                        student_logits = outputs['logits']
                        
                        loss_dict = self.loss_fn(student_logits, teacher_logits_list, labels)
                        loss = loss_dict['total_loss'] / config.gradient_accumulation_steps
                    
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask)
                    student_logits = outputs['logits']
                    
                    loss_dict = self.loss_fn(student_logits, teacher_logits_list, labels)
                    loss = loss_dict['total_loss'] / config.gradient_accumulation_steps
                    loss.backward()
                
                # Update weights
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    
                    nn.utils.clip_grad_norm_(self.student_model.parameters(), config.max_grad_norm)
                    
                    # Optimizer step
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update state
                    self.global_step += 1
                    progress_tracker.update()
                    epoch_loss += loss_dict['total_loss'].item()
                    epoch_steps += 1
                    
                    # Log metrics
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.metrics['steps'].append(self.global_step)
                    self.metrics['epochs'].append(epoch)
                    self.metrics['total_loss'].append(loss_dict['total_loss'].item())
                    self.metrics['distill_loss'].append(loss_dict['distill_loss'].item())
                    self.metrics['hard_loss'].append(loss_dict['hard_loss'].item())
                    self.metrics['learning_rate'].append(current_lr)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss_dict['total_loss'].item():.4f}",
                        'lr': f"{current_lr:.2e}",
                        'step': self.global_step
                    })
                    
                    # Save checkpoint
                    if self.global_step % config.save_steps == 0:
                        self.save_checkpoint()
            
            # Epoch summary
            avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
            logger.info(f"\nEpoch {epoch+1} completed - Avg loss: {avg_loss:.4f}")
        
        # Training complete
        total_time = time.time() - start_time
        print_banner("TRAINING COMPLETED")
        logger.info(f"Total time: {format_time(total_time)}")
        logger.info(f"Total steps: {self.global_step}")
        
        # Save final model
        self.save_final_model()
        
        return self.metrics
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'config': self.config.to_dict(),
            'metrics': self.metrics
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path.name}")
        
        # Cleanup old checkpoints
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > self.config.training.save_total_limit:
            for old_ckpt in checkpoints[:-self.config.training.save_total_limit]:
                old_ckpt.unlink()
    
    def save_final_model(self):
        """Save final trained model."""
        output_dir = Path(self.config.output_dir) / "final_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.student_model.state_dict(), output_dir / "pytorch_model.bin")
        
        # Save config
        self.config.to_yaml(output_dir / "config.yaml")
        
        # Save metrics
        save_dict_to_json(self.metrics, output_dir / "training_metrics.json")
        
        logger.info(f"\nFinal model saved to: {output_dir}")


# Example usage
if __name__ == "__main__":
    from config import create_example_config
    
    # Create example configuration
    config = create_example_config("example_config.yaml")
    
    # Initialize engine
    engine = AutomatedDistillationEngine(config)
    
    # Run setup and training
    engine.setup()
    metrics = engine.train()
    
    print("\nTraining completed successfully!")
