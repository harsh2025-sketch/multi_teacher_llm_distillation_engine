#!/usr/bin/env python3
"""
Model Manager for Automated LLM Distillation Engine.

Handles automatic downloading, loading, and management of teacher models
from HuggingFace Hub with support for quantization, caching, and tokenizer alignment.
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from huggingface_hub import hf_hub_download, snapshot_download

from config import TeacherConfig

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages teacher model loading and caching for distillation.
    
    Features:
    - Automatic download from HuggingFace Hub
    - Quantization (4-bit, 8-bit)
    - Tokenizer management and alignment
    - Model caching and reuse
    - Device placement optimization
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize model manager.
        
        Args:
            cache_dir: Directory for caching downloaded models
            device: Target device (auto-detected if None)
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cache for loaded models and tokenizers
        self._model_cache: Dict[str, PreTrainedModel] = {}
        self._tokenizer_cache: Dict[str, PreTrainedTokenizer] = {}
        
        logger.info(f"ModelManager initialized")
        logger.info(f"  Cache directory: {self.cache_dir}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def load_teacher(
        self,
        teacher_config: TeacherConfig,
        force_reload: bool = False
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a teacher model and tokenizer.
        
        Args:
            teacher_config: Configuration for the teacher model
            force_reload: Force reload even if cached
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_id = teacher_config.model_id
        cache_key = f"{model_id}_{teacher_config.use_8bit}_{teacher_config.use_4bit}"
        
        # Check cache
        if not force_reload and cache_key in self._model_cache:
            logger.info(f"Using cached model: {model_id}")
            return self._model_cache[cache_key], self._tokenizer_cache[cache_key]
        
        logger.info(f"Loading teacher model: {model_id}")
        logger.info(f"  Using 8-bit: {teacher_config.use_8bit}")
        logger.info(f"  Using 4-bit: {teacher_config.use_4bit}")
        
        # Prepare quantization config
        quantization_config = None
        if teacher_config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("  Using 4-bit NF4 quantization with double quantization")
        elif teacher_config.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            logger.info("  Using 8-bit LLM.int8() quantization")
        
        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map=teacher_config.device_map,
                trust_remote_code=teacher_config.trust_remote_code,
                cache_dir=self.cache_dir,
                token=teacher_config.token,
                revision=teacher_config.revision,
                torch_dtype=torch.bfloat16 if quantization_config is None else None
            )
            
            # Set to eval mode
            model.eval()
            
            # Disable gradient computation for teacher
            for param in model.parameters():
                param.requires_grad = False
            
            logger.info(f"  ✓ Model loaded successfully")
            
            # Get model memory footprint
            if hasattr(model, 'get_memory_footprint'):
                memory_mb = model.get_memory_footprint() / 1024**2
                logger.info(f"  Memory footprint: {memory_mb:.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Could not load teacher model '{model_id}': {e}")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=teacher_config.trust_remote_code,
                cache_dir=self.cache_dir,
                token=teacher_config.token,
                revision=teacher_config.revision
            )
            
            # Set padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"  Set pad_token to eos_token: {tokenizer.eos_token}")
            
            logger.info(f"  ✓ Tokenizer loaded successfully")
            logger.info(f"  Vocab size: {len(tokenizer)}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_id}: {e}")
            raise RuntimeError(f"Could not load tokenizer for '{model_id}': {e}")
        
        # Cache the model and tokenizer
        self._model_cache[cache_key] = model
        self._tokenizer_cache[cache_key] = tokenizer
        
        return model, tokenizer
    
    def load_teachers(
        self,
        teacher_configs: List[TeacherConfig]
    ) -> Tuple[List[PreTrainedModel], List[PreTrainedTokenizer]]:
        """
        Load multiple teacher models.
        
        Args:
            teacher_configs: List of teacher configurations
        
        Returns:
            Tuple of (list of models, list of tokenizers)
        """
        if not teacher_configs:
            raise ValueError("At least one teacher config required")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Loading {len(teacher_configs)} teacher model(s)")
        logger.info(f"{'='*80}\n")
        
        models = []
        tokenizers = []
        
        for i, config in enumerate(teacher_configs, 1):
            logger.info(f"[Teacher {i}/{len(teacher_configs)}]")
            model, tokenizer = self.load_teacher(config)
            models.append(model)
            tokenizers.append(tokenizer)
            logger.info("")
        
        logger.info(f"{'='*80}")
        logger.info(f"All teachers loaded successfully!")
        logger.info(f"{'='*80}\n")
        
        return models, tokenizers
    
    def align_tokenizers(
        self,
        tokenizers: List[PreTrainedTokenizer],
        target_vocab_size: Optional[int] = None
    ) -> PreTrainedTokenizer:
        """
        Align multiple tokenizers and select primary tokenizer.
        
        For distillation, we typically use the tokenizer from the first teacher
        or the one with vocabulary closest to the target size.
        
        Args:
            tokenizers: List of tokenizers from teachers
            target_vocab_size: Target vocabulary size for student
        
        Returns:
            Selected primary tokenizer
        """
        if not tokenizers:
            raise ValueError("At least one tokenizer required")
        
        logger.info("Aligning tokenizers...")
        
        # Log vocab sizes
        for i, tokenizer in enumerate(tokenizers, 1):
            logger.info(f"  Teacher {i} vocab size: {len(tokenizer)}")
        
        # Select primary tokenizer
        if target_vocab_size is not None:
            # Select tokenizer closest to target
            vocab_diffs = [abs(len(t) - target_vocab_size) for t in tokenizers]
            primary_idx = vocab_diffs.index(min(vocab_diffs))
            logger.info(f"  Selected teacher {primary_idx+1} tokenizer (closest to target size {target_vocab_size})")
        else:
            # Use first teacher tokenizer
            primary_idx = 0
            logger.info(f"  Using teacher 1 tokenizer as primary")
        
        primary_tokenizer = tokenizers[primary_idx]
        
        # Check for significant vocabulary differences
        max_diff = max(abs(len(t) - len(primary_tokenizer)) for t in tokenizers)
        if max_diff > 1000:
            logger.warning(
                f"  WARNING: Large vocabulary difference detected (max diff: {max_diff}). "
                f"This may affect distillation quality."
            )
        
        return primary_tokenizer
    
    def check_model_compatibility(
        self,
        models: List[PreTrainedModel]
    ) -> Dict[str, Any]:
        """
        Check compatibility between teacher models.
        
        Returns compatibility report with warnings if needed.
        
        Args:
            models: List of teacher models
        
        Returns:
            Dictionary with compatibility information
        """
        logger.info("Checking model compatibility...")
        
        report = {
            'compatible': True,
            'warnings': [],
            'info': {}
        }
        
        # Check model architectures
        architectures = []
        for i, model in enumerate(models, 1):
            arch = model.config.model_type
            architectures.append(arch)
            logger.info(f"  Teacher {i}: {arch}")
        
        report['info']['architectures'] = architectures
        
        # Warn if architectures differ significantly
        unique_archs = set(architectures)
        if len(unique_archs) > 1:
            report['warnings'].append(
                f"Different architectures detected: {unique_archs}. "
                f"Distillation may be less effective."
            )
        
        # Check hidden sizes
        hidden_sizes = []
        for model in models:
            hidden_size = getattr(model.config, 'hidden_size', None)
            hidden_sizes.append(hidden_size)
        
        report['info']['hidden_sizes'] = hidden_sizes
        
        if any(h is None for h in hidden_sizes):
            report['warnings'].append("Could not determine hidden size for all models")
        elif max(hidden_sizes) / min(hidden_sizes) > 2:
            report['warnings'].append(
                f"Large hidden size difference detected: {min(hidden_sizes)} to {max(hidden_sizes)}. "
                f"Consider using teachers of similar size."
            )
        
        # Print warnings
        if report['warnings']:
            logger.warning("Compatibility warnings:")
            for warning in report['warnings']:
                logger.warning(f"  WARNING: {warning}")
        else:
            logger.info("  Models are compatible")
        
        return report
    
    def get_model_info(self, model: PreTrainedModel) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model: Model to inspect
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': model.config.model_type,
            'architectures': getattr(model.config, 'architectures', []),
            'hidden_size': getattr(model.config, 'hidden_size', None),
            'num_layers': getattr(model.config, 'num_hidden_layers', None),
            'num_attention_heads': getattr(model.config, 'num_attention_heads', None),
            'vocab_size': getattr(model.config, 'vocab_size', None),
            'max_position_embeddings': getattr(model.config, 'max_position_embeddings', None),
        }
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info['total_parameters'] = total_params
        info['trainable_parameters'] = trainable_params
        info['parameters_M'] = total_params / 1e6
        
        return info
    
    def clear_cache(self):
        """Clear model and tokenizer cache."""
        self._model_cache.clear()
        self._tokenizer_cache.clear()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model cache cleared")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.clear_cache()


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    # Create model manager
    manager = ModelManager(cache_dir="./cache")
    
    # Example: Load teachers
    teacher_configs = [
        TeacherConfig(
            model_id="gpt2",  # Small model for testing
            weight=0.6,
            use_8bit=False  # GPT-2 is small, no quantization needed
        ),
        TeacherConfig(
            model_id="distilgpt2",
            weight=0.4,
            use_8bit=False
        )
    ]
    
    print("\nLoading teacher models...")
    models, tokenizers = manager.load_teachers(teacher_configs)
    
    print("\nChecking compatibility...")
    compatibility = manager.check_model_compatibility(models)
    
    print("\nModel information:")
    for i, model in enumerate(models, 1):
        info = manager.get_model_info(model)
        print(f"\nTeacher {i}:")
        print(f"  Type: {info['model_type']}")
        print(f"  Parameters: {info['parameters_M']:.1f}M")
        print(f"  Vocab size: {info['vocab_size']}")
    
    print("\nAligning tokenizers...")
    primary_tokenizer = manager.align_tokenizers(tokenizers, target_vocab_size=50257)
    print(f"Primary tokenizer vocab size: {len(primary_tokenizer)}")
    
    print("\n✓ Model manager test completed successfully!")
