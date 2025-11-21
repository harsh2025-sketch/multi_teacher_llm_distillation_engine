#!/usr/bin/env python3
"""
Configuration Management System for Automated LLM Distillation Engine.

Supports YAML/JSON configuration files with validation, defaults, and type checking.
Handles teacher models, datasets (HuggingFace or user-uploaded), training parameters.
"""

import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TeacherConfig:
    """Configuration for a single teacher model."""
    
    # HuggingFace model ID or local path
    model_id: str
    
    # Weight for this teacher in ensemble (default: equal weighting)
    weight: float = 1.0
    
    # Use 8-bit quantization to save memory
    use_8bit: bool = True
    
    # Use 4-bit quantization (more aggressive)
    use_4bit: bool = False
    
    # Trust remote code (needed for some models)
    trust_remote_code: bool = True
    
    # Device map for model placement
    device_map: str = "auto"
    
    # Optional: Use specific revision/branch
    revision: Optional[str] = None
    
    # Optional: Authentication token for private models
    token: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.weight <= 0:
            raise ValueError(f"Teacher weight must be positive, got {self.weight}")
        if self.use_4bit and self.use_8bit:
            logger.warning("Both 4-bit and 8-bit quantization enabled. Using 4-bit.")
            self.use_8bit = False


@dataclass
class DatasetConfig:
    """Configuration for dataset source."""
    
    # Dataset source type: 'huggingface', 'local_file', 'directory'
    source_type: str
    
    # For HuggingFace: dataset ID (e.g., 'wikitext', 'openwebtext')
    # For local: path to file or directory
    source_path: str
    
    # For HuggingFace datasets: which split to use
    split: str = "train"
    
    # For HuggingFace datasets: which configuration/subset
    config_name: Optional[str] = None
    
    # Text column name (for datasets with multiple columns)
    text_column: str = "text"
    
    # For local files: file format ('json', 'jsonl', 'csv', 'txt', 'parquet')
    file_format: Optional[str] = None
    
    # Maximum number of samples (None = use all)
    max_samples: Optional[int] = None
    
    # Streaming mode (for very large datasets)
    streaming: bool = False
    
    # Cache directory for downloaded datasets
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        valid_sources = ['huggingface', 'local_file', 'directory']
        if self.source_type not in valid_sources:
            raise ValueError(
                f"source_type must be one of {valid_sources}, got '{self.source_type}'"
            )
        
        if self.source_type == 'local_file' and self.file_format is None:
            # Try to infer from extension
            ext = Path(self.source_path).suffix.lower()
            format_map = {
                '.json': 'json',
                '.jsonl': 'jsonl',
                '.csv': 'csv',
                '.txt': 'txt',
                '.parquet': 'parquet'
            }
            self.file_format = format_map.get(ext)
            if self.file_format is None:
                raise ValueError(
                    f"Could not infer file_format from '{self.source_path}'. "
                    f"Please specify explicitly."
                )


@dataclass
class StudentModelConfig:
    """Configuration for student model architecture."""
    
    vocab_size: int = 128256
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None  # For grouped-query attention
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    
    # Training
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    
    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-9
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    
    # Learning rate schedule
    lr_scheduler_type: str = "cosine"  # 'cosine', 'linear', 'constant'
    
    # Distillation
    temperature: float = 2.0
    alpha: float = 0.7  # Weight for distillation loss vs hard loss
    
    # Mixed precision
    use_fp16: bool = False
    use_bf16: bool = True  # Better for A100/H100
    
    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3  # Keep only last N checkpoints
    
    # Logging
    logging_steps: int = 10
    logging_dir: Optional[str] = None
    
    # Evaluation
    eval_steps: Optional[int] = None
    eval_strategy: str = "no"  # 'no', 'steps', 'epoch'
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0
    
    # Distributed training
    local_rank: int = -1
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Hardware
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.use_fp16 and self.use_bf16:
            logger.warning("Both FP16 and BF16 enabled. Using BF16.")
            self.use_fp16 = False
        
        valid_schedulers = ['cosine', 'linear', 'constant']
        if self.lr_scheduler_type not in valid_schedulers:
            raise ValueError(
                f"lr_scheduler_type must be one of {valid_schedulers}, "
                f"got '{self.lr_scheduler_type}'"
            )


@dataclass
class DistillationConfig:
    """Complete configuration for automated distillation engine."""
    
    # Project metadata
    project_name: str
    output_dir: str
    
    # Teacher models (list of configs)
    teachers: List[TeacherConfig] = field(default_factory=list)
    
    # Dataset configuration
    dataset: Optional[DatasetConfig] = None
    
    # Student model architecture
    student: StudentModelConfig = field(default_factory=StudentModelConfig)
    
    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Cache directory for models and datasets
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate complete configuration."""
        if not self.teachers:
            raise ValueError("At least one teacher model must be specified")
        
        if self.dataset is None:
            raise ValueError("Dataset configuration is required")
        
        # Normalize teacher weights to sum to 1.0
        total_weight = sum(t.weight for t in self.teachers)
        for teacher in self.teachers:
            teacher.weight /= total_weight
        
        logger.info(f"Normalized teacher weights: {[t.weight for t in self.teachers]}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration as JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved configuration to {path}")
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration as YAML."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DistillationConfig':
        """Load configuration from dictionary."""
        # Parse teachers
        teachers = [
            TeacherConfig(**t) if isinstance(t, dict) else t
            for t in config_dict.get('teachers', [])
        ]
        
        # Parse dataset
        dataset_dict = config_dict.get('dataset')
        dataset = DatasetConfig(**dataset_dict) if dataset_dict else None
        
        # Parse student
        student_dict = config_dict.get('student', {})
        student = StudentModelConfig(**student_dict)
        
        # Parse training
        training_dict = config_dict.get('training', {})
        training = TrainingConfig(**training_dict)
        
        return cls(
            project_name=config_dict['project_name'],
            output_dir=config_dict['output_dir'],
            teachers=teachers,
            dataset=dataset,
            student=student,
            training=training,
            seed=config_dict.get('seed', 42),
            cache_dir=config_dict.get('cache_dir')
        )
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'DistillationConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'DistillationConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'DistillationConfig':
        """Auto-detect file format and load configuration."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        ext = path.suffix.lower()
        if ext in ['.yaml', '.yml']:
            return cls.from_yaml(path)
        elif ext == '.json':
            return cls.from_json(path)
        else:
            raise ValueError(
                f"Unsupported configuration file format: {ext}. "
                f"Use .yaml, .yml, or .json"
            )


def create_example_config(output_path: Union[str, Path] = "config_example.yaml") -> DistillationConfig:
    """Create an example configuration file."""
    
    config = DistillationConfig(
        project_name="my_distilled_llm",
        output_dir="./outputs",
        teachers=[
            TeacherConfig(
                model_id="meta-llama/Llama-3.2-3B-Instruct",
                weight=0.6,
                use_8bit=True
            ),
            TeacherConfig(
                model_id="google/gemma-2-2b-it",
                weight=0.4,
                use_8bit=True
            )
        ],
        dataset=DatasetConfig(
            source_type="huggingface",
            source_path="wikitext",
            config_name="wikitext-103-raw-v1",
            split="train",
            text_column="text",
            max_samples=10000
        ),
        student=StudentModelConfig(
            vocab_size=128256,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12
        ),
        training=TrainingConfig(
            batch_size=4,
            num_epochs=3,
            learning_rate=5e-5,
            max_length=512,
            use_bf16=True,
            save_steps=500
        ),
        seed=42,
        cache_dir="./cache"
    )
    
    # Save example
    config.to_yaml(output_path)
    logger.info(f"Created example configuration at {output_path}")
    
    return config


def validate_config(config: DistillationConfig) -> List[str]:
    """
    Validate configuration and return list of warnings/suggestions.
    
    Returns:
        List of validation messages (empty if all good)
    """
    messages = []
    
    # Check teacher models
    if len(config.teachers) > 4:
        messages.append(
            f"Warning: {len(config.teachers)} teachers specified. "
            f"Using many teachers may slow training significantly."
        )
    
    # Check memory requirements
    if config.training.batch_size > 8:
        messages.append(
            f"Warning: Large batch size ({config.training.batch_size}). "
            f"May cause OOM errors. Consider gradient accumulation instead."
        )
    
    # Check mixed precision
    if not config.training.use_bf16 and not config.training.use_fp16:
        messages.append(
            "Suggestion: Enable mixed precision (bf16 or fp16) for faster training."
        )
    
    # Check sequence length
    if config.training.max_length > 2048:
        messages.append(
            f"Warning: Long sequence length ({config.training.max_length}). "
            f"This will increase memory usage significantly."
        )
    
    # Check student model size
    estimated_params = (
        config.student.vocab_size * config.student.hidden_size + 
        config.student.num_hidden_layers * (
            4 * config.student.hidden_size ** 2 +  # Attention
            3 * config.student.hidden_size * config.student.intermediate_size  # FFN
        )
    ) / 1e6
    
    if estimated_params > 1000:
        messages.append(
            f"Warning: Large student model (~{estimated_params:.0f}M parameters). "
            f"Consider reducing hidden_size or num_layers."
        )
    
    return messages


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    print("Creating example configuration...")
    config = create_example_config("config_example.yaml")
    
    print("\n" + "="*80)
    print("Example Configuration:")
    print("="*80)
    print(f"Project: {config.project_name}")
    print(f"Teachers: {len(config.teachers)}")
    for i, teacher in enumerate(config.teachers, 1):
        print(f"  {i}. {teacher.model_id} (weight: {teacher.weight:.2f})")
    print(f"Dataset: {config.dataset.source_path} ({config.dataset.source_type})")
    print(f"Student: {config.student.num_hidden_layers} layers, {config.student.hidden_size}d")
    print(f"Training: {config.training.num_epochs} epochs, batch_size={config.training.batch_size}")
    
    print("\nValidating configuration...")
    validation_messages = validate_config(config)
    if validation_messages:
        print("\nValidation Messages:")
        for msg in validation_messages:
            print(f"  • {msg}")
    else:
        print("  ✓ Configuration is valid!")
    
    print("\n" + "="*80)
    print("Example configuration saved to: config_example.yaml")
    print("="*80)
