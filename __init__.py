"""
Automated Multi-Teacher LLM Distillation Engine.

A production-ready, fully automated framework for knowledge distillation
from multiple teacher language models into compact student models.

Features:
- Automatic model downloading from HuggingFace Hub
- Support for custom datasets (HF, JSON, CSV, TXT, etc.)
- Configuration via YAML/JSON files
- Command-line interface and Web UI
- Progress tracking and checkpoint management
- 4-bit/8-bit quantization support
- Mixed precision training
"""

__version__ = "2.0.0"
__author__ = "LLM Distillation Project"

# Configuration system
from .config import (
    DistillationConfig,
    TeacherConfig,
    DatasetConfig,
    StudentModelConfig,
    TrainingConfig,
    create_example_config,
    validate_config
)

# Model management
from .model_manager import ModelManager

# Dataset management
from .dataset_manager import DatasetManager, TextDataset

# Student architectures
from .model import (
    StudentModel,
    StudentConfig,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    Attention,
    TransformerBlock,
    create_student_model,
)

from .student_architecture import (
    StudentLLM,
    StudentArchitectureConfig,
    GroupedQueryAttention,
    TransformerBlock as AdvancedTransformerBlock
)

# Training engine
from .automated_distillation import (
    AutomatedDistillationEngine,
    MultiTeacherDistillationLoss
)

# Loss functions
from .losses import (
    DistillationLoss,
    create_distillation_loss,
)

# Utilities
from .utils import (
    setup_logging,
    set_seed,
    get_device,
    format_time,
    format_number,
    Timer,
    ProgressTracker
)

__all__ = [
    # Version
    '__version__',
    
    # Configuration
    'DistillationConfig',
    'TeacherConfig',
    'DatasetConfig',
    'StudentModelConfig',
    'TrainingConfig',
    'create_example_config',
    'validate_config',
    
    # Managers
    'ModelManager',
    'DatasetManager',
    'TextDataset',
    
    # Student Models
    'StudentModel',
    'StudentConfig',
    'StudentLLM',
    'StudentArchitectureConfig',
    'create_student_model',
    
    # Architecture Components
    'RMSNorm',
    'RotaryEmbedding',
    'SwiGLU',
    'Attention',
    'GroupedQueryAttention',
    'TransformerBlock',
    'AdvancedTransformerBlock',
    
    # Training
    'AutomatedDistillationEngine',
    'MultiTeacherDistillationLoss',
    
    # Loss Functions
    'DistillationLoss',
    'create_distillation_loss',
    
    # Utilities
    'setup_logging',
    'set_seed',
    'get_device',
    'format_time',
    'format_number',
    'Timer',
    'ProgressTracker',
]

