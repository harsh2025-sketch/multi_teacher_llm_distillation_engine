#!/usr/bin/env python3
"""
Command-Line Interface for Automated LLM Distillation Engine.

Provides easy-to-use CLI for running distillation with various options.
"""

import argparse
import sys
import logging
from pathlib import Path

from config import (
    DistillationConfig, TeacherConfig, DatasetConfig,
    StudentModelConfig, TrainingConfig, create_example_config
)
from automated_distillation import AutomatedDistillationEngine


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated Multi-Teacher LLM Distillation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file
  python cli.py --config config.yaml
  
  # Quick start with minimal args
  python cli.py \\
    --project my_distilled_llm \\
    --teachers meta-llama/Llama-3.2-3B-Instruct google/gemma-2-2b-it \\
    --dataset wikitext \\
    --output ./outputs
  
  # With custom dataset file
  python cli.py \\
    --config config.yaml \\
    --dataset-file my_data.json \\
    --dataset-text-column text
  
  # Generate example config
  python cli.py --create-example-config my_config.yaml
        """
    )
    
    # Config file (highest priority)
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML/JSON configuration file'
    )
    
    # Generate example config
    parser.add_argument(
        '--create-example-config',
        type=str,
        metavar='PATH',
        help='Create example configuration file and exit'
    )
    
    # Project settings
    parser.add_argument(
        '--project',
        type=str,
        default='distilled_llm',
        help='Project name (default: distilled_llm)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./outputs',
        help='Output directory (default: ./outputs)'
    )
    
    # Teacher models
    parser.add_argument(
        '--teachers',
        type=str,
        nargs='+',
        help='HuggingFace model IDs for teachers (space-separated)'
    )
    
    parser.add_argument(
        '--teacher-weights',
        type=float,
        nargs='+',
        help='Weights for each teacher (default: equal weights)'
    )
    
    parser.add_argument(
        '--use-8bit',
        action='store_true',
        help='Use 8-bit quantization for teachers'
    )
    
    parser.add_argument(
        '--use-4bit',
        action='store_true',
        help='Use 4-bit quantization for teachers'
    )
    
    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        help='HuggingFace dataset ID (e.g., wikitext)'
    )
    
    parser.add_argument(
        '--dataset-config',
        type=str,
        help='Dataset configuration name'
    )
    
    parser.add_argument(
        '--dataset-split',
        type=str,
        default='train',
        help='Dataset split to use (default: train)'
    )
    
    parser.add_argument(
        '--dataset-file',
        type=str,
        help='Path to local dataset file (JSON/JSONL/CSV/TXT)'
    )
    
    parser.add_argument(
        '--dataset-text-column',
        type=str,
        default='text',
        help='Column name for text data (default: text)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to use'
    )
    
    # Student model
    parser.add_argument(
        '--student-hidden-size',
        type=int,
        default=768,
        help='Student hidden size (default: 768)'
    )
    
    parser.add_argument(
        '--student-num-layers',
        type=int,
        default=12,
        help='Student number of layers (default: 12)'
    )
    
    parser.add_argument(
        '--student-num-heads',
        type=int,
        default=12,
        help='Student number of attention heads (default: 12)'
    )
    
    # Training
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size (default: 4)'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-5,
        help='Learning rate (default: 5e-5)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=4,
        help='Gradient accumulation steps (default: 4)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=2.0,
        help='Distillation temperature (default: 2.0)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.7,
        help='Weight for distillation loss (default: 0.7)'
    )
    
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use FP16 mixed precision'
    )
    
    parser.add_argument(
        '--bf16',
        action='store_true',
        help='Use BF16 mixed precision'
    )
    
    # Other
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Cache directory for models and datasets'
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> DistillationConfig:
    """Create DistillationConfig from CLI arguments."""
    
    # Teacher configs
    if args.teachers:
        teachers = []
        weights = args.teacher_weights if args.teacher_weights else [1.0] * len(args.teachers)
        
        if len(weights) != len(args.teachers):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of teachers ({len(args.teachers)})"
            )
        
        for model_id, weight in zip(args.teachers, weights):
            teachers.append(TeacherConfig(
                model_id=model_id,
                weight=weight,
                use_8bit=args.use_8bit,
                use_4bit=args.use_4bit
            ))
    else:
        raise ValueError("At least one teacher model must be specified (--teachers)")
    
    # Dataset config
    if args.dataset:
        # HuggingFace dataset
        dataset = DatasetConfig(
            source_type='huggingface',
            source_path=args.dataset,
            config_name=args.dataset_config,
            split=args.dataset_split,
            text_column=args.dataset_text_column,
            max_samples=args.max_samples
        )
    elif args.dataset_file:
        # Local file
        dataset = DatasetConfig(
            source_type='local_file',
            source_path=args.dataset_file,
            text_column=args.dataset_text_column,
            max_samples=args.max_samples
        )
    else:
        raise ValueError("Either --dataset or --dataset-file must be specified")
    
    # Student config
    student = StudentModelConfig(
        hidden_size=args.student_hidden_size,
        num_hidden_layers=args.student_num_layers,
        num_attention_heads=args.student_num_heads
    )
    
    # Training config
    training = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        temperature=args.temperature,
        alpha=args.alpha,
        use_fp16=args.fp16,
        use_bf16=args.bf16
    )
    
    # Complete config
    config = DistillationConfig(
        project_name=args.project,
        output_dir=args.output,
        teachers=teachers,
        dataset=dataset,
        student=student,
        training=training,
        seed=args.seed,
        cache_dir=args.cache_dir
    )
    
    return config


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Handle example config creation
    if args.create_example_config:
        print(f"Creating example configuration at: {args.create_example_config}")
        create_example_config(args.create_example_config)
        print("✓ Example configuration created successfully!")
        print(f"\nTo use it: python cli.py --config {args.create_example_config}")
        sys.exit(0)
    
    # Load or create configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = DistillationConfig.from_file(args.config)
    else:
        print("Creating configuration from command-line arguments...")
        config = create_config_from_args(args)
    
    # Print configuration summary
    print("\n" + "="*80)
    print("Configuration Summary")
    print("="*80)
    print(f"Project: {config.project_name}")
    print(f"Output: {config.output_dir}")
    print(f"\nTeachers ({len(config.teachers)}):")
    for i, teacher in enumerate(config.teachers, 1):
        print(f"  {i}. {teacher.model_id} (weight: {teacher.weight:.2f})")
    print(f"\nDataset:")
    print(f"  Type: {config.dataset.source_type}")
    print(f"  Path: {config.dataset.source_path}")
    print(f"\nStudent Model:")
    print(f"  Layers: {config.student.num_hidden_layers}")
    print(f"  Hidden size: {config.student.hidden_size}")
    print(f"\nTraining:")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate:.2e}")
    print("="*80 + "\n")
    
    # Confirm to proceed
    try:
        response = input("Proceed with training? [Y/n]: ").strip().lower()
        if response and response not in ['y', 'yes']:
            print("Training cancelled.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nTraining cancelled.")
        sys.exit(0)
    
    # Initialize and run engine
    print("\nInitializing distillation engine...")
    engine = AutomatedDistillationEngine(config)
    
    try:
        # Setup (load models, data, etc.)
        engine.setup()
        
        # Train
        metrics = engine.train()
        
        print("\n" + "="*80)
        print("✓ Training completed successfully!")
        print("="*80)
        print(f"Output directory: {config.output_dir}")
        print(f"Total steps: {engine.global_step}")
        print(f"Final loss: {metrics['total_loss'][-1]:.4f}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        engine.save_checkpoint()
        print("Checkpoint saved. You can resume training later.")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n\nERROR: Error during training: {e}")
        logging.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
