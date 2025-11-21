#!/usr/bin/env python3
"""
Utilities and Helper Functions for Automated LLM Distillation Engine.

Includes logging, validation, file handling, and common utilities.
"""

import os
import sys
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file
        level: Logging level
        format_str: Custom format string
    
    Returns:
        Root logger
    """
    if format_str is None:
        format_str = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger()
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make cudnn deterministic (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info(f"Random seed set to {seed}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get appropriate torch device.
    
    Args:
        prefer_cuda: Prefer CUDA if available
    
    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU device")
    
    return device


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "2h 30m 15s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_number(num: float, precision: int = 2) -> str:
    """
    Format large numbers with K/M/B suffixes.
    
    Args:
        num: Number to format
        precision: Decimal precision
    
    Returns:
        Formatted string (e.g., "1.5M", "250K")
    """
    if num >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with memory stats (in MB)
    """
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated': torch.cuda.memory_allocated() / 1024**2,
        'reserved': torch.cuda.memory_reserved() / 1024**2,
        'max_allocated': torch.cuda.max_memory_allocated() / 1024**2,
        'total': torch.cuda.get_device_properties(0).total_memory / 1024**2
    }


def print_gpu_memory():
    """Print current GPU memory usage."""
    info = get_gpu_memory_info()
    if info:
        logging.info(
            f"GPU Memory: {info['allocated']:.0f}MB allocated, "
            f"{info['reserved']:.0f}MB reserved "
            f"({info['allocated']/info['total']*100:.1f}% of {info['total']:.0f}MB total)"
        )


def create_output_directory(base_dir: str, project_name: str) -> Path:
    """
    Create timestamped output directory.
    
    Args:
        base_dir: Base directory path
        project_name: Project name
    
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{project_name}_{timestamp}"
    output_dir = Path(base_dir) / dir_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    
    return output_dir


def save_dict_to_json(data: Dict[str, Any], file_path: str):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    import json
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logging.info(f"Saved data to {file_path}")


def load_dict_from_json(file_path: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        file_path: Input file path
    
    Returns:
        Loaded dictionary
    """
    import json
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data


class Timer:
    """Simple timer for measuring code execution time."""
    
    def __init__(self, name: str = "Timer"):
        """
        Initialize timer.
        
        Args:
            name: Name for this timer
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timer (context manager)."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Stop timer (context manager)."""
        self.stop()
        elapsed = self.elapsed()
        logging.info(f"{self.name} completed in {format_time(elapsed)}")
    
    def start(self):
        """Start timing."""
        self.start_time = time.time()
    
    def stop(self):
        """Stop timing."""
        self.end_time = time.time()
    
    def elapsed(self) -> float:
        """
        Get elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def reset(self):
        """Reset timer."""
        self.start_time = None
        self.end_time = None


class ProgressTracker:
    """Track training progress with ETA estimation."""
    
    def __init__(self, total_steps: int, log_interval: int = 10):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            log_interval: Steps between progress logs
        """
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
        self.max_step_history = 100
    
    def update(self, step: Optional[int] = None):
        """
        Update progress.
        
        Args:
            step: Current step number (auto-increments if None)
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # Track step time
        current_time = time.time()
        if len(self.step_times) > 0:
            step_duration = current_time - self.step_times[-1]
            self.step_times.append(current_time)
            
            # Limit history
            if len(self.step_times) > self.max_step_history:
                self.step_times = self.step_times[-self.max_step_history:]
        else:
            self.step_times.append(current_time)
    
    def get_eta(self) -> float:
        """
        Estimate time remaining.
        
        Returns:
            Estimated seconds remaining
        """
        if self.current_step == 0 or len(self.step_times) < 2:
            return 0.0
        
        # Calculate average step time from recent history
        recent_steps = min(len(self.step_times) - 1, 50)
        avg_step_time = (self.step_times[-1] - self.step_times[-recent_steps-1]) / recent_steps
        
        remaining_steps = self.total_steps - self.current_step
        return remaining_steps * avg_step_time
    
    def get_progress(self) -> float:
        """
        Get progress percentage.
        
        Returns:
            Progress as 0-100 percentage
        """
        return (self.current_step / self.total_steps) * 100
    
    def should_log(self) -> bool:
        """Check if we should log at this step."""
        return self.current_step % self.log_interval == 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current progress statistics.
        
        Returns:
            Dictionary with progress stats
        """
        elapsed = time.time() - self.start_time
        eta = self.get_eta()
        progress = self.get_progress()
        
        return {
            'step': self.current_step,
            'total_steps': self.total_steps,
            'progress_pct': progress,
            'elapsed': elapsed,
            'eta': eta,
            'elapsed_str': format_time(elapsed),
            'eta_str': format_time(eta)
        }


def validate_model_compatibility(
    teacher_vocab_sizes: List[int],
    student_vocab_size: int
) -> List[str]:
    """
    Validate vocabulary size compatibility between teachers and student.
    
    Args:
        teacher_vocab_sizes: List of teacher vocabulary sizes
        student_vocab_size: Student vocabulary size
    
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Check if all teachers have same vocab size
    if len(set(teacher_vocab_sizes)) > 1:
        warnings.append(
            f"Teachers have different vocabulary sizes: {teacher_vocab_sizes}. "
            f"This may affect distillation quality."
        )
    
    # Check if student vocab matches teachers
    max_teacher_vocab = max(teacher_vocab_sizes)
    min_teacher_vocab = min(teacher_vocab_sizes)
    
    if student_vocab_size < min_teacher_vocab:
        warnings.append(
            f"Student vocabulary size ({student_vocab_size}) is smaller than "
            f"minimum teacher size ({min_teacher_vocab}). "
            f"Some tokens will be unmapped."
        )
    elif student_vocab_size > max_teacher_vocab:
        warnings.append(
            f"Student vocabulary size ({student_vocab_size}) is larger than "
            f"maximum teacher size ({max_teacher_vocab}). "
            f"Consider using a teacher tokenizer."
        )
    
    return warnings


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
    seconds_per_step: float = 2.0
) -> float:
    """
    Estimate total training time.
    
    Args:
        num_samples: Number of training samples
        batch_size: Batch size
        num_epochs: Number of epochs
        gradient_accumulation_steps: Gradient accumulation steps
        seconds_per_step: Estimated seconds per optimization step
    
    Returns:
        Estimated total seconds
    """
    steps_per_epoch = num_samples // (batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    return total_steps * seconds_per_step


def print_banner(text: str, char: str = "=", width: int = 80):
    """
    Print a banner with text.
    
    Args:
        text: Text to display
        char: Character for banner lines
        width: Total width of banner
    """
    logging.info(char * width)
    logging.info(text.center(width))
    logging.info(char * width)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(log_file="test.log")
    
    print_banner("Utilities Test")
    
    # Test seed
    set_seed(42)
    
    # Test device
    device = get_device()
    
    # Test formatting
    logger.info(f"Formatted number: {format_number(1234567)}")
    logger.info(f"Formatted time: {format_time(7325)}")
    
    # Test timer
    with Timer("Test operation"):
        time.sleep(1)
    
    # Test GPU memory
    if torch.cuda.is_available():
        print_gpu_memory()
    
    # Test progress tracker
    tracker = ProgressTracker(total_steps=100, log_interval=10)
    for i in range(1, 101):
        tracker.update()
        if tracker.should_log():
            stats = tracker.get_stats()
            logger.info(
                f"Step {stats['step']}/{stats['total_steps']} "
                f"({stats['progress_pct']:.1f}%) - "
                f"ETA: {stats['eta_str']}"
            )
        time.sleep(0.01)
    
    print("\n✓ Utilities test completed!")
