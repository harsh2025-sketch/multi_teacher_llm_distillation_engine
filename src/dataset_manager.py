#!/usr/bin/env python3
"""
Dataset Manager for Automated LLM Distillation Engine.

Handles loading datasets from multiple sources:
- HuggingFace Hub datasets
- Local files (JSON, JSONL, CSV, TXT, Parquet)
- User-uploaded files
- Directories of text files

Includes validation, preprocessing, and tokenization.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Union, Iterator, Any
from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from transformers import PreTrainedTokenizer

from config import DatasetConfig

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """PyTorch Dataset for text data with tokenization."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        add_special_tokens: bool = True
    ):
        """
        Initialize text dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            add_special_tokens: Whether to add special tokens (BOS, EOS)
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        logger.info(f"TextDataset initialized with {len(texts)} samples")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized sample."""
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=self.add_special_tokens
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


class DatasetManager:
    """
    Manages dataset loading and preprocessing for distillation.
    
    Features:
    - Load from HuggingFace Hub
    - Load local files (JSON, JSONL, CSV, TXT, Parquet)
    - Handle directories of text files
    - Validation and preprocessing
    - Sample limiting and filtering
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize dataset manager.
        
        Args:
            cache_dir: Directory for caching downloaded datasets
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/datasets")
        
        logger.info(f"DatasetManager initialized")
        logger.info(f"  Cache directory: {self.cache_dir}")
    
    def load_dataset(
        self,
        dataset_config: DatasetConfig
    ) -> List[str]:
        """
        Load dataset from any supported source.
        
        Args:
            dataset_config: Dataset configuration
        
        Returns:
            List of text strings
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Loading dataset")
        logger.info(f"{'='*80}")
        logger.info(f"Source type: {dataset_config.source_type}")
        logger.info(f"Source path: {dataset_config.source_path}")
        
        # Route to appropriate loader
        if dataset_config.source_type == 'huggingface':
            texts = self._load_from_huggingface(dataset_config)
        elif dataset_config.source_type == 'local_file':
            texts = self._load_from_local_file(dataset_config)
        elif dataset_config.source_type == 'directory':
            texts = self._load_from_directory(dataset_config)
        else:
            raise ValueError(f"Unknown source_type: {dataset_config.source_type}")
        
        # Apply sample limit
        if dataset_config.max_samples is not None and len(texts) > dataset_config.max_samples:
            logger.info(f"Limiting dataset to {dataset_config.max_samples} samples (from {len(texts)})")
            texts = texts[:dataset_config.max_samples]
        
        logger.info(f"{'='*80}")
        logger.info(f"Dataset loaded: {len(texts)} samples")
        logger.info(f"{'='*80}\n")
        
        return texts
    
    def _load_from_huggingface(
        self,
        config: DatasetConfig
    ) -> List[str]:
        """Load dataset from HuggingFace Hub."""
        logger.info(f"Loading from HuggingFace Hub: {config.source_path}")
        
        try:
            # Load dataset
            dataset = load_dataset(
                config.source_path,
                name=config.config_name,
                split=config.split,
                streaming=config.streaming,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"  ✓ Dataset loaded")
            
            # Extract text column
            if config.streaming:
                # For streaming datasets, convert to list
                texts = []
                for i, example in enumerate(dataset):
                    if config.max_samples and i >= config.max_samples:
                        break
                    texts.append(example[config.text_column])
                logger.info(f"  Collected {len(texts)} samples from stream")
            else:
                # For regular datasets
                if config.text_column not in dataset.column_names:
                    raise ValueError(
                        f"Text column '{config.text_column}' not found. "
                        f"Available columns: {dataset.column_names}"
                    )
                
                texts = dataset[config.text_column]
                logger.info(f"  Extracted {len(texts)} texts from column '{config.text_column}'")
            
            # Filter empty texts
            original_count = len(texts)
            texts = [t for t in texts if t and t.strip()]
            if len(texts) < original_count:
                logger.warning(f"  Filtered out {original_count - len(texts)} empty texts")
            
            return texts
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset: {e}")
            raise RuntimeError(f"Could not load dataset from HuggingFace: {e}")
    
    def _load_from_local_file(
        self,
        config: DatasetConfig
    ) -> List[str]:
        """Load dataset from local file."""
        file_path = Path(config.source_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        logger.info(f"Loading from local file: {file_path}")
        logger.info(f"  Format: {config.file_format}")
        
        try:
            if config.file_format == 'json':
                texts = self._load_json(file_path, config.text_column)
            elif config.file_format == 'jsonl':
                texts = self._load_jsonl(file_path, config.text_column)
            elif config.file_format == 'csv':
                texts = self._load_csv(file_path, config.text_column)
            elif config.file_format == 'txt':
                texts = self._load_txt(file_path)
            elif config.file_format == 'parquet':
                texts = self._load_parquet(file_path, config.text_column)
            else:
                raise ValueError(f"Unsupported file format: {config.file_format}")
            
            logger.info(f"  ✓ Loaded {len(texts)} texts")
            return texts
            
        except Exception as e:
            logger.error(f"Failed to load local file: {e}")
            raise RuntimeError(f"Could not load dataset from file: {e}")
    
    def _load_json(self, file_path: Path, text_column: str) -> List[str]:
        """Load from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of objects
            if isinstance(data[0], dict):
                if text_column not in data[0]:
                    raise ValueError(f"Column '{text_column}' not found in JSON objects")
                return [item[text_column] for item in data if text_column in item]
            else:
                # List of strings
                return [str(item) for item in data]
        elif isinstance(data, dict):
            # Single object or dict of lists
            if text_column in data:
                return data[text_column] if isinstance(data[text_column], list) else [data[text_column]]
            else:
                raise ValueError(f"Column '{text_column}' not found in JSON")
        else:
            raise ValueError(f"Unsupported JSON structure")
    
    def _load_jsonl(self, file_path: Path, text_column: str) -> List[str]:
        """Load from JSONL file (JSON Lines)."""
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        if text_column in obj:
                            texts.append(obj[text_column])
                    elif isinstance(obj, str):
                        texts.append(obj)
                except json.JSONDecodeError as e:
                    logger.warning(f"  Skipping invalid JSON on line {line_num}: {e}")
        return texts
    
    def _load_csv(self, file_path: Path, text_column: str) -> List[str]:
        """Load from CSV file."""
        df = pd.read_csv(file_path)
        
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found. Available: {list(df.columns)}"
            )
        
        texts = df[text_column].dropna().astype(str).tolist()
        return texts
    
    def _load_txt(self, file_path: Path) -> List[str]:
        """Load from plain text file (one sample per line or paragraph)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to split intelligently
        # First try paragraphs (double newline)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if len(paragraphs) > 1:
            logger.info(f"  Split into {len(paragraphs)} paragraphs")
            return paragraphs
        else:
            # Fall back to lines
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            logger.info(f"  Split into {len(lines)} lines")
            return lines
    
    def _load_parquet(self, file_path: Path, text_column: str) -> List[str]:
        """Load from Parquet file."""
        df = pd.read_parquet(file_path)
        
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found. Available: {list(df.columns)}"
            )
        
        texts = df[text_column].dropna().astype(str).tolist()
        return texts
    
    def _load_from_directory(
        self,
        config: DatasetConfig
    ) -> List[str]:
        """Load all text files from a directory."""
        dir_path = Path(config.source_path)
        
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        logger.info(f"Loading from directory: {dir_path}")
        
        # Find all text files
        text_extensions = {'.txt', '.text', '.md'}
        text_files = [
            f for f in dir_path.rglob('*')
            if f.is_file() and f.suffix.lower() in text_extensions
        ]
        
        logger.info(f"  Found {len(text_files)} text files")
        
        texts = []
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
            except Exception as e:
                logger.warning(f"  Could not read {file_path.name}: {e}")
        
        logger.info(f"  Loaded {len(texts)} documents")
        return texts
    
    def validate_dataset(
        self,
        texts: List[str],
        min_length: int = 10,
        max_length: Optional[int] = None
    ) -> List[str]:
        """
        Validate and filter dataset.
        
        Args:
            texts: List of texts
            min_length: Minimum character length
            max_length: Maximum character length (None = no limit)
        
        Returns:
            Filtered list of texts
        """
        logger.info("Validating dataset...")
        
        original_count = len(texts)
        
        # Filter by length
        valid_texts = []
        for text in texts:
            text_len = len(text)
            if text_len < min_length:
                continue
            if max_length is not None and text_len > max_length:
                continue
            valid_texts.append(text)
        
        removed = original_count - len(valid_texts)
        if removed > 0:
            logger.info(f"  Filtered out {removed} texts (length constraints)")
        
        logger.info(f"  Valid texts: {len(valid_texts)}")
        
        # Statistics
        lengths = [len(t) for t in valid_texts]
        if lengths:
            logger.info(f"  Length statistics:")
            logger.info(f"    Min: {min(lengths)} chars")
            logger.info(f"    Max: {max(lengths)} chars")
            logger.info(f"    Mean: {sum(lengths)/len(lengths):.0f} chars")
        
        return valid_texts
    
    def create_dataloader(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int = 512,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        """
        Create PyTorch DataLoader for training.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer for encoding
            batch_size: Batch size
            max_length: Maximum sequence length
            shuffle: Whether to shuffle
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
        
        Returns:
            DataLoader instance
        """
        logger.info("Creating DataLoader...")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Max length: {max_length}")
        logger.info(f"  Shuffle: {shuffle}")
        logger.info(f"  Workers: {num_workers}")
        
        dataset = TextDataset(texts, tokenizer, max_length)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # Drop incomplete batches
        )
        
        logger.info(f"  Total batches: {len(dataloader)}")
        
        return dataloader
    
    def get_dataset_stats(self, texts: List[str]) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        if not texts:
            return {'num_samples': 0}
        
        lengths = [len(t) for t in texts]
        word_counts = [len(t.split()) for t in texts]
        
        return {
            'num_samples': len(texts),
            'total_chars': sum(lengths),
            'total_words': sum(word_counts),
            'avg_chars': sum(lengths) / len(lengths),
            'avg_words': sum(word_counts) / len(word_counts),
            'min_chars': min(lengths),
            'max_chars': max(lengths),
            'min_words': min(word_counts),
            'max_words': max(word_counts)
        }


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    from config import DatasetConfig
    
    # Create dataset manager
    manager = DatasetManager(cache_dir="./cache")
    
    # Example 1: Load from HuggingFace
    print("\n" + "="*80)
    print("Example 1: HuggingFace Dataset")
    print("="*80)
    
    hf_config = DatasetConfig(
        source_type="huggingface",
        source_path="wikitext",
        config_name="wikitext-2-raw-v1",
        split="train",
        text_column="text",
        max_samples=100
    )
    
    texts_hf = manager.load_dataset(hf_config)
    stats_hf = manager.get_dataset_stats(texts_hf)
    
    print(f"\nDataset statistics:")
    print(f"  Samples: {stats_hf['num_samples']}")
    print(f"  Avg length: {stats_hf['avg_chars']:.0f} chars ({stats_hf['avg_words']:.0f} words)")
    
    # Example 2: Load from local file
    print("\n" + "="*80)
    print("Example 2: Local JSON File")
    print("="*80)
    
    # Create a test JSON file
    test_data = [
        {"text": "This is the first example sentence."},
        {"text": "Here is another example for testing."},
        {"text": "And a third one to make it interesting."}
    ]
    
    test_file = Path("test_dataset.json")
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    json_config = DatasetConfig(
        source_type="local_file",
        source_path=str(test_file),
        file_format="json",
        text_column="text"
    )
    
    texts_json = manager.load_dataset(json_config)
    print(f"\nLoaded {len(texts_json)} samples from JSON:")
    for i, text in enumerate(texts_json, 1):
        print(f"  {i}. {text}")
    
    # Cleanup
    test_file.unlink()
    
    print("\n✓ Dataset manager test completed successfully!")
