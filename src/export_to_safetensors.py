#!/usr/bin/env python3
"""
Export Direct Multi-Teacher Distillation Checkpoints to HuggingFace Format.

This script converts FSDP-sharded PyTorch checkpoints (.pt files) from
direct multi-teacher distillation training into HuggingFace-compatible
.safetensors format.

Features:
- FSDP checkpoint loading and consolidation
- SafeTensors conversion with configurable sharding
- HuggingFace model index and config generation
- Distributed training compatibility (rank 0 coordination)
- Metadata preservation

Usage:
    # Export from FSDP checkpoint
    python export_to_safetensors.py \\
        --in_dir ./checkpoints/step_10000 \\
        --ckpt_prefix checkpoint \\
        --release_dir ./hf_model \\
        --max_shard_size 5GB

    # With custom config
    python export_to_safetensors.py \\
        --in_dir ./checkpoints/step_10000 \\
        --release_dir ./hf_model \\
        --config_path ./student_config.json

Output structure:
    hf_model/
    ├── model-00001-of-00003.safetensors
    ├── model-00002-of-00003.safetensors
    ├── model-00003-of-00003.safetensors
    ├── model.safetensors.index.json
    ├── config.json
    ├── generation_config.json (optional)
    └── tokenizer files (if provided)
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
from safetensors.torch import save_file
from tqdm import tqdm

# Try to import student architecture
try:
    from student_architecture import StudentArchitectureConfig, StudentLLM
    HAS_STUDENT_ARCH = True
except ImportError:
    HAS_STUDENT_ARCH = False
    print("Warning: Could not import student_architecture.py. Will need config file.")


def parse_size(size_str: str) -> int:
    """Parse human-readable size string to bytes."""
    size_str = size_str.strip().upper()
    
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4
    }
    
    # Extract number and unit
    match = re.match(r'(\d+(?:\.\d+)?)\s*([KMGT]?B)', size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}. Use format like '5GB' or '500MB'")
    
    number, unit = match.groups()
    return int(float(number) * units[unit])


def rank0_print(*args, **kwargs):
    """Print only from rank 0 in distributed settings."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def get_sharded_checkpoint_paths(in_dir: Path, ckpt_prefix: str = "checkpoint") -> List[Path]:
    """
    Find all FSDP-sharded checkpoint files.
    
    Looks for files like:
        checkpoint-rank-00000-of-00004.pt
        checkpoint-rank-00001-of-00004.pt
        ...
    """
    pattern = f"{ckpt_prefix}-rank-*.pt"
    checkpoint_files = sorted(in_dir.glob(pattern))
    
    if not checkpoint_files:
        # Try alternative pattern without "rank" prefix
        pattern = f"{ckpt_prefix}-*.pt"
        checkpoint_files = sorted(in_dir.glob(pattern))
    
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files found in {in_dir} with prefix '{ckpt_prefix}'. "
            f"Expected format: {ckpt_prefix}-rank-00000-of-XXXX.pt"
        )
    
    rank0_print(f"Found {len(checkpoint_files)} checkpoint shards")
    return checkpoint_files


def load_and_consolidate_sharded_checkpoints(checkpoint_paths: List[Path]) -> Dict[str, torch.Tensor]:
    """
    Load FSDP-sharded checkpoints and consolidate into single state dict.
    
    FSDP saves model parameters split across ranks. This function:
    1. Loads each shard
    2. Extracts model state_dict (ignoring optimizer state)
    3. Consolidates parameters with matching keys
    """
    rank0_print("Loading and consolidating FSDP shards...")
    
    consolidated_state = {}
    
    for ckpt_path in tqdm(checkpoint_paths, desc="Loading shards"):
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Extract model state dict (may be nested under 'model' key)
        if 'model' in checkpoint:
            shard_state = checkpoint['model']
        elif 'state_dict' in checkpoint:
            shard_state = checkpoint['state_dict']
        else:
            shard_state = checkpoint
        
        # Consolidate parameters
        for key, value in shard_state.items():
            if key in consolidated_state:
                # For FSDP, we typically concatenate along dim 0
                # But this depends on the FSDP sharding strategy
                # Most common: parameters are fully replicated or sharded along first dim
                if consolidated_state[key].shape != value.shape:
                    # Try concatenation if shapes differ
                    try:
                        consolidated_state[key] = torch.cat([consolidated_state[key], value], dim=0)
                    except RuntimeError:
                        rank0_print(f"Warning: Could not concatenate {key}, keeping first occurrence")
                else:
                    # If shapes match, take the first occurrence (parameters might be replicated)
                    pass
            else:
                consolidated_state[key] = value.clone()
    
    rank0_print(f"Consolidated {len(consolidated_state)} parameters")
    return consolidated_state


def clean_parameter_names(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Clean up parameter names from training wrappers.
    
    Removes prefixes like:
        - '_forward_module.'
        - '_fsdp_wrapped_module.'
        - 'module.'
        - 'model.'
    """
    cleaned_state = {}
    
    for key, value in state_dict.items():
        # Remove common wrapper prefixes
        clean_key = key
        prefixes_to_remove = [
            '_forward_module.',
            '_fsdp_wrapped_module.',
            'module.',
        ]
        
        for prefix in prefixes_to_remove:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        
        cleaned_state[clean_key] = value
    
    return cleaned_state


def calculate_tensor_size(tensor: torch.Tensor) -> int:
    """Calculate size of tensor in bytes."""
    return tensor.numel() * tensor.element_size()


def shard_state_dict_for_hf(
    state_dict: Dict[str, torch.Tensor],
    max_shard_size: int
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    """
    Shard state dict for HuggingFace format.
    
    Splits large models into multiple .safetensors files to stay under
    the max_shard_size limit.
    
    Returns:
        shards: List of state dict shards
        index: Weight map for model.safetensors.index.json
    """
    rank0_print(f"Sharding model (max shard size: {max_shard_size / 1e9:.2f}GB)...")
    
    shards = []
    current_shard = {}
    current_shard_size = 0
    weight_map = {}
    
    # Sort parameters by size (largest first) for better packing
    sorted_params = sorted(
        state_dict.items(),
        key=lambda x: calculate_tensor_size(x[1]),
        reverse=True
    )
    
    for param_name, param_tensor in sorted_params:
        param_size = calculate_tensor_size(param_tensor)
        
        # Check if we need a new shard
        if current_shard and (current_shard_size + param_size > max_shard_size):
            shards.append(current_shard)
            current_shard = {}
            current_shard_size = 0
        
        # Add parameter to current shard
        current_shard[param_name] = param_tensor
        current_shard_size += param_size
        
        # Track which shard contains this parameter
        shard_idx = len(shards)
        weight_map[param_name] = f"model-{shard_idx+1:05d}-of-XXXXX.safetensors"
    
    # Add final shard
    if current_shard:
        shards.append(current_shard)
    
    # Update shard filenames with total count
    num_shards = len(shards)
    for param_name in weight_map:
        weight_map[param_name] = weight_map[param_name].replace("XXXXX", f"{num_shards:05d}")
    
    rank0_print(f"Model sharded into {num_shards} files")
    
    # Create index structure
    index = {
        "metadata": {
            "total_size": sum(calculate_tensor_size(t) for t in state_dict.values())
        },
        "weight_map": weight_map
    }
    
    return shards, index


def create_model_config(
    state_dict: Dict[str, torch.Tensor],
    config_override: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create HuggingFace-compatible config.json.
    
    Infers architecture from state dict if StudentArchitectureConfig is available,
    otherwise uses provided config or defaults.
    """
    # Start with defaults
    config = {
        "architectures": ["StudentLLM"],
        "model_type": "student_llm",
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0",
    }
    
    # Try to infer from state dict
    if "embed_tokens.weight" in state_dict:
        vocab_size, hidden_size = state_dict["embed_tokens.weight"].shape
        config["vocab_size"] = vocab_size
        config["hidden_size"] = hidden_size
    
    # Count layers
    layer_keys = [k for k in state_dict.keys() if k.startswith("layers.")]
    if layer_keys:
        num_layers = max(int(k.split(".")[1]) for k in layer_keys) + 1
        config["num_hidden_layers"] = num_layers
    
    # Infer attention heads from attention projection
    for key in state_dict.keys():
        if "self_attn.q_proj.weight" in key:
            q_proj_shape = state_dict[key].shape
            if "hidden_size" in config:
                num_heads = q_proj_shape[0] // (config["hidden_size"] // 12)  # Assuming head_dim = hidden_size / num_heads
                config["num_attention_heads"] = num_heads
            break
    
    # Apply overrides from config file
    if config_override:
        config.update(config_override)
    
    # Add some common defaults if not present
    config.setdefault("max_position_embeddings", 2048)
    config.setdefault("rms_norm_eps", 1e-5)
    config.setdefault("rope_theta", 10000.0)
    config.setdefault("tie_word_embeddings", True)
    config.setdefault("bos_token_id", 128000)
    config.setdefault("eos_token_id", 128001)
    
    return config


def export_to_safetensors(
    in_dir: Path,
    release_dir: Path,
    ckpt_prefix: str = "checkpoint",
    max_shard_size: int = 5 * 1024**3,  # 5GB default
    config_path: Optional[Path] = None
):
    """
    Main export function.
    
    Args:
        in_dir: Directory containing FSDP checkpoint shards
        release_dir: Output directory for HuggingFace model
        ckpt_prefix: Prefix for checkpoint files
        max_shard_size: Maximum size per .safetensors file in bytes
        config_path: Optional path to config.json override
    """
    rank0_print("=" * 80)
    rank0_print("Exporting Direct Multi-Teacher Distillation Checkpoint to HuggingFace Format")
    rank0_print("=" * 80)
    
    # Create output directory
    release_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Find checkpoint files
    checkpoint_paths = get_sharded_checkpoint_paths(in_dir, ckpt_prefix)
    
    # Step 2: Load and consolidate
    state_dict = load_and_consolidate_sharded_checkpoints(checkpoint_paths)
    
    # Step 3: Clean parameter names
    state_dict = clean_parameter_names(state_dict)
    
    # Step 4: Convert to bfloat16 for efficiency
    rank0_print("Converting to bfloat16...")
    for key in state_dict:
        if state_dict[key].dtype in [torch.float32, torch.float16]:
            state_dict[key] = state_dict[key].to(torch.bfloat16)
    
    # Step 5: Load config override if provided
    config_override = None
    if config_path and config_path.exists():
        rank0_print(f"Loading config from {config_path}")
        with open(config_path) as f:
            config_override = json.load(f)
    
    # Step 6: Create model config
    model_config = create_model_config(state_dict, config_override)
    
    # Save config.json
    config_file = release_dir / "config.json"
    rank0_print(f"Saving config to {config_file}")
    with open(config_file, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Step 7: Shard state dict
    shards, index = shard_state_dict_for_hf(state_dict, max_shard_size)
    
    # Step 8: Save safetensors files
    rank0_print("Saving safetensors files...")
    num_shards = len(shards)
    
    for i, shard in enumerate(tqdm(shards, desc="Writing shards")):
        shard_filename = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"
        shard_path = release_dir / shard_filename
        
        # Convert to CPU and contiguous for safetensors
        shard_cpu = {k: v.cpu().contiguous() for k, v in shard.items()}
        
        # Save with metadata
        metadata = {
            "format": "pt",
            "framework": "pytorch",
            "shard": f"{i+1}/{num_shards}"
        }
        save_file(shard_cpu, str(shard_path), metadata=metadata)
        
        rank0_print(f"  Saved {shard_filename} ({os.path.getsize(shard_path) / 1e9:.2f}GB)")
    
    # Step 9: Save index
    index_file = release_dir / "model.safetensors.index.json"
    rank0_print(f"Saving index to {index_file}")
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    # Step 10: Create generation config (optional)
    generation_config = {
        "bos_token_id": model_config.get("bos_token_id", 128000),
        "eos_token_id": model_config.get("eos_token_id", 128001),
        "max_length": model_config.get("max_position_embeddings", 2048),
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True
    }
    
    generation_config_file = release_dir / "generation_config.json"
    with open(generation_config_file, 'w') as f:
        json.dump(generation_config, f, indent=2)
    
    # Summary
    rank0_print("\n" + "=" * 80)
    rank0_print("Export completed successfully!")
    rank0_print("=" * 80)
    rank0_print(f"Output directory: {release_dir}")
    rank0_print(f"Files created:")
    rank0_print(f"  - config.json")
    rank0_print(f"  - generation_config.json")
    rank0_print(f"  - model.safetensors.index.json")
    rank0_print(f"  - {num_shards} model shard(s)")
    
    total_size = sum(os.path.getsize(release_dir / f"model-{i+1:05d}-of-{num_shards:05d}.safetensors") 
                     for i in range(num_shards))
    rank0_print(f"\nTotal model size: {total_size / 1e9:.2f}GB")
    rank0_print(f"Number of parameters: {sum(t.numel() for t in state_dict.values()) / 1e6:.1f}M")
    
    rank0_print("\nTo load this model with HuggingFace:")
    rank0_print(f"  from transformers import AutoModelForCausalLM")
    rank0_print(f"  model = AutoModelForCausalLM.from_pretrained('{release_dir}')")
    rank0_print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Export Direct Multi-Teacher Distillation checkpoints to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python export_to_safetensors.py --in_dir ./checkpoints/step_10000 --release_dir ./hf_model
  
  # With custom shard size
  python export_to_safetensors.py --in_dir ./ckpts --release_dir ./model --max_shard_size 2GB
  
  # With config override
  python export_to_safetensors.py --in_dir ./ckpts --release_dir ./model --config_path ./config.json
        """
    )
    
    parser.add_argument(
        '--in_dir',
        type=Path,
        required=True,
        help='Directory containing FSDP checkpoint shards'
    )
    
    parser.add_argument(
        '--ckpt_prefix',
        type=str,
        default='checkpoint',
        help='Prefix for checkpoint files (default: checkpoint)'
    )
    
    parser.add_argument(
        '--release_dir',
        type=Path,
        required=True,
        help='Output directory for HuggingFace model'
    )
    
    parser.add_argument(
        '--max_shard_size',
        type=str,
        default='5GB',
        help='Maximum size per shard (e.g., 5GB, 500MB). Default: 5GB'
    )
    
    parser.add_argument(
        '--config_path',
        type=Path,
        help='Optional path to config.json override'
    )
    
    args = parser.parse_args()
    
    # Parse shard size
    max_shard_size = parse_size(args.max_shard_size)
    
    # Validate input directory
    if not args.in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.in_dir}")
    
    # Run export
    export_to_safetensors(
        in_dir=args.in_dir,
        release_dir=args.release_dir,
        ckpt_prefix=args.ckpt_prefix,
        max_shard_size=max_shard_size,
        config_path=args.config_path
    )


if __name__ == "__main__":
    main()
