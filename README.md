# Automated Multi-Teacher LLM Distillation Engine

<div align="center">
  <img src="assets/svg.png" alt="Multi-Teacher LLM Distillation Architecture" width="800"/>
</div>

<div align="center">

**Production-ready, fully automated engine** for knowledge distillation from multiple large language models into compact student models.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)

</div>

---

## ✨ Key Features

**Complete Automation** • **HuggingFace Integration** • **Web UI** • **Multi-Teacher Support** • **Quantization Ready**

## New Features (Automated Engine)

 **Complete Automation**
- Automatic model downloading from HuggingFace Hub
- Support for any HuggingFace model as teacher
- Smart tokenizer alignment and vocabulary management
- Automated dataset loading and preprocessing

 **Flexible Dataset Support**
- HuggingFace datasets (30,000+ datasets available)
- User-uploaded files (JSON, JSONL, CSV, TXT, Parquet)
- Directory scanning for text files
- Automatic validation and preprocessing

 **Easy Configuration**
- YAML/JSON configuration files
- Command-line interface with full options
- Web UI for non-technical users
- Example configs for quick start

 **Advanced Features**
- 4-bit and 8-bit quantization support
- Mixed precision training (FP16/BF16)
- Progress tracking with ETA
- Automatic checkpoint management
- Resume training from checkpoints

## What is Multi-Teacher Distillation?

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/harsh2025-sketch/multi_teacher_llm_distillation_engine.git
cd multi_teacher_llm_distillation_engine

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Option 1: Web UI (Recommended for Beginners)

Launch the user-friendly web interface:

```bash
python src/web_ui.py
```

Then open your browser to `http://localhost:7860`

**Features:**
- Visual form for all configurations
- Drag-and-drop dataset upload
- Model selection from HuggingFace
- Real-time training monitoring
- Built-in help and guidance

<div align="center">
  <img src="assets/Screenshot 2025-11-20 223422.png" alt="Web UI Interface" width="700"/>
  <p><em>User-friendly Web UI for configuration and training</em></p>
</div>

### Option 2: Command Line Interface

#### Quick Start with Example Config

```bash
# Generate example configuration
python src/cli.py --create-example-config my_config.yaml

# Edit the config file, then run
python src/cli.py --config my_config.yaml
```

#### Direct CLI Arguments

```bash
python src/cli.py \
 --project "my_distilled_llm" \
 --teachers meta-llama/Llama-3.2-3B-Instruct google/gemma-2-2b-it \
 --dataset wikitext \
 --dataset-config wikitext-103-raw-v1 \
 --output ./outputs \
 --batch-size 4 \
 --num-epochs 3 \
 --use-8bit \
 --bf16
```

### Option 3: Python API

```python
from config import DistillationConfig, TeacherConfig, DatasetConfig
from automated_distillation import AutomatedDistillationEngine

# Create configuration
config = DistillationConfig(
 project_name="my_llm",
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
 max_samples=10000
 )
)

# Initialize and run
engine = AutomatedDistillationEngine(config)
engine.setup()
metrics = engine.train()
```

## Detailed Usage

### Using HuggingFace Datasets

The engine supports 30,000+ datasets from HuggingFace:

```yaml
dataset:
 source_type: "huggingface"
 source_path: "wikitext" # Dataset ID
 config_name: "wikitext-103-raw-v1" # Configuration
 split: "train"
 text_column: "text"
 max_samples: 100000
```

**Popular datasets:**
- `wikitext` - Wikipedia text
- `openwebtext` - Web content
- `HuggingFaceFW/fineweb-edu` - Educational content
- `c4` - Colossal Clean Crawled Corpus
- `bookcorpus` - Books

### Using Custom Datasets

#### JSON Format

```json
[
 {"text": "First training example..."},
 {"text": "Second training example..."},
 {"text": "Third training example..."}
]
```

```yaml
dataset:
 source_type: "local_file"
 source_path: "./my_data.json"
 file_format: "json"
 text_column: "text"
```

#### JSONL Format (Recommended for large datasets)

```jsonl
{"text": "First example..."}
{"text": "Second example..."}
{"text": "Third example..."}
```

#### CSV Format

```csv
text,label
"First example...",1
"Second example...",2
```

#### Plain Text

```
First paragraph of text.

Second paragraph of text.

Third paragraph of text.
```

### Teacher Model Selection

You can use **any** causal language model from HuggingFace:

**Small models (for testing):**
- `gpt2` (124M)
- `distilgpt2` (82M)
- `EleutherAI/pythia-160m`

**Medium models:**
- `meta-llama/Llama-3.2-3B-Instruct` (3B)
- `google/gemma-2-2b-it` (2B)
- `microsoft/phi-2` (2.7B)

**Large models:**
- `meta-llama/Llama-3.2-8B-Instruct` (8B)
- `mistralai/Mistral-7B-v0.3` (7B)
- `google/gemma-2-9b-it` (9B)

**Configure teachers:**

```yaml
teachers:
 - model_id: "meta-llama/Llama-3.2-3B-Instruct"
 weight: 0.6 # 60% weight in ensemble
 use_8bit: true # Enable 8-bit quantization
 use_4bit: false # Or use 4-bit for more compression
 trust_remote_code: true
```

### Student Model Architecture

Configure the student model size:

```yaml
student:
 vocab_size: 128256 # Match teacher vocabulary
 hidden_size: 768 # Embedding dimension
 intermediate_size: 2048 # FFN dimension
 num_hidden_layers: 12 # Number of transformer layers
 num_attention_heads: 12 # Attention heads
 max_position_embeddings: 2048 # Max sequence length
```

**Architecture presets:**

| Size | Hidden | Layers | Params | Use Case |
|------|--------|--------|---------|----------|
| Tiny | 384 | 6 | ~50M | Edge devices, testing |
| Small | 768 | 12 | ~250M | Mobile, embedded |
| Medium | 1024 | 16 | ~500M | Standard deployment |
| Large | 2048 | 24 | ~1B | High-quality applications |

### Training Configuration

```yaml
training:
 # Batch settings
 batch_size: 4 # Per-device batch size
 gradient_accumulation_steps: 4 # Effective batch = 16
 num_epochs: 3
 max_length: 512
 
 # Optimization
 learning_rate: 5.0e-5
 weight_decay: 0.01
 warmup_ratio: 0.1
 lr_scheduler_type: "cosine" # or "linear", "constant"
 
 # Distillation
 temperature: 2.0 # Softening temperature (1-5)
 alpha: 0.7 # Distillation weight (0-1)
 
 # Mixed precision
 use_bf16: true # BF16 for A100/H100
 use_fp16: false # FP16 for V100/T4
 
 # Checkpointing
 save_steps: 500
 save_total_limit: 3 # Keep last 3 checkpoints
```

## Example Configurations

The `configs/` directory contains ready-to-use configurations:

### 1. Quick Start (examples/configs/quick_start.yaml)
```bash
python src/cli.py --config examples/configs/quick_start.yaml
```
- Small models (GPT-2, DistilGPT-2)
- Limited dataset (1000 samples)
- Fast training (~10 minutes on T4)
- Perfect for testing

### 2. Production (examples/configs/production.yaml)
```bash
python src/cli.py --config examples/configs/production.yaml
```
- Modern models (Llama 3.2, Gemma 2)
- Large dataset (100K samples)
- Full training (~6-12 hours on A100)
- Production-quality results

### 3. Custom Dataset (examples/configs/custom_dataset.yaml)
```bash
python src/cli.py --config examples/configs/custom_dataset.yaml
```
- Use your own data
- Upload JSON/JSONL/CSV files
- Full control over preprocessing

## Configuration Reference

<div align="center">
  <img src="assets/Screenshot 2025-11-20 223445.png" alt="Configuration Options" width="700"/>
  <p><em>Example configuration with all available options</em></p>
</div>

### Complete YAML Structure

```yaml
# Project metadata
project_name: "my_project"
output_dir: "./outputs"
seed: 42
cache_dir: "./cache"

# Teachers
teachers:
 - model_id: "teacher_model_id"
 weight: 0.5
 use_8bit: true
 use_4bit: false
 trust_remote_code: true
 device_map: "auto"
 revision: null # Git branch/tag
 token: null # HF auth token for private models

# Dataset
dataset:
 source_type: "huggingface" # or "local_file", "directory"
 source_path: "dataset_id"
 split: "train"
 config_name: null
 text_column: "text"
 file_format: null # For local files: json, jsonl, csv, txt, parquet
 max_samples: null
 streaming: false
 cache_dir: null

# Student
student:
 vocab_size: 128256
 hidden_size: 768
 intermediate_size: 2048
 num_hidden_layers: 12
 num_attention_heads: 12
 num_key_value_heads: 12
 max_position_embeddings: 2048
 rms_norm_eps: 1.0e-5
 rope_theta: 10000.0
 attention_dropout: 0.0
 hidden_dropout: 0.0
 initializer_range: 0.02
 tie_word_embeddings: true

# Training
training:
 batch_size: 4
 num_epochs: 3
 max_length: 512
 gradient_accumulation_steps: 4
 learning_rate: 5.0e-5
 weight_decay: 0.01
 adam_beta1: 0.9
 adam_beta2: 0.95
 adam_epsilon: 1.0e-9
 max_grad_norm: 1.0
 warmup_ratio: 0.1
 lr_scheduler_type: "cosine"
 temperature: 2.0
 alpha: 0.7
 use_fp16: false
 use_bf16: true
 save_steps: 500
 save_total_limit: 3
 logging_steps: 10
 dataloader_num_workers: 4
 dataloader_pin_memory: true
```

## System Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (T4, V100)
- **RAM**: 32GB system memory
- **Storage**: 50GB free space
- **CUDA**: 11.8 or higher

### Recommended for Production
- **GPU**: A100 (40GB) or H100 (80GB)
- **RAM**: 64GB+ system memory
- **Storage**: 200GB+ SSD
- **CUDA**: 12.0+

### Memory Usage Estimates

| Configuration | GPU Memory | Training Time (100K samples) |
|---------------|------------|------------------------------|
| 2 Teachers (8-bit) + Small Student | ~14GB | ~6 hours (T4) |
| 2 Teachers (8-bit) + Medium Student | ~16GB | ~10 hours (V100) |
| 2 Teachers (4-bit) + Large Student | ~12GB | ~8 hours (T4) |
| 3 Teachers (8-bit) + Medium Student | ~20GB | ~12 hours (A100) |

## Advanced Features

### Resume Training

```bash
# Training saves checkpoints automatically
# To resume, just run the same command
python src/cli.py --config my_config.yaml
```

The engine automatically detects and resumes from the latest checkpoint.

### Custom Model Architectures

Modify `student_architecture.py` to implement custom architectures:

```python
from student_architecture import StudentArchitectureConfig, StudentLLM

# Custom configuration
config = StudentArchitectureConfig(
 hidden_size=1024,
 num_hidden_layers=16,
 # ... custom settings
)

model = StudentLLM(config)
```

### Multi-GPU Training

```bash
# Use accelerate for multi-GPU
accelerate config # Configure once

accelerate launch src/cli.py --config my_config.yaml
```

### Export to HuggingFace

```bash
# Convert checkpoint to HuggingFace format
python src/export_to_safetensors.py \
 --in_dir ./outputs/checkpoints/step_10000 \
 --release_dir ./hf_model \
 --max_shard_size 5GB
```

## Monitoring Training

<div align="center">
  <img src="assets/Screenshot 2025-11-20 223503.png" alt="Training Monitoring" width="700"/>
  <p><em>Real-time training progress and metrics</em></p>
</div>

### TensorBoard (Coming Soon)

```bash
tensorboard --logdir ./outputs/logs
```

### Web UI Monitor

The web UI includes real-time training monitoring:
- Current step and epoch
- Loss values (total, distillation, hard)
- Learning rate
- ETA to completion

### Log Files

Detailed logs are saved to:
```
outputs/
 training.log # Complete training log
 checkpoints/ # Model checkpoints
 final_model/ # Final trained model
 training_metrics.json # Metrics history
```

<div align="center">
  <img src="assets/Screenshot 2025-11-20 223518.png" alt="Training Results" width="700"/>
  <p><em>Training completion and model export</em></p>
</div>

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Enable quantization: `use_8bit: true` or `use_4bit: true`
2. Reduce batch size: `batch_size: 2`
3. Increase gradient accumulation: `gradient_accumulation_steps: 8`
4. Reduce sequence length: `max_length: 256`
5. Use smaller student model: `hidden_size: 512`

### Slow Training

**Solutions:**
1. Enable mixed precision: `use_bf16: true`
2. Increase batch size (if memory allows)
3. Use fewer teachers
4. Reduce max_length
5. Use faster dataset (not streaming)

### Model Download Issues

**Solutions:**
1. Set HF token: `export HF_TOKEN=your_token`
2. Use cache: `cache_dir: "./cache"`
3. Download manually first:
 ```python
 from transformers import AutoModelForCausalLM
 model = AutoModelForCausalLM.from_pretrained("model_id")
 ```

### Tokenizer Mismatch

**Solutions:**
1. Use same vocab_size as primary teacher
2. Let engine auto-align: `vocab_size: 128256`
3. Check tokenizer compatibility in logs

## Architecture Details

### What Makes This Different?

This is a **direct/online distillation** framework where teacher models run inference at each training step:

```
Training Step:
1. Load batch of text
2. Teacher 1 forward pass → logits_1
3. Teacher 2 forward pass → logits_2 
4. Average: teacher_logits = weighted_avg(logits_1, logits_2)
5. Student forward pass → student_logits
6. Compute loss: KL(student || teacher) + CE(student, labels)
7. Backprop and update student
```

**Advantages:**
- Flexible - easily swap teachers
- Accurate - real-time teacher guidance
- Simple - no pre-processing needed

**Trade-offs:**
- Higher memory (all models in memory)
- Slower (teacher inference at each step)

### Student Architecture

Modern **LLaMA-style** transformer:
- **RMSNorm** for normalization
- **RoPE** for positional encoding
- **SwiGLU** activation
- **Grouped-Query Attention** (optional)

Optimized for:
- Fast inference
- Efficient training
- Small model size
- HuggingFace compatibility

## Research Papers

This implementation is based on:

1. **Knowledge Distillation** (Hinton et al., 2015)
2. **DistilBERT** (Sanh et al., 2019) 
3. **Multi-Teacher Distillation** (You et al., 2017)
4. **LLaMA** (Touvron et al., 2023)

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- HuggingFace for Transformers and Datasets
- Meta AI for LLaMA architecture
- Google for Gemma models
- PyTorch team

## Comparison with Original Framework

### Original Framework (Snapshot-based)
- Pre-computes teacher logits once
- Stores to disk (~200GB per teacher)
- 4x faster training
- Best for production at scale

### This Framework (Automated Engine)
- **Teachers run live** during training
- **No pre-computation** needed
- **Automated everything** (download, setup, train)
- **Web UI** for easy use
- **Flexible dataset** sources
- Best for experimentation and automation

## What's Next?

### Planned Features
- [ ] Distributed training support
- [ ] TensorBoard integration
- [ ] Automatic hyperparameter tuning
- [ ] Model pruning and quantization
- [ ] Evaluation suite
- [ ] More student architectures
- [ ] API server deployment

### Contributing

Contributions welcome! Please see CONTRIBUTING.md (coming soon).

## Support

- **Issues**: Open a GitHub issue
- **Discussions**: GitHub Discussions
- **Documentation**: See Wiki

---

**Happy Distilling! **

```
Training Step:
1. Load batch of text
2. Forward pass through Teacher 1 → logits_1
3. Forward pass through Teacher 2 → logits_2 
4. Average: teacher_logits = (logits_1 + logits_2) / 2
5. Forward pass through Student → student_logits
6. Compute loss: KL(student || teacher) + CE(student, labels)
7. Backprop and update student
```

### Direct vs. Snapshot Distillation

| Feature | Direct (This Framework) | Snapshot (Other Framework) |
|---------|------------------------|---------------------------|
| Teacher loading | Both in memory (8-bit) | Pre-computed once |
| Teacher forward | Every training step | Never (cached) |
| Memory | High (2 teachers + student) | Low (student only) |
| Storage | Minimal | ~200GB per teacher |
| Flexibility | Easy to swap teachers | Must regenerate snapshots |
| Training speed | Slower (teacher inference) | 4x faster |
| Accuracy | Potentially more accurate | Very close to direct |
| **Best for** | **Experimentation, flexibility** | **Production, large-scale** |

## Components

### Core Files

- **`direct_train.py`** - Main training script for direct multi-teacher distillation
- **`model.py`** - Student LLM architecture (~250M parameters, LLaMA-style)
- **`losses.py`** - Distillation loss functions (KL divergence + cross-entropy)
- **`snapshot_dataset.py`** - (Legacy) Not used for direct training

### Key Classes

#### `DirectMultiTeacherDistillationLoss`
Computes distillation loss from multiple teachers:
```python
# Averages teacher logits in real-time
teacher_logits = weighted_avg([teacher1_logits, teacher2_logits])

# Soft targets (distillation)
distill_loss = KL_div(student_logits / T, teacher_logits / T) * T²

# Hard targets (cross-entropy)
hard_loss = CE(student_logits, true_labels)

# Combined
total_loss = α * distill_loss + (1-α) * hard_loss
```

#### `DirectMultiTeacherTrainer`
Complete training pipeline:
- Loads multiple teachers with 8-bit quantization
- Runs teacher forward passes every step
- Supports mixed precision (FP16/BF16)
- Gradient accumulation for memory efficiency
- Cosine annealing LR schedule
- Automatic checkpointing

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers accelerate bitsandbytes datasets tqdm
```

### 2. Prepare Teacher Models

Download teacher models to local disk:

```python
# Example: Using HuggingFace models
teacher1_path = "/path/to/Meta-Llama-3-8B"
teacher2_path = "/path/to/gemma-2-4b-it"
```

**Tip**: For faster loading on Google Colab/TPU, copy models to local disk (`/tmp/models`) instead of using Google Drive.

### 3. Train Student

```bash
python direct_train.py \\
 --teacher1_path /path/to/llama3_8b \\
 --teacher2_path /path/to/gemma3_4b \\
 --dataset_path /path/to/training_data \\
 --output_dir ./outputs \\
 --batch_size 4 \\
 --num_epochs 3 \\
 --learning_rate 5e-5 \\
 --use_8bit \\
 --fp16
```

### 4. Configuration Options

```python
from direct_train import TrainingConfig

config = TrainingConfig(
 # Paths
 teacher1_path="/path/to/teacher1",
 teacher2_path="/path/to/teacher2",
 dataset_path="/path/to/data",
 output_dir="./outputs",
 
 # Model architecture
 student_vocab_size=128256,
 student_hidden_size=768,
 student_num_layers=12,
 student_num_heads=12,
 
 # Training
 batch_size=4,
 num_epochs=3,
 gradient_accumulation_steps=4, # Effective batch = 16
 
 # Optimization
 learning_rate=5e-5,
 weight_decay=0.01,
 adam_beta1=0.9,
 adam_beta2=0.95, # Better for transformers
 max_grad_norm=1.0,
 warmup_ratio=0.1,
 
 # Distillation
 temperature=2.0,
 alpha=0.7, # 70% distillation, 30% hard labels
 teacher_weights=[0.5, 0.5], # Equal weighting
 
 # Mixed precision
 use_fp16=True, # or BF16 for A100/H100
 
 # Checkpointing
 save_steps=500,
 keep_last_n=3,
 logging_steps=10
)
```

## Example: Training on Google Colab (T4 GPU)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader
from direct_train import DirectMultiTeacherTrainer, TrainingConfig, TextDataset
import torch

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load teachers with 8-bit quantization (for T4 GPU)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

teacher1 = AutoModelForCausalLM.from_pretrained(
 "/content/drive/MyDrive/models/Meta-Llama-3-8B",
 quantization_config=quantization_config,
 device_map='auto',
 trust_remote_code=True
)

teacher2 = AutoModelForCausalLM.from_pretrained(
 "/content/drive/MyDrive/models/gemma-2-4b-it",
 quantization_config=quantization_config,
 device_map='auto',
 trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
 "/content/drive/MyDrive/models/Meta-Llama-3-8B"
)

# Create student model
from model import StudentModel, StudentConfig

student_config = StudentConfig(
 vocab_size=128256,
 hidden_size=768,
 num_hidden_layers=12
)
student_model = StudentModel(student_config)

# Load dataset
train_texts = [...] # Your training texts
train_dataset = TextDataset(train_texts, tokenizer, max_length=512)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Create trainer
config = TrainingConfig(
 teacher1_path="", # Already loaded
 teacher2_path="",
 dataset_path="",
 output_dir="/content/drive/MyDrive/outputs",
 batch_size=4,
 num_epochs=3,
 use_fp16=True
)

trainer = DirectMultiTeacherTrainer(
 config=config,
 student_model=student_model,
 teacher_models=[teacher1, teacher2],
 student_tokenizer=tokenizer,
 teacher_tokenizers=[tokenizer, tokenizer],
 train_dataloader=train_dataloader,
 device=device
)

# Train!
metrics = trainer.train()
```

## Advanced Usage

### Custom Teacher Weighting

Weight teachers differently based on their performance:

```python
config = TrainingConfig(
 ...
 teacher_weights=[0.7, 0.3], # 70% Llama, 30% Gemma
)
```

### Resume Training from Checkpoint

```python
# Load checkpoint
checkpoint = torch.load("outputs/checkpoint-step-1000.pt")

trainer.student.load_state_dict(checkpoint['model_state_dict'])
trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
trainer.global_step = checkpoint['global_step']
trainer.epoch = checkpoint['epoch']

# Continue training
trainer.train()
```

### Integrate with RSRM Framework

For crash recovery on Google Colab:

```python
from rsrm_framework import AutoRecovery, TrainingStateManager, CheckpointManager

# Setup RSRM
state_mgr = TrainingStateManager("training_state.json")
ckpt_mgr = CheckpointManager("checkpoints", keep_last_n=10)
recovery = AutoRecovery(state_mgr, ckpt_mgr)

# Training loop with auto-save
for batch in train_dataloader:
 # ... training step ...
 
 if step % 100 == 0:
 state_mgr.save_state({
 'step': step,
 'epoch': epoch,
 'loss': loss.item()
 })
 ckpt_mgr.save_checkpoint(student_model, optimizer, step)
```

## Expected Performance

### T4 GPU (15GB VRAM)
- **Batch size**: 4
- **Gradient accumulation**: 4 (effective batch = 16)
- **Teachers**: Llama 3 8B + Gemma 3 4B (8-bit)
- **Training speed**: ~2-3 steps/second
- **Memory usage**: ~13-14GB VRAM
- **Time to train**: ~12-18 hours for 3 epochs on 100k samples

### V100 GPU (16GB VRAM)
- **Batch size**: 8
- **Gradient accumulation**: 2 (effective batch = 16)
- **Training speed**: ~4-5 steps/second
- **Time to train**: ~6-10 hours

### A100 GPU (40GB VRAM)
- **Batch size**: 16
- **Gradient accumulation**: 1
- **Precision**: BF16 (native support)
- **Training speed**: ~8-10 steps/second
- **Time to train**: ~3-5 hours

## When to Use Direct vs. Snapshot Distillation

### Use Direct Multi-Teacher (This Framework) When:
- **Experimenting** with different teacher combinations
- **Prototyping** new distillation techniques
- You want **flexibility** to swap teachers easily
- Training on **smaller datasets** (<100k samples)
- You have **sufficient GPU memory** (16GB+ VRAM)
- **Real-time teacher updates** are important

### Use Snapshot Distillation (Other Framework) When:
- **Production training** at scale
- Training on **very large datasets** (>1M samples)
- **Limited GPU memory** (<16GB VRAM)
- You want **4x faster training**
- Teachers are **fixed** and won't change
- You can afford **~200GB storage** per teacher

## Example Notebook

See `multi-teacher_distillation_colab/student_llm_via_multi_teacher_distillation.ipynb` for a complete Google Colab example demonstrating:
- Loading teachers with 8-bit quantization
- Creating custom datasets
- Training with multiple teachers
- Monitoring training progress
- Saving and loading checkpoints
- Integration with FAT and RSRM frameworks

## Research Papers

This implementation is based on:

1. **Knowledge Distillation** (Hinton et al., 2015)
 - Temperature-scaled softmax for soft targets
 - α-weighted combination of distillation and hard loss

2. **Multi-Teacher Distillation**
 - Averaging multiple teacher predictions
 - Weighted teacher contributions
 - Ensemble knowledge transfer

3. **DistilBERT** (Sanh et al., 2019)
 - Training smaller models efficiently
 - Knowledge distillation for language models

## 🤝 Integration with Other Frameworks

### With RSRM Framework
```python
from rsrm_framework import TrainingStateManager, CheckpointManager
# Add crash recovery and state persistence
```

### With FAT Framework
```python
from fat_framework import FATLinear, FATTransformer
# Use feedback alignment for biologically-inspired learning
```

### With Snapshot Distillation Framework
```python
# Pre-compute teacher logits once, then switch to fast training
from snapshot_distillation_framework import SnapshotGenerator
```

## Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1**: Reduce batch size
```python
config.batch_size = 2 # Down from 4
config.gradient_accumulation_steps = 8 # Up from 4
```

**Solution 2**: Enable 8-bit quantization
```python
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
```

**Solution 3**: Use CPU offloading
```python
teacher = AutoModelForCausalLM.from_pretrained(
 path,
 device_map='auto', # Automatic CPU offloading
 offload_folder='/tmp/offload'
)
```

### Slow Training Speed

**Issue**: Each step requires 2 teacher forward passes + 1 student forward pass

**Solutions**:
1. Use snapshot distillation instead (4x faster)
2. Reduce max sequence length
3. Use gradient accumulation to reduce steps
4. Use mixed precision (FP16/BF16)
5. Copy models to local disk (not Google Drive)

### NaN Losses

**Causes**: 
- Gradient overflow in FP16
- Extreme logit values from teachers

**Solutions**:
1. Enable GradScaler for FP16
2. Lower learning rate
3. Enable gradient clipping
4. Use BF16 instead of FP16

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by the notebook: `student_llm_via_multi_teacher_distillation.ipynb`
- Teacher models: Meta Llama 3, Google Gemma 3
- Built with PyTorch, Transformers, and Accelerate
