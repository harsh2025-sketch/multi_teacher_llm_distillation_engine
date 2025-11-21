#!/usr/bin/env python3
"""
Web UI for Automated LLM Distillation Engine using Gradio.

Provides user-friendly interface for:
- Model selection
- Dataset upload/selection
- Training configuration
- Progress monitoring
"""

import gradio as gr
import json
import yaml
from pathlib import Path
from typing import Optional, List, Tuple
import threading
import logging

from config import (
    DistillationConfig, TeacherConfig, DatasetConfig,
    StudentModelConfig, TrainingConfig
)
from automated_distillation import AutomatedDistillationEngine

# Global state
training_engine = None
training_thread = None
training_in_progress = False

logger = logging.getLogger(__name__)


def create_config_from_ui(
    project_name: str,
    teacher_models: str,
    teacher_weights: str,
    use_quantization: str,
    dataset_source: str,
    dataset_path: str,
    dataset_file,
    text_column: str,
    max_samples: Optional[int],
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    max_length: int,
    temperature: float,
    alpha: float,
    mixed_precision: str,
    output_dir: str
) -> Tuple[DistillationConfig, str]:
    """Create configuration from UI inputs."""
    
    try:
        # Parse teachers
        teacher_list = [t.strip() for t in teacher_models.split(',') if t.strip()]
        if not teacher_list:
            return None, "ERROR: At least one teacher model required"
        
        # Parse weights
        if teacher_weights.strip():
            weights = [float(w.strip()) for w in teacher_weights.split(',')]
            if len(weights) != len(teacher_list):
                return None, f"ERROR: Number of weights ({len(weights)}) must match teachers ({len(teacher_list)})"
        else:
            weights = [1.0] * len(teacher_list)
        
        # Create teacher configs
        use_8bit = '8-bit' in use_quantization
        use_4bit = '4-bit' in use_quantization
        
        teachers = [
            TeacherConfig(
                model_id=model_id,
                weight=weight,
                use_8bit=use_8bit,
                use_4bit=use_4bit
            )
            for model_id, weight in zip(teacher_list, weights)
        ]
        
        # Dataset config
        if dataset_source == "HuggingFace":
            dataset = DatasetConfig(
                source_type='huggingface',
                source_path=dataset_path,
                text_column=text_column,
                max_samples=max_samples if max_samples > 0 else None
            )
        else:  # Local file
            if dataset_file is None:
                return None, "ERROR: Please upload a dataset file"
            
            dataset = DatasetConfig(
                source_type='local_file',
                source_path=dataset_file.name,
                text_column=text_column,
                max_samples=max_samples if max_samples > 0 else None
            )
        
        # Student config
        student = StudentModelConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads
        )
        
        # Training config
        training = TrainingConfig(
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            max_length=max_length,
            temperature=temperature,
            alpha=alpha,
            use_fp16=mixed_precision == 'FP16',
            use_bf16=mixed_precision == 'BF16'
        )
        
        # Complete config
        config = DistillationConfig(
            project_name=project_name,
            output_dir=output_dir,
            teachers=teachers,
            dataset=dataset,
            student=student,
            training=training
        )
        
        return config, "SUCCESS: Configuration created successfully"
        
    except Exception as e:
        return None, f"ERROR: Error creating configuration: {str(e)}"


def start_training(
    project_name: str,
    teacher_models: str,
    teacher_weights: str,
    use_quantization: str,
    dataset_source: str,
    dataset_path: str,
    dataset_file,
    text_column: str,
    max_samples: Optional[int],
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    max_length: int,
    temperature: float,
    alpha: float,
    mixed_precision: str,
    output_dir: str
) -> str:
    """Start training in background thread."""
    
    global training_engine, training_thread, training_in_progress
    
    if training_in_progress:
        return "WARNING: Training already in progress!"
    
    # Create configuration
    config, message = create_config_from_ui(
        project_name, teacher_models, teacher_weights, use_quantization,
        dataset_source, dataset_path, dataset_file, text_column, max_samples,
        hidden_size, num_layers, num_heads, batch_size, num_epochs,
        learning_rate, max_length, temperature, alpha, mixed_precision, output_dir
    )
    
    if config is None:
        return message
    
    # Initialize engine
    try:
        training_engine = AutomatedDistillationEngine(config)
        training_in_progress = True
        
        # Start training in background
        def train_worker():
            global training_in_progress
            try:
                training_engine.setup()
                training_engine.train()
            except Exception as e:
                logger.error(f"Training error: {e}")
            finally:
                training_in_progress = False
        
        training_thread = threading.Thread(target=train_worker)
        training_thread.start()
        
        return f"SUCCESS: Training started!\n\nProject: {project_name}\nOutput: {output_dir}"
        
    except Exception as e:
        training_in_progress = False
        return f"ERROR: Error starting training: {str(e)}"


def get_training_status() -> str:
    """Get current training status."""
    global training_engine, training_in_progress
    
    if not training_in_progress:
        return "STATUS: No training in progress"
    
    if training_engine is None:
        return "STATUS: Initializing..."
    
    try:
        step = training_engine.global_step
        epoch = training_engine.current_epoch + 1
        total_epochs = training_engine.config.training.num_epochs
        
        if training_engine.metrics['total_loss']:
            recent_loss = training_engine.metrics['total_loss'][-1]
            return (
                f"STATUS: Training in progress\n\n"
                f"Epoch: {epoch}/{total_epochs}\n"
                f"Step: {step}\n"
                f"Recent loss: {recent_loss:.4f}"
            )
        else:
            return f"STATUS: Training in progress\n\nEpoch: {epoch}/{total_epochs}\nStep: {step}"
    
    except Exception as e:
        return f"WARNING: Error getting status: {str(e)}"


def create_interface() -> gr.Blocks:
    """Create Gradio interface."""
    
    with gr.Blocks(title="LLM Distillation Engine", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # Automated Multi-Teacher LLM Distillation Engine
            
            Train compact student language models by distilling knowledge from multiple teacher models.
            This engine automates the entire process from model downloading to training.
            """
        )
        
        with gr.Tabs():
            # Tab 1: Configuration
            with gr.Tab("Configuration"):
                gr.Markdown("### Project Settings")
                
                with gr.Row():
                    project_name = gr.Textbox(
                        label="Project Name",
                        value="my_distilled_llm",
                        placeholder="Enter project name"
                    )
                    output_dir = gr.Textbox(
                        label="Output Directory",
                        value="./outputs",
                        placeholder="Path to save outputs"
                    )
                
                gr.Markdown("### Teacher Models")
                
                teacher_models = gr.Textbox(
                    label="Teacher Model IDs (comma-separated)",
                    placeholder="meta-llama/Llama-3.2-3B-Instruct, google/gemma-2-2b-it",
                    info="HuggingFace model IDs separated by commas"
                )
                
                with gr.Row():
                    teacher_weights = gr.Textbox(
                        label="Teacher Weights (optional)",
                        placeholder="0.6, 0.4",
                        info="Weights for each teacher (default: equal)"
                    )
                    use_quantization = gr.Dropdown(
                        choices=["None", "8-bit", "4-bit"],
                        value="8-bit",
                        label="Quantization",
                        info="Reduce teacher memory usage"
                    )
                
                gr.Markdown("### Dataset")
                
                dataset_source = gr.Radio(
                    choices=["HuggingFace", "Local File"],
                    value="HuggingFace",
                    label="Dataset Source"
                )
                
                dataset_path = gr.Textbox(
                    label="Dataset Path",
                    placeholder="wikitext",
                    info="HuggingFace dataset ID or local file path"
                )
                
                dataset_file = gr.File(
                    label="Upload Dataset File (JSON/JSONL/CSV/TXT)",
                    file_types=[".json", ".jsonl", ".csv", ".txt", ".parquet"],
                    visible=False
                )
                
                with gr.Row():
                    text_column = gr.Textbox(
                        label="Text Column",
                        value="text",
                        placeholder="Column name containing text"
                    )
                    max_samples = gr.Number(
                        label="Max Samples (0 = all)",
                        value=10000,
                        precision=0
                    )
                
                # Show/hide file upload based on source
                dataset_source.change(
                    fn=lambda x: gr.update(visible=(x == "Local File")),
                    inputs=[dataset_source],
                    outputs=[dataset_file]
                )
                
                gr.Markdown("### Student Model Architecture")
                
                with gr.Row():
                    hidden_size = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        value=768,
                        step=64,
                        label="Hidden Size"
                    )
                    num_layers = gr.Slider(
                        minimum=4,
                        maximum=32,
                        value=12,
                        step=2,
                        label="Number of Layers"
                    )
                    num_heads = gr.Slider(
                        minimum=4,
                        maximum=32,
                        value=12,
                        step=2,
                        label="Attention Heads"
                    )
                
                gr.Markdown("### Training Parameters")
                
                with gr.Row():
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=32,
                        value=4,
                        step=1,
                        label="Batch Size"
                    )
                    num_epochs = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Epochs"
                    )
                
                with gr.Row():
                    learning_rate = gr.Number(
                        label="Learning Rate",
                        value=5e-5,
                        precision=6
                    )
                    max_length = gr.Slider(
                        minimum=128,
                        maximum=2048,
                        value=512,
                        step=128,
                        label="Max Sequence Length"
                    )
                
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=1.0,
                        maximum=5.0,
                        value=2.0,
                        step=0.5,
                        label="Temperature",
                        info="Softening temperature for distillation"
                    )
                    alpha = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Alpha",
                        info="Weight for distillation loss"
                    )
                    mixed_precision = gr.Dropdown(
                        choices=["None", "FP16", "BF16"],
                        value="BF16",
                        label="Mixed Precision"
                    )
                
                start_btn = gr.Button("Start Training", variant="primary", size="lg")
                status_output = gr.Textbox(label="Status", lines=5)
                
                start_btn.click(
                    fn=start_training,
                    inputs=[
                        project_name, teacher_models, teacher_weights, use_quantization,
                        dataset_source, dataset_path, dataset_file, text_column, max_samples,
                        hidden_size, num_layers, num_heads, batch_size, num_epochs,
                        learning_rate, max_length, temperature, alpha, mixed_precision, output_dir
                    ],
                    outputs=[status_output]
                )
            
            # Tab 2: Monitor
            with gr.Tab("Monitor"):
                gr.Markdown("### Training Progress")
                
                status_display = gr.Textbox(label="Current Status", lines=8)
                refresh_btn = gr.Button("Refresh Status")
                
                refresh_btn.click(
                    fn=get_training_status,
                    inputs=[],
                    outputs=[status_display]
                )
                
                # Auto-refresh every 5 seconds
                timer = gr.Timer(5)
                timer.tick(
                    fn=get_training_status,
                    inputs=[],
                    outputs=[status_display]
                )
            
            # Tab 3: Help
            with gr.Tab("Help"):
                gr.Markdown(
                    """
                    ## How to Use
                    
                    ### 1. Configure Project
                    - **Project Name**: A descriptive name for your distillation project
                    - **Output Directory**: Where to save checkpoints and final model
                    
                    ### 2. Select Teacher Models
                    - Enter HuggingFace model IDs (e.g., `meta-llama/Llama-3.2-3B-Instruct`)
                    - Separate multiple teachers with commas
                    - Optionally specify weights (default: equal weighting)
                    - Choose quantization to reduce memory usage
                    
                    ### 3. Prepare Dataset
                    - **HuggingFace**: Use public datasets like `wikitext`, `openwebtext`
                    - **Local File**: Upload your own JSON, JSONL, CSV, or TXT file
                    - Specify the column/field containing text data
                    - Limit samples for quick testing
                    
                    ### 4. Design Student Model
                    - **Hidden Size**: Wider = more capacity (768 recommended)
                    - **Layers**: Deeper = more capacity (12 recommended)
                    - **Heads**: More heads = better attention (12 recommended)
                    
                    ### 5. Set Training Parameters
                    - **Batch Size**: Larger = faster but more memory (4-8 recommended)
                    - **Epochs**: More = better but longer (3 recommended)
                    - **Learning Rate**: 5e-5 is a good default
                    - **Temperature**: Higher = softer distillation (2.0 recommended)
                    - **Alpha**: Balance distillation vs ground truth (0.7 recommended)
                    
                    ### 6. Monitor Training
                    - Switch to the **Monitor** tab after starting training
                    - Status updates automatically every 5 seconds
                    - Check the output directory for checkpoints
                    
                    ## Tips
                    
                    - Start with small datasets (max_samples=1000) for testing
                    - Use 8-bit quantization if you have limited GPU memory
                    - Enable BF16 mixed precision on modern GPUs (A100, H100)
                    - Keep batch_size × gradient_accumulation = 16 for best results
                    
                    ## Requirements
                    
                    - CUDA-capable GPU (16GB+ VRAM recommended)
                    - Python 3.8+
                    - PyTorch 2.0+
                    - Transformers 4.30+
                    - 50GB+ disk space for caching models
                    """
                )
        
        gr.Markdown(
            """
            ---
            **Automated Multi-Teacher LLM Distillation Engine** | 
            Train compact, efficient language models from multiple teachers
            """
        )
    
    return app


def main():
    """Launch Gradio web UI."""
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
