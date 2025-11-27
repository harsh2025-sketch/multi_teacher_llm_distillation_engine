"""
Streamlit UI for Multi-Teacher LLM Distillation Engine

A modern, user-friendly web interface for configuring and running
multi-teacher knowledge distillation experiments.
"""

import streamlit as st
import yaml
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Multi-Teacher LLM Distillation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .config-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_example_configs():
    """Load example configuration files"""
    config_dir = Path("examples/configs")
    configs = {}
    
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    configs[config_file.stem] = yaml.safe_load(f)
            except Exception as e:
                st.warning(f"Could not load {config_file.name}: {e}")
    
    return configs

def save_config(config, filename):
    """Save configuration to YAML file"""
    output_dir = Path("configs/user_configs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return filepath

def main():
    # Header
    st.markdown('<div class="main-header">🧠 Multi-Teacher LLM Distillation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated knowledge distillation from multiple teacher models</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("assets/svg.png", use_container_width=True)
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Home", "⚙️ Configuration", "🚀 Training", "📊 Monitoring", "📚 Documentation"]
        )
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        if st.button("💾 Save Configuration"):
            st.session_state.save_config = True
        
        if st.button("📥 Load Configuration"):
            st.session_state.load_config = True
        
        st.markdown("---")
        st.markdown("""
        **Project Info**
        - 🐍 Python 3.10+
        - 🔥 PyTorch 2.0+
        - 🤗 HuggingFace
        - 📄 Apache 2.0 License
        """)
    
    # Main content area
    if page == "🏠 Home":
        show_home_page()
    elif page == "⚙️ Configuration":
        show_configuration_page()
    elif page == "🚀 Training":
        show_training_page()
    elif page == "📊 Monitoring":
        show_monitoring_page()
    elif page == "📚 Documentation":
        show_documentation_page()

def show_home_page():
    """Home page with overview and quick start"""
    st.header("Welcome to Multi-Teacher LLM Distillation! 👋")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ✨ Key Features")
        st.markdown("""
        - 🤖 **Multi-Teacher Support**: Use multiple LLMs as teachers
        - 🔄 **Full Automation**: Automatic model download and setup
        - 📊 **Flexible Datasets**: HuggingFace, JSON, CSV, TXT support
        - ⚡ **Quantization**: 4-bit and 8-bit model compression
        - 🎯 **Mixed Precision**: FP16/BF16 training
        - 💾 **Auto Checkpointing**: Resume training anytime
        """)
    
    with col2:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. **Configure**: Set up teachers, student, and dataset
        2. **Train**: Start the distillation process
        3. **Monitor**: Track progress and metrics
        4. **Export**: Save your trained student model
        
        Use the sidebar to navigate between sections!
        """)
    
    with col3:
        st.markdown("### 📈 System Requirements")
        st.markdown("""
        **Minimum:**
        - GPU: 16GB VRAM (T4, V100)
        - RAM: 32GB
        - Storage: 50GB
        
        **Recommended:**
        - GPU: A100 (40GB)
        - RAM: 64GB+
        - Storage: 200GB SSD
        """)
    
    st.markdown("---")
    
    # Example configurations
    st.subheader("📁 Example Configurations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Quick Start**\n\nSmall models (GPT-2)\n~10 minutes on T4\nPerfect for testing")
        if st.button("Load Quick Start Config"):
            st.session_state.selected_example = "quick_start"
    
    with col2:
        st.success("**Production**\n\nLlama 3.2 + Gemma 2\n6-12 hours on A100\nProduction quality")
        if st.button("Load Production Config"):
            st.session_state.selected_example = "production"
    
    with col3:
        st.warning("**Custom Dataset**\n\nUse your own data\nJSON/JSONL/CSV\nFull control")
        if st.button("Load Custom Config"):
            st.session_state.selected_example = "custom_dataset"

def show_configuration_page():
    """Configuration page for setting up distillation"""
    st.header("⚙️ Configuration")
    
    # Initialize session state for config
    if 'config' not in st.session_state:
        st.session_state.config = {
            'project_name': 'my_distillation_project',
            'output_dir': './outputs',
            'seed': 42,
            'cache_dir': './cache',
            'teachers': [],
            'dataset': {},
            'student': {},
            'training': {}
        }
    
    # Tabs for different configuration sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Project", "👨‍🏫 Teachers", "📚 Dataset", "🎓 Student", "🎯 Training"])
    
    with tab1:
        st.subheader("Project Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            project_name = st.text_input(
                "Project Name",
                value=st.session_state.config.get('project_name', 'my_project'),
                help="Unique identifier for this project"
            )
            st.session_state.config['project_name'] = project_name
            
            output_dir = st.text_input(
                "Output Directory",
                value=st.session_state.config.get('output_dir', './outputs'),
                help="Where to save models and logs"
            )
            st.session_state.config['output_dir'] = output_dir
        
        with col2:
            seed = st.number_input(
                "Random Seed",
                value=st.session_state.config.get('seed', 42),
                min_value=0,
                help="For reproducibility"
            )
            st.session_state.config['seed'] = seed
            
            cache_dir = st.text_input(
                "Cache Directory",
                value=st.session_state.config.get('cache_dir', './cache'),
                help="For caching downloaded models"
            )
            st.session_state.config['cache_dir'] = cache_dir
    
    with tab2:
        st.subheader("Teacher Models")
        
        # Add teacher button
        if st.button("➕ Add Teacher Model"):
            if 'teachers' not in st.session_state.config:
                st.session_state.config['teachers'] = []
            st.session_state.config['teachers'].append({
                'model_id': '',
                'weight': 0.5,
                'use_8bit': True,
                'use_4bit': False,
                'trust_remote_code': True
            })
        
        # Display existing teachers
        teachers = st.session_state.config.get('teachers', [])
        
        for idx, teacher in enumerate(teachers):
            with st.expander(f"Teacher {idx + 1}: {teacher.get('model_id', 'Not set')}", expanded=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    model_id = st.text_input(
                        "Model ID (HuggingFace)",
                        value=teacher.get('model_id', ''),
                        key=f"teacher_model_{idx}",
                        placeholder="e.g., meta-llama/Llama-3.2-3B-Instruct"
                    )
                    teacher['model_id'] = model_id
                
                with col2:
                    weight = st.slider(
                        "Weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=teacher.get('weight', 0.5),
                        step=0.1,
                        key=f"teacher_weight_{idx}"
                    )
                    teacher['weight'] = weight
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    use_8bit = st.checkbox(
                        "Use 8-bit quantization",
                        value=teacher.get('use_8bit', True),
                        key=f"teacher_8bit_{idx}"
                    )
                    teacher['use_8bit'] = use_8bit
                
                with col2:
                    use_4bit = st.checkbox(
                        "Use 4-bit quantization",
                        value=teacher.get('use_4bit', False),
                        key=f"teacher_4bit_{idx}"
                    )
                    teacher['use_4bit'] = use_4bit
                
                with col3:
                    if st.button("🗑️ Remove", key=f"remove_teacher_{idx}"):
                        st.session_state.config['teachers'].pop(idx)
                        st.rerun()
        
        # Popular models suggestions
        st.markdown("---")
        st.markdown("**Popular Teacher Models:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.code("meta-llama/Llama-3.2-3B-Instruct")
            st.code("gpt2")
        
        with col2:
            st.code("google/gemma-2-2b-it")
            st.code("distilgpt2")
        
        with col3:
            st.code("microsoft/phi-2")
            st.code("EleutherAI/pythia-160m")
    
    with tab3:
        st.subheader("Dataset Configuration")
        
        if 'dataset' not in st.session_state.config:
            st.session_state.config['dataset'] = {}
        
        dataset = st.session_state.config['dataset']
        
        source_type = st.selectbox(
            "Dataset Source",
            ["huggingface", "local_file", "directory"],
            index=["huggingface", "local_file", "directory"].index(dataset.get('source_type', 'huggingface'))
        )
        dataset['source_type'] = source_type
        
        if source_type == "huggingface":
            col1, col2 = st.columns(2)
            
            with col1:
                source_path = st.text_input(
                    "Dataset ID",
                    value=dataset.get('source_path', 'wikitext'),
                    placeholder="e.g., wikitext, openwebtext"
                )
                dataset['source_path'] = source_path
                
                split = st.text_input(
                    "Split",
                    value=dataset.get('split', 'train')
                )
                dataset['split'] = split
            
            with col2:
                config_name = st.text_input(
                    "Config Name (optional)",
                    value=dataset.get('config_name', ''),
                    placeholder="e.g., wikitext-103-raw-v1"
                )
                dataset['config_name'] = config_name if config_name else None
                
                text_column = st.text_input(
                    "Text Column",
                    value=dataset.get('text_column', 'text')
                )
                dataset['text_column'] = text_column
            
            max_samples = st.number_input(
                "Max Samples (0 = all)",
                value=dataset.get('max_samples', 0),
                min_value=0,
                step=1000
            )
            dataset['max_samples'] = max_samples if max_samples > 0 else None
        
        elif source_type == "local_file":
            source_path = st.text_input(
                "File Path",
                value=dataset.get('source_path', ''),
                placeholder="e.g., ./my_data.json"
            )
            dataset['source_path'] = source_path
            
            file_format = st.selectbox(
                "File Format",
                ["json", "jsonl", "csv", "txt", "parquet"],
                index=["json", "jsonl", "csv", "txt", "parquet"].index(dataset.get('file_format', 'json'))
            )
            dataset['file_format'] = file_format
            
            uploaded_file = st.file_uploader(
                "Or upload a file",
                type=['json', 'jsonl', 'csv', 'txt', 'parquet']
            )
            
            if uploaded_file:
                # Save uploaded file
                upload_dir = Path("data/uploads")
                upload_dir.mkdir(parents=True, exist_ok=True)
                filepath = upload_dir / uploaded_file.name
                
                with open(filepath, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                dataset['source_path'] = str(filepath)
                st.success(f"File uploaded: {filepath}")
    
    with tab4:
        st.subheader("Student Model Architecture")
        
        if 'student' not in st.session_state.config:
            st.session_state.config['student'] = {}
        
        student = st.session_state.config['student']
        
        # Architecture presets
        preset = st.selectbox(
            "Architecture Preset",
            ["Custom", "Tiny (~50M)", "Small (~250M)", "Medium (~500M)", "Large (~1B)"]
        )
        
        if preset == "Tiny (~50M)":
            student.update({'hidden_size': 384, 'num_hidden_layers': 6, 'num_attention_heads': 6})
        elif preset == "Small (~250M)":
            student.update({'hidden_size': 768, 'num_hidden_layers': 12, 'num_attention_heads': 12})
        elif preset == "Medium (~500M)":
            student.update({'hidden_size': 1024, 'num_hidden_layers': 16, 'num_attention_heads': 16})
        elif preset == "Large (~1B)":
            student.update({'hidden_size': 2048, 'num_hidden_layers': 24, 'num_attention_heads': 24})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vocab_size = st.number_input(
                "Vocabulary Size",
                value=student.get('vocab_size', 128256),
                min_value=1000,
                step=1000
            )
            student['vocab_size'] = vocab_size
            
            hidden_size = st.number_input(
                "Hidden Size",
                value=student.get('hidden_size', 768),
                min_value=128,
                step=64
            )
            student['hidden_size'] = hidden_size
        
        with col2:
            intermediate_size = st.number_input(
                "Intermediate Size",
                value=student.get('intermediate_size', 2048),
                min_value=256,
                step=256
            )
            student['intermediate_size'] = intermediate_size
            
            num_hidden_layers = st.number_input(
                "Number of Layers",
                value=student.get('num_hidden_layers', 12),
                min_value=1,
                max_value=48
            )
            student['num_hidden_layers'] = num_hidden_layers
        
        with col3:
            num_attention_heads = st.number_input(
                "Attention Heads",
                value=student.get('num_attention_heads', 12),
                min_value=1,
                max_value=32
            )
            student['num_attention_heads'] = num_attention_heads
            
            max_position_embeddings = st.number_input(
                "Max Position Embeddings",
                value=student.get('max_position_embeddings', 2048),
                min_value=512,
                step=512
            )
            student['max_position_embeddings'] = max_position_embeddings
    
    with tab5:
        st.subheader("Training Configuration")
        
        if 'training' not in st.session_state.config:
            st.session_state.config['training'] = {}
        
        training = st.session_state.config['training']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Batch Settings**")
            
            batch_size = st.number_input(
                "Batch Size",
                value=training.get('batch_size', 4),
                min_value=1,
                max_value=64
            )
            training['batch_size'] = batch_size
            
            gradient_accumulation_steps = st.number_input(
                "Gradient Accumulation Steps",
                value=training.get('gradient_accumulation_steps', 4),
                min_value=1,
                max_value=32
            )
            training['gradient_accumulation_steps'] = gradient_accumulation_steps
            
            st.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
            
            num_epochs = st.number_input(
                "Number of Epochs",
                value=training.get('num_epochs', 3),
                min_value=1,
                max_value=100
            )
            training['num_epochs'] = num_epochs
            
            max_length = st.number_input(
                "Max Sequence Length",
                value=training.get('max_length', 512),
                min_value=128,
                max_value=4096,
                step=128
            )
            training['max_length'] = max_length
        
        with col2:
            st.markdown("**Optimization**")
            
            learning_rate = st.number_input(
                "Learning Rate",
                value=training.get('learning_rate', 5e-5),
                min_value=1e-6,
                max_value=1e-3,
                format="%.2e"
            )
            training['learning_rate'] = learning_rate
            
            weight_decay = st.number_input(
                "Weight Decay",
                value=training.get('weight_decay', 0.01),
                min_value=0.0,
                max_value=0.1,
                step=0.01
            )
            training['weight_decay'] = weight_decay
            
            warmup_ratio = st.slider(
                "Warmup Ratio",
                min_value=0.0,
                max_value=0.5,
                value=training.get('warmup_ratio', 0.1),
                step=0.05
            )
            training['warmup_ratio'] = warmup_ratio
            
            lr_scheduler_type = st.selectbox(
                "LR Scheduler",
                ["cosine", "linear", "constant"],
                index=["cosine", "linear", "constant"].index(training.get('lr_scheduler_type', 'cosine'))
            )
            training['lr_scheduler_type'] = lr_scheduler_type
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distillation**")
            
            temperature = st.slider(
                "Temperature",
                min_value=1.0,
                max_value=5.0,
                value=training.get('temperature', 2.0),
                step=0.5
            )
            training['temperature'] = temperature
            
            alpha = st.slider(
                "Alpha (distillation weight)",
                min_value=0.0,
                max_value=1.0,
                value=training.get('alpha', 0.7),
                step=0.1
            )
            training['alpha'] = alpha
        
        with col2:
            st.markdown("**Mixed Precision**")
            
            use_bf16 = st.checkbox(
                "Use BF16 (A100/H100)",
                value=training.get('use_bf16', True)
            )
            training['use_bf16'] = use_bf16
            
            use_fp16 = st.checkbox(
                "Use FP16 (V100/T4)",
                value=training.get('use_fp16', False)
            )
            training['use_fp16'] = use_fp16
        
        st.markdown("---")
        st.markdown("**Checkpointing**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            save_steps = st.number_input(
                "Save Steps",
                value=training.get('save_steps', 500),
                min_value=10,
                step=100
            )
            training['save_steps'] = save_steps
        
        with col2:
            save_total_limit = st.number_input(
                "Keep Last N Checkpoints",
                value=training.get('save_total_limit', 3),
                min_value=1,
                max_value=10
            )
            training['save_total_limit'] = save_total_limit
    
    # Save/Export configuration
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Configuration", use_container_width=True):
            filename = f"{st.session_state.config['project_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            filepath = save_config(st.session_state.config, filename)
            st.success(f"Configuration saved: {filepath}")
    
    with col2:
        if st.button("📋 View YAML", use_container_width=True):
            st.code(yaml.dump(st.session_state.config, default_flow_style=False), language='yaml')
    
    with col3:
        config_yaml = yaml.dump(st.session_state.config, default_flow_style=False)
        st.download_button(
            label="📥 Download Config",
            data=config_yaml,
            file_name=f"{st.session_state.config['project_name']}.yaml",
            mime="text/yaml",
            use_container_width=True
        )

def show_training_page():
    """Training page to start and manage distillation"""
    st.header("🚀 Training")
    
    if 'config' not in st.session_state or not st.session_state.config.get('teachers'):
        st.warning("⚠️ Please configure your project in the Configuration page first!")
        return
    
    # Display current configuration summary
    st.subheader("Current Configuration Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Project", st.session_state.config.get('project_name', 'N/A'))
        st.metric("Teachers", len(st.session_state.config.get('teachers', [])))
    
    with col2:
        student = st.session_state.config.get('student', {})
        st.metric("Student Layers", student.get('num_hidden_layers', 'N/A'))
        st.metric("Hidden Size", student.get('hidden_size', 'N/A'))
    
    with col3:
        training = st.session_state.config.get('training', {})
        st.metric("Epochs", training.get('num_epochs', 'N/A'))
        st.metric("Batch Size", training.get('batch_size', 'N/A'))
    
    st.markdown("---")
    
    # Training command
    st.subheader("Training Command")
    
    config_name = f"{st.session_state.config['project_name']}.yaml"
    
    st.code(f"""
# Save configuration first, then run:
python src/cli.py --config configs/user_configs/{config_name}

# Or with accelerate for multi-GPU:
accelerate launch src/cli.py --config configs/user_configs/{config_name}
    """, language='bash')
    
    st.info("💡 **Note**: Copy the command above and run it in your terminal to start training.")
    
    # Quick start button (for demonstration)
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.warning("⚠️ Training will run in your terminal. Make sure you have:")
        st.markdown("""
        - Sufficient GPU memory
        - All required dependencies installed
        - HuggingFace token set (if using private models)
        """)
    
    with col2:
        if st.button("🎯 Prepare & Start", type="primary", use_container_width=True):
            # Save config
            filename = f"{st.session_state.config['project_name']}.yaml"
            filepath = save_config(st.session_state.config, filename)
            
            st.success(f"✅ Configuration saved: {filepath}")
            st.info(f"🚀 Run the command above in your terminal to start training!")

def show_monitoring_page():
    """Monitoring page for tracking training progress"""
    st.header("📊 Monitoring")
    
    st.info("🔄 Training monitoring features coming soon! For now, monitor training via terminal logs.")
    
    # Placeholder for future monitoring features
    st.subheader("Training Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Epoch", "-", help="Current epoch")
    
    with col2:
        st.metric("Step", "-", help="Current training step")
    
    with col3:
        st.metric("Loss", "-", help="Current loss value")
    
    with col4:
        st.metric("ETA", "-", help="Estimated time remaining")
    
    st.markdown("---")
    
    # Log viewer
    st.subheader("Training Logs")
    
    log_file = st.text_input("Log File Path", value="outputs/training.log")
    
    if st.button("📖 Load Logs"):
        if Path(log_file).exists():
            with open(log_file, 'r') as f:
                logs = f.read()
            st.text_area("Logs", logs, height=400)
        else:
            st.error(f"Log file not found: {log_file}")
    
    st.markdown("---")
    
    # Checkpoint manager
    st.subheader("Checkpoints")
    
    checkpoint_dir = st.text_input("Checkpoint Directory", value="outputs/checkpoints")
    
    if Path(checkpoint_dir).exists():
        checkpoints = list(Path(checkpoint_dir).glob("*/"))
        if checkpoints:
            st.success(f"Found {len(checkpoints)} checkpoint(s)")
            
            checkpoint_data = []
            for ckpt in checkpoints:
                checkpoint_data.append({
                    "Name": ckpt.name,
                    "Path": str(ckpt),
                    "Size": f"{sum(f.stat().st_size for f in ckpt.rglob('*')) / (1024**2):.2f} MB"
                })
            
            df = pd.DataFrame(checkpoint_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No checkpoints found yet.")
    else:
        st.warning("Checkpoint directory does not exist. Start training to create checkpoints.")

def show_documentation_page():
    """Documentation page with guides and help"""
    st.header("📚 Documentation")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Quick Start", "📖 User Guide", "❓ FAQ", "🔗 Links"])
    
    with tab1:
        st.markdown("""
        ## Quick Start Guide
        
        ### 1. Configure Your Project
        
        Navigate to the **Configuration** page and set up:
        - **Project Settings**: Name, output directory
        - **Teachers**: Add 2+ teacher models from HuggingFace
        - **Dataset**: Choose HuggingFace dataset or upload your own
        - **Student**: Configure model architecture
        - **Training**: Set batch size, learning rate, epochs
        
        ### 2. Save Configuration
        
        Click "💾 Save Configuration" to save your settings.
        
        ### 3. Start Training
        
        Go to the **Training** page and copy the command to run in your terminal:
        
        ```bash
        python src/cli.py --config configs/user_configs/your_config.yaml
        ```
        
        ### 4. Monitor Progress
        
        Track training in your terminal or check the **Monitoring** page for logs.
        
        ### 5. Use Your Model
        
        After training, find your model in `outputs/final_model/`.
        """)
    
    with tab2:
        st.markdown("""
        ## User Guide
        
        ### Teacher Models
        
        You can use any causal language model from HuggingFace:
        
        **Popular choices:**
        - `meta-llama/Llama-3.2-3B-Instruct` (3B params)
        - `google/gemma-2-2b-it` (2B params)
        - `microsoft/phi-2` (2.7B params)
        - `gpt2` (124M params, for testing)
        
        **Tips:**
        - Use 8-bit quantization to save memory
        - Weight teachers based on their quality
        - Mix models of different sizes for best results
        
        ### Datasets
        
        **HuggingFace Datasets:**
        - `wikitext` - Wikipedia articles
        - `openwebtext` - Web content
        - `HuggingFaceFW/fineweb-edu` - Educational content
        
        **Custom Data:**
        - Upload JSON, JSONL, CSV, or TXT files
        - Each example should have a "text" field
        - Larger datasets = better student models
        
        ### Student Architecture
        
        Choose based on your deployment needs:
        
        | Size | Params | Use Case |
        |------|--------|----------|
        | Tiny | ~50M | Edge devices, testing |
        | Small | ~250M | Mobile, embedded |
        | Medium | ~500M | Standard deployment |
        | Large | ~1B | High-quality applications |
        
        ### Training Tips
        
        - **Batch size**: Start with 4, increase if you have memory
        - **Learning rate**: 5e-5 is a good default
        - **Temperature**: 2.0 for soft distillation
        - **Alpha**: 0.7 means 70% distillation, 30% hard labels
        - **Epochs**: 3 is usually sufficient
        
        ### System Requirements
        
        **Minimum:**
        - NVIDIA GPU with 16GB+ VRAM
        - 32GB RAM
        - 50GB storage
        
        **Recommended:**
        - A100 GPU (40GB VRAM)
        - 64GB+ RAM
        - 200GB SSD
        """)
    
    with tab3:
        st.markdown("""
        ## Frequently Asked Questions
        
        ### Q: How long does training take?
        
        **A**: Depends on your configuration:
        - Quick start (1000 samples): ~10 minutes on T4
        - Small dataset (10K samples): ~1-2 hours on V100
        - Production (100K samples): ~6-12 hours on A100
        
        ### Q: Out of memory errors?
        
        **A**: Try these solutions:
        1. Enable 8-bit quantization for teachers
        2. Reduce batch size to 2
        3. Increase gradient accumulation steps
        4. Use smaller student model
        5. Reduce sequence length to 256
        
        ### Q: Can I use multiple GPUs?
        
        **A**: Yes! Use `accelerate`:
        ```bash
        accelerate config  # Run once
        accelerate launch src/cli.py --config your_config.yaml
        ```
        
        ### Q: How do I resume training?
        
        **A**: Just run the same command again. The engine automatically detects and resumes from the latest checkpoint.
        
        ### Q: Which teacher models should I use?
        
        **A**: 
        - For quality: Llama 3.2 + Gemma 2
        - For speed: GPT-2 + DistilGPT-2
        - For balance: Mix a large model (8B) with a medium one (2-3B)
        
        ### Q: Can I use private HuggingFace models?
        
        **A**: Yes, set your HF token:
        ```bash
        export HF_TOKEN=your_token_here
        ```
        
        ### Q: How do I export my trained model?
        
        **A**: Use the export script:
        ```bash
        python src/export_to_safetensors.py \\
            --in_dir ./outputs/checkpoints/step_10000 \\
            --release_dir ./hf_model
        ```
        """)
    
    with tab4:
        st.markdown("""
        ## Useful Links
        
        ### Project Resources
        
        - 📁 [GitHub Repository](https://github.com/harsh2025-sketch/multi_teacher_llm_distillation_engine)
        - 📄 [Full Documentation](docs/DETAILS.md)
        - 📝 [Contributing Guide](CONTRIBUTING.md)
        - ⚖️ [License (Apache 2.0)](LICENSE)
        
        ### External Resources
        
        - 🤗 [HuggingFace Models](https://huggingface.co/models)
        - 📊 [HuggingFace Datasets](https://huggingface.co/datasets)
        - 🔥 [PyTorch Documentation](https://pytorch.org/docs/)
        - ⚡ [Accelerate Documentation](https://huggingface.co/docs/accelerate)
        
        ### Research Papers
        
        - [Knowledge Distillation (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
        - [DistilBERT (Sanh et al., 2019)](https://arxiv.org/abs/1910.01108)
        - [Multi-Teacher Distillation (You et al., 2017)](https://arxiv.org/abs/1711.02132)
        - [LLaMA (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971)
        
        ### Community
        
        - 💬 [GitHub Discussions](https://github.com/harsh2025-sketch/multi_teacher_llm_distillation_engine/discussions)
        - 🐛 [Report Issues](https://github.com/harsh2025-sketch/multi_teacher_llm_distillation_engine/issues)
        - ⭐ [Star on GitHub](https://github.com/harsh2025-sketch/multi_teacher_llm_distillation_engine)
        """)

if __name__ == "__main__":
    main()
