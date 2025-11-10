# Stable Diffusion 3.5 LoRA Fine-tuning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/)
[![Diffusers](https://img.shields.io/badge/🧨-Diffusers-orange)](https://github.com/huggingface/diffusers)

A professional, production-ready implementation for fine-tuning Stable Diffusion 3.5 models using LoRA (Low-Rank Adaptation) adapters. This script provides comprehensive support for both transformer and text encoder LoRA training with advanced features for memory efficiency and distributed training.

> **🎯 Perfect for**: Custom image generation, style transfer, domain adaptation, and specialized visual content creation with minimal computational overhead.

## ✨ Features

-   **🚀 SD3.5 Support**: Full compatibility with Stable Diffusion 3.5 Medium architecture
-   **🔧 LoRA Training**: Efficient fine-tuning using Low-Rank Adaptation for both transformer and text encoders
-   **⚡ Mixed Precision**: FP16/BF16 training support with automatic gradient scaling
-   **💾 Memory Efficient**: Gradient checkpointing and optimized memory usage
-   **🔄 Distributed Training**: Multi-GPU support via Accelerate framework
-   **📊 Advanced Sampling**: Custom timestep sampling with configurable weighting schemes
-   **✅ Validation**: Built-in validation pipeline with image generation during training
-   **📈 Comprehensive Logging**: TensorBoard and Weights & Biases integration
-   **🛡️ Robust Error Handling**: Professional error handling and recovery mechanisms
-   **🔄 Resume Training**: Checkpoint saving and resuming capabilities

## 📋 Requirements

### Dependencies

#### Core Dependencies

```bash
# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core ML libraries
pip install accelerate>=0.25.0 transformers>=4.35.0 diffusers>=0.25.0
pip install peft>=0.7.0 datasets>=2.15.0

# Image processing and utilities
pip install pillow>=9.0.0 tqdm>=4.64.0
```

#### Optional Dependencies

```bash
# For experiment tracking and logging
pip install wandb tensorboard

# For advanced optimizers
pip install bitsandbytes>=0.41.0  # 8-bit AdamW
pip install prodigyopt>=1.0       # Prodigy optimizer

# For development
pip install black flake8 pytest
```

### Hardware Requirements

-   **Minimum**: 12GB VRAM GPU (RTX 3060 12GB, RTX 4070, etc.)
-   **Recommended**: 16GB+ VRAM GPU (RTX 4080, RTX 4090, A100, etc.)
-   **For distributed training**: Multiple GPUs with NVLink recommended

### Model Requirements

-   Stable Diffusion 3.5 Medium model weights
-   Can be loaded from HuggingFace Hub or local path

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/seochan99/stable-diffusion-3.5-text2image-lora.git
cd stable-diffusion-3.5-text2image-lora

# Run setup script (installs dependencies and configures accelerate)
bash scripts/setup.sh
```

### 2. Prepare Your Dataset

We provide an example dataset structure in `examples/dataset/`. You can:

**Option A: Use the example structure**

```bash
# Add your images to examples/dataset/images/
# Update examples/dataset/metadata.jsonl with your captions
```

**Option B: Create your own dataset**

```bash
your_dataset/
├── metadata.jsonl
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Where `metadata.jsonl` contains:

```json
{"image": "images/image1.jpg", "caption": "A beautiful landscape"}
{"image": "images/image2.jpg", "caption": "A portrait of a person"}
```

### 3. Start Training

**Easy way (recommended for beginners):**

```bash
# Basic training with good defaults
bash scripts/train_basic.sh

# Advanced training with all features
bash scripts/train_advanced.sh
```

**Manual way (for customization):**

```bash
accelerate launch train_text_to_image_lora_sd35.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-3.5-medium" \
  --train_data_dir "./examples/dataset" \
  --output_dir "./outputs/sd35-lora" \
  --resolution 1024 \
  --train_batch_size 2 \
  --num_train_epochs 10 \
  --rank 64 \
  --learning_rate 1e-4 \
  --mixed_precision fp16 \
  --validation_prompt "a beautiful landscape" \
  --validation_epochs 5
```

### 4. Generate Images with Your Trained LoRA

Once training is complete, generate images with your custom LoRA:

**Easy way (recommended):**

```bash
# Generate images with default settings
bash scripts/inference.sh

# Customize with environment variables
PROMPT="a futuristic cityscape at sunset" \
NUM_IMAGES=8 \
bash scripts/inference.sh
```

**Manual way (for full control):**

```bash
python inference.py \
  --lora_path "./outputs/sd35-lora-basic" \
  --prompt "your amazing prompt here" \
  --num_images 4 \
  --height 1024 \
  --width 1024 \
  --num_inference_steps 28 \
  --guidance_scale 7.0 \
  --seed 42
```

### 5. Environment Variables (Optional)

Customize training and inference with environment variables:

```bash
# Training customization
MODEL_NAME="stabilityai/stable-diffusion-3.5-medium" \
DATASET_DIR="./your_custom_dataset" \
BATCH_SIZE=4 \
EPOCHS=20 \
bash scripts/train_basic.sh

# Inference customization
PROMPT="your custom prompt" \
NUM_IMAGES=8 \
STEPS=50 \
RESOLUTION=1024 \
bash scripts/inference.sh
```

## ☁️ Google Colab Fine-Tuning

Want to fine-tune SD3.5 on free or paid Colab GPUs? Use the bundled notebook:

- Notebook: [`SD35_LoRA_Colab.ipynb`](SD35_LoRA_Colab.ipynb)
- Step-by-step guide: [`docs/COLAB_FINETUNING.md`](docs/COLAB_FINETUNING.md)

The notebook performs environment setup, Hugging Face auth, dataset validation, LoRA training, and inference validation. Follow the guide to configure dataset paths, adjust hyperparameters for T4/A100 runtimes, and resume or export runs.

## ⚙️ Configuration Options

### Core Training Parameters

| Parameter            | Description               | Default | Recommended           |
| -------------------- | ------------------------- | ------- | --------------------- |
| `--resolution`       | Training image resolution | 1024    | 1024 for SD3.5        |
| `--train_batch_size` | Batch size per device     | 4       | 2-4 depending on VRAM |
| `--learning_rate`    | Learning rate             | 1e-4    | 1e-4 to 5e-4          |
| `--num_train_epochs` | Number of epochs          | 1       | 10-50                 |
| `--mixed_precision`  | Precision mode            | None    | `fp16` or `bf16`      |

### LoRA Configuration

| Parameter              | Description                | Default | Recommended             |
| ---------------------- | -------------------------- | ------- | ----------------------- |
| `--rank`               | LoRA rank                  | 4       | 64-128                  |
| `--lora_alpha`         | LoRA alpha scaling         | None    | rank \* 2               |
| `--train_text_encoder` | Train text encoders        | False   | True for better results |
| `--text_encoder_lr`    | Text encoder learning rate | 5e-5    | 1e-5 to 5e-5            |

### Advanced Features

| Parameter                  | Description                   | Default        | Notes                              |
| -------------------------- | ----------------------------- | -------------- | ---------------------------------- |
| `--gradient_checkpointing` | Enable gradient checkpointing | False          | Reduces memory by ~50%             |
| `--weighting_scheme`       | Loss weighting scheme         | "logit_normal" | Options: sigma_sqrt, mode, cosmap  |
| `--validation_prompt`      | Prompt for validation images  | None           | Required for validation            |
| `--validation_epochs`      | Epochs between validations    | 50             | Set to 1-5 for frequent validation |
| `--checkpointing_steps`    | Steps between checkpoints     | 500            | Adjust based on training length    |
| `--precondition_outputs`   | Enable output preconditioning | 1              | As per SD3 paper                   |

### Inference Parameters

| Parameter               | Description                  | Default  | Notes                       |
| ----------------------- | ---------------------------- | -------- | --------------------------- |
| `--lora_path`           | Path to trained LoRA weights | Required | Directory with .safetensors |
| `--lora_scale`          | LoRA effect strength         | 1.0      | 0.0-2.0 range typical       |
| `--num_images`          | Number of images to generate | 1        | Batch generation            |
| `--num_inference_steps` | Denoising steps              | 28       | More steps = better quality |
| `--guidance_scale`      | Prompt adherence strength    | 7.0      | Higher = more faithful      |
| `--seed`                | Random seed                  | None     | For reproducible results    |

## 📊 Monitoring and Logging

### TensorBoard

```bash
tensorboard --logdir ./outputs/sd35-lora/logs
```

### Weights & Biases

```bash
# Login first
wandb login

# Then add to training command
--report_to wandb --run_name "my-experiment"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Option 1: Reduce batch size and enable gradient checkpointing
--train_batch_size 1 --gradient_checkpointing --gradient_accumulation_steps 4

# Option 2: Use CPU offloading for models
--mixed_precision fp16 --gradient_checkpointing

# Option 3: Reduce resolution temporarily
--resolution 512
```

#### 2. FP16 Gradient Scaling Errors

```bash
# Switch to bfloat16 (recommended for modern GPUs)
--mixed_precision bf16

# Or use full precision (slower but stable)
--mixed_precision no
```

#### 3. Slow Training Performance

```bash
# Enable all optimizations
--gradient_checkpointing \
--dataloader_num_workers 4 \
--mixed_precision bf16

# Use 8-bit optimizer for memory efficiency
--use_8bit_adam
```

#### 4. Text Encoder Training Issues

```bash
# Use lower learning rate for text encoders
--train_text_encoder --text_encoder_lr 1e-5

# Or disable if not needed
# (remove --train_text_encoder flag)
```

### Performance Optimization Guide

#### Memory Optimization

-   **Gradient Checkpointing**: Reduces memory by ~50% with minimal speed impact
-   **Mixed Precision**: Use `bf16` for RTX 30/40 series, `fp16` for older GPUs
-   **Batch Size**: Start with 1-2 and increase based on available VRAM
-   **Resolution**: Train at 512px first, then fine-tune at 1024px

#### Speed Optimization

-   **DataLoader Workers**: Set to number of CPU cores / 4
-   **Gradient Accumulation**: Use instead of large batch sizes
-   **8-bit Optimizers**: Reduce memory with minimal accuracy loss

## 🤝 Contributing

Contributions are welcome! We appreciate all forms of contributions including bug reports, feature requests, documentation improvements, and code contributions.

### How to Contribute

1. **Fork the repository** and create your feature branch
2. **Make your changes** with clear, descriptive commits
3. **Add tests** for any new functionality
4. **Update documentation** as needed
5. **Submit a Pull Request** with a clear description

### Development Setup

```bash
# Clone the repository
git clone https://github.com/seochan99/stable-diffusion-3.5-text2image-lora.git
cd stable-diffusion-3.5-text2image-lora

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt  # If available

# Install pre-commit hooks (optional)
pre-commit install
```

### Contributing Guidelines

-   Follow PEP 8 style guidelines
-   Add docstrings to all functions and classes
-   Write meaningful commit messages
-   Test your changes thoroughly
-   Update README if adding new features

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-   [Stability AI](https://stability.ai/) for Stable Diffusion 3.5
-   [Hugging Face](https://huggingface.co/) for the Diffusers library
-   [Microsoft](https://github.com/microsoft/LoRA) for the LoRA technique
-   The open-source community for continuous improvements

## 📧 Contact

For questions, issues, or collaboration opportunities:

-   **Email**: <gmlcks00513@gmail.com>
-   **GitHub Issues**: [Create an issue](https://github.com/seochan99/stable-diffusion-3.5-text2image-lora/issues)
-   **Discussions**: [GitHub Discussions](https://github.com/seochan99/stable-diffusion-3.5-text2image-lora/discussions)

---

⭐ **Star this repository if it helped you!** ⭐
