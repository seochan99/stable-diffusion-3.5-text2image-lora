# Stable Diffusion 3.5 LoRA Fine-Tuning on Google Colab

This guide describes how to use the `SD35_LoRA_Colab.ipynb` notebook to fine-tune Stable Diffusion 3.5 with LoRA adapters on Google Colab GPUs.

## 📚 Documentation

- **[🎨 Dataset Ideas Guide](docs/DATASET_IDEAS.md)** - What should I train? Get concrete ideas!
- **[🔄 Complete Workflow](docs/WORKFLOW.md)** - End-to-end training & upload process
- **[☁️ Upload to Hugging Face](docs/UPLOAD_TO_HF.md)** - Share your LoRA with the world
- **[📓 Colab Fine-tuning](docs/COLAB_FINETUNING.md)** - Train on Google Colab GPUs

---

## 1. Fork and Clone

1. Fork [`seochan99/stable-diffusion-3.5-text2image-lora`](https://github.com/seochan99/stable-diffusion-3.5-text2image-lora) to your GitHub account.
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/stable-diffusion-3.5-text2image-lora.git
   cd stable-diffusion-3.5-text2image-lora
   ```
3. Copy `SD35_LoRA_Colab.ipynb` into the repository root (already provided in this fork).
4. Commit and push the notebook:
   ```bash
   git add SD35_LoRA_Colab.ipynb docs/COLAB_FINETUNING.md
   git commit -m "Add Colab workflow for SD3.5 LoRA fine-tuning"
   git push origin main
   ```

## 2. Open the Notebook in Colab

1. Navigate to your fork on GitHub.
2. Open `SD35_LoRA_Colab.ipynb` and click the **Open in Colab** badge (or change the URL to `https://colab.research.google.com/github/<user>/stable-diffusion-3.5-text2image-lora/blob/main/SD35_LoRA_Colab.ipynb`).
3. Switch the runtime to **GPU** (`Runtime > Change runtime type > GPU`).

## 3. Run the Notebook Cells

1. **Environment check** – verify GPU with `!nvidia-smi`.
2. **Install dependencies** – install PyTorch (CUDA 12.1 build), Diffusers, Transformers, PEFT, etc.
3. **Hugging Face login** – run the `login()` cell and paste a token with `read` scope.
   - Ensure you have accepted the license for [`stabilityai/stable-diffusion-3.5-medium`](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) with the same account.
4. **Clone the repo** – clones your fork into `/content/stable-diffusion-3.5-text2image-lora` and sets it as the working directory.
5. *(Optional)* **Mount Google Drive** if your dataset or outputs live there.
6. **Dataset setup** – update the paths in the cell defining `DATA_ROOT` and `OUTPUT_DIR`.
   - Upload your pre-captioned dataset (e.g., `prepared_dataset`) into the Colab runtime or Drive.
   - Run the sanity-check cell to ensure `metadata.jsonl` and images are present.
7. **Configure training** – adjust `training_config` and `flag_args` as needed:
   - `train_data_dir` should point to your dataset directory inside Colab.
   - Prefer `mixed_precision = "fp16"` on T4/L4 GPUs; leave as `"bf16"` only when using A100/V100/H100.
   - Enable `train_text_encoder` only if VRAM allows.
8. **Launch training** – set `START_TRAINING = True` and run the cell. Logs stream in-line.
   - Checkpoints and LoRA weights land in `OUTPUT_DIR`.
   - Use TensorBoard cell after `logs/` exists if desired.

## 4. Resume or Adjust Runs

- To resume, set `training_config["resume_from_checkpoint"] = "latest"` or to a specific checkpoint directory.
- Reduce VRAM pressure via `train_batch_size = 1`, higher `gradient_accumulation_steps`, smaller `rank`, or training at `512` resolution initially.

## 5. Run Inference

1. Execute the inference cell at the bottom of the notebook.
2. Set `base_model`, `lora_dir`, and prompts to match your experiment.
3. The generated sample is saved to `<OUTPUT_DIR>/sample_inference.png`.

## 6. Export and Share Results

- Zip the final LoRA directory or push it to the Hugging Face Hub via `StableDiffusion3Pipeline.save_lora_weights` results.
- Commit updated configs or datasets (excluding large binary files) to a new branch and open a PR, if desired.

---
This workflow mirrors the training script in the repository while adding Colab-friendly setup, dataset verification, automated command construction, and inference validation.
