# MACHINE-LEARNING-MODELS-TRAINING

A curated collection of end-to-end machine learning pipelines, scripts, and model-training notebooks across various data domains, including tabular data, speech, and predictive analytics. Optimized for performance and reproducible training workflows.

---

## 📂 Repository Structure

* **`whisper_v3_large_turbo.py`**: Parameter-efficient automatic speech recognition (ASR) model fine-tuning pipeline.
* *(Add your other model notebooks here as you upload them!)*

---

## 🎙️ Project Feature: Whisper Large-v3-Turbo Fine-Tuning for Kikuyu

An end-to-end pipeline to fine-tune OpenAI's `whisper-large-v3-turbo` on the Kikuyu language using LoRA (Low-Rank Adaptation) and a custom Connectionist Temporal Classification (CTC) auxiliary loss.

### 🚀 Features & Architecture

* **Model Base:** `openai/whisper-large-v3-turbo`
* **Target Language:** Kikuyu (`kik`) with Swahili fallback tokenizer parameters configured for initial vocabulary matching.
* **Parameter-Efficient Tuning:** Uses Hugging Face `peft` with **LoRA** targeting `q_proj` and `v_proj` layers to minimize VRAM footprint.
* **Dual-Loss Optimization:** Features a custom `WhisperCTCAuxTrainer` that introduces a **CTC Auxiliary Loss** ($\lambda = 0.3$) applied to the encoder hidden states, significantly accelerating low-resource speech recognition convergence.
* **Robust Preprocessing:** Implements Unicode Normalization (NFC) and custom diacritic mapping tables specific to orthographic representations of Kikuyu vowels.

### 📊 Dataset Configuration

The training pipeline consumes the streaming dataset `Anv-ke/kikuyu` from Hugging Face:
* **Chunk Windowing:** Configured by default to process chunks of 10,000 samples.
* **Train/Test Split:** 90% Training / 10% Validation split with standard deterministic seeding.
* **Duration Filtering:** Keeps audio clips strictly bounded between 0.5 seconds and 15.0 seconds.
* **Audio Features:** Resampled dynamically to 16 kHz mono-channel arrays.

### ⚙️ Training Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Learning Rate** | `1e-4` | Peak learning rate with AdamW |
| **Warmup Steps** | `50` | Linear warmup length |
| **Max Steps** | `500` | Total training iterations |
| **Per-Device Batch Size** | `4` | Batch size handled simultaneously per device |
| **Gradient Accumulation** | `2` | Number of updates to accumulate before backprop (Effective batch size of 8) |
| **Mixed Precision** | `fp16` | Half-precision floating-point format training activated |
| **LoRA Rank (r)** | `32` | Dimensional factor for low-rank matrices |
| **LoRA Alpha ($\alpha$)** | `64` | Scaling factor for adapter weights |

---

## 🛠️ Requirements & Environment

Dependencies vary by project module. For general deep learning and speech modeling components, ensure your environment has access to:
* `torch`, `transformers`, `peft`, `datasets`, `accelerate`, `bitsandbytes`, `jiwer`, `librosa`, `soundfile`, `evaluate`, `unsloth`


