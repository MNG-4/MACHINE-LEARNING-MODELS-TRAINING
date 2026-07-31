# MACHINE-LEARNING-MODELS-TRAINING

A curated collection of end-to-end machine learning pipelines, engineering scripts, and model-training workflows across various data domains, including speech processing, audio pipeline engineering, and predictive analytics. Optimized for reproducible workflows.

---

## Repository Structure & Project Modules

### 1. Audio Pipeline & Preprocessing Tools

* **`audio_splitter_3.py`**: An advanced Voice Activity Detection (VAD) audio-splitting pipeline. It leverages `silero-vad` to dynamically segment long audio tracks into optimal training lengths ($<30$ seconds). Features a back-scanning energy search to intelligently cut audio at natural silence gaps/word boundaries instead of blind hard cuts.
* **`Audio_Cleaner.ipynb`**: A comprehensive audio preprocessing and noise reduction utility designed to strip background static, balance audio levels, and isolate speech signals to create clean training datasets.
* **`EDA on ANV Dataset.ipynb`**: An exploratory data analysis notebook specifically evaluating speech audio and transcript metrics. It handles audio quality assessments (clipping rates, silence ratios, RMS loudness) and uncovers orthographic inconsistencies like severe diacritic variations in low-resource language transcripts.

### 2. Model Training Pipelines

* **`whisper_v3_large_turbo.py`**: An end-to-end pipeline to fine-tune OpenAI's `whisper-large-v3-turbo` on low-resource African languages (e.g., Kikuyu) using parameter-efficient **LoRA** framework. It features a custom `WhisperCTCAuxTrainer` which implements an auxiliary Connectionist Temporal Classification (CTC) loss ($\lambda = 0.3$) onto the encoder hidden states to drastically speed up low-resource convergence.

---

## Project Feature Deep-Dives

### Module A: Whisper Large-v3-Turbo Fine-Tuning
* **Model Base:** `openai/whisper-large-v3-turbo`
* **Tuning Strategy:** Hugging Face `peft` with LoRA matrices applied to attention project projections (`q_proj`, `v_proj`).
* **Hyperparameters:** Max Steps: `500` | Warmup: `50` | Batch Size: `4` per device (Effective `8` via grad accumulation) | Mixed Precision: `fp16`.

### Module B: Energy-Based VAD Audio Splitter (`audio_splitter_3.py`)
* **Frameworks:** `torch`, `torchaudio`, `pydub`, `soundfile`
* **Mechanic:** Instead of standard absolute slicing, it checks a configurable 5-second window backward from the maximum length boundary to pinpoint the quietest 100ms window (word/sentence boundaries), minimizing broken or truncated mid-word segments.

---

## General Requirements & Dependencies

To set up the complete suite across all audio preprocessing, analytics, and modeling tasks, install the collective environment dependencies:

```bash
pip install torch torchaudio transformers peft datasets accelerate bitsandbytes jiwer librosa soundfile evaluate unsloth pydub ipywidgets





