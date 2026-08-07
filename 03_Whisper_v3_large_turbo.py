"""Whisper V3-Large Turbo.ipynb"""

!pip install -q jiwer evaluate
!pip install -qU accelerate
!pip install -q transformers[torch]
!pip install -q peft soundfile
!pip install -q unsloth transformers datasets accelerate bitsandbytes peft librosa
!pip install --upgrade transformers
!pip install -q jiwer evaluate accelerate transformers[torch] peft bitsandbytes soundfile datasets
!git clone https://github.com/sunbirdai/salt.git
!pip install -qr salt/requirements.txt

import gc
import os
import unicodedata
import re

import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Any, Dict, List, Union

gc.collect()
torch.cuda.empty_cache()

pretrained_model = "openai/whisper-large-v3-turbo"

feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(pretrained_model)
processor = transformers.WhisperProcessor.from_pretrained(
    pretrained_model, language="sw", task="transcribe"
)

model = transformers.WhisperForConditionalGeneration.from_pretrained(
    pretrained_model,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)

# Pad token setup
processor.tokenizer.pad_token    = processor.tokenizer.eos_token
processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id                  = processor.tokenizer.pad_token_id
model.generation_config.pad_token_id       = processor.tokenizer.pad_token_id

# Generation config for Kikuyu/Swahili transcription
model.generation_config.language     = "sw"
model.generation_config.task         = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids            = None
model.generation_config.suppress_tokens   = []
model.config.use_cache = False      # required during training

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = model.to(device)

print(f"Model loaded on {device}  |  dtype: {model.dtype}  |  "
      f"params: {sum(p.numel() for p in model.parameters()):,}")

from huggingface_hub import notebook_login
notebook_login()

from datasets import load_dataset, Audio, Dataset

BATCH_START = 0
BATCH_SIZE  = 10000

print(f"Loading rows {BATCH_START} → {BATCH_START + BATCH_SIZE} …")

stream = load_dataset(
    "Anv-ke/kikuyu", split="train", streaming=True, trust_remote_code=True
)
stream = stream.skip(BATCH_START).take(BATCH_SIZE)
batch_dataset = Dataset.from_generator(lambda: stream)
batch_dataset = batch_dataset.cast_column("audio", Audio(sampling_rate=16000))

split    = batch_dataset.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
valid_ds = split["test"]

gc.collect()
print(f" Loaded  —  train: {len(train_ds)}  |  valid: {len(valid_ds)} ")
print(f"Sample: {train_ds[0]['transcription'][:100]}")

def get_duration(example):
    try:
        arr  = example["audio"]["array"]
        rate = example["audio"]["sampling_rate"]
        return {"duration": len(arr) / rate}
    except RuntimeError as e:
        print(f"Duration error: {e}")
        return {"duration": -1}

print("Computing durations …")
train_ds = train_ds.map(get_duration)
valid_ds = valid_ds.map(get_duration)

train_ds = train_ds.sort("duration")
valid_ds = valid_ds.sort("duration")

train_ds = train_ds.filter(lambda x: 0.5 <= x["duration"] <= 15.0)
valid_ds = valid_ds.filter(lambda x: 0.5 <= x["duration"] <= 15.0)

print(f"After filtering  —  train: {len(train_ds)}  |  valid: {len(valid_ds)}")
print(f"Clip range: {train_ds[0]['duration']:.2f}s  →  {train_ds[-1]['duration']:.2f}s")

DIACRITIC_NORMALISATION_MAP: Dict[str, str] = {
    # macrons (long vowels) 
    "\u0101": "\u0101",   # ā  (already NFC, kept for explicitness)
    "\u0113": "\u0113",   # ē
    "\u012b": "\u012b",   # ī
    "\u014d": "\u014d",   # ō
    "\u016b": "\u016b",   # ū
    # upper-case counterparts
    "\u0100": "\u0100",   # Ā
    "\u0112": "\u0112",   # Ē
    "\u012a": "\u012a",   # Ī
    "\u014c": "\u014c",   # Ō
    "\u016a": "\u016a",   # Ū
    # any other variant spellings seen in the corpus go here 
    # e.g. "a\u0304": "\u0101",   # a + combining macron  → ā
}

def normalize_diacritics(text: str) -> str:
    """
    Normalise diacritics in *text* to a single canonical Unicode form.

    Steps:
      1. Apply Unicode NFC composition so that decomposed sequences
         (base char + combining mark) are collapsed into precomposed code-points.
      2. Apply any corpus-specific remappings from DIACRITIC_NORMALISATION_MAP.
      3. Strip or replace any residual combining marks that are not part of the
         accepted Kikuyu orthography (optional; controlled by STRIP_UNKNOWN_COMBINING).

    This guarantees that e.g. 'ā' written as U+0101 and 'ā' written as
    U+0061+U+0304 are both tokenised identically.
    """
    # Step 1 – NFC composition (most important step)
    text = unicodedata.normalize("NFC", text)

    # Step 2 – explicit remap (handles any residual edge-cases)
    for src, tgt in DIACRITIC_NORMALISATION_MAP.items():
        text = text.replace(src, tgt)
 return text


# Quick smoke-test: both inputs must produce the same output
_nfd = unicodedata.normalize("NFD", "māhiga")  # a + combining macron
_nfc = "māhiga"                                 # precomposed ā
assert normalize_diacritics(_nfd) == normalize_diacritics(_nfc), \
    "Diacritic normalisation failed: NFD and NFC forms not unified!"
print(" Diacritic normalisation smoke-test passed. ")

def prepare_dataset(example):
    audio_array = example["audio"]["array"]
    sample_rate  = example["audio"]["sampling_rate"]

    if not isinstance(audio_array, np.ndarray):
        audio_array = np.array(audio_array, dtype=np.float32)
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    input_features = feature_extractor(
        audio_array, sampling_rate=sample_rate
    ).input_features[0]

    # Apply diacritic normalisation before tokenisation
    transcription = normalize_diacritics(example["transcription"])

    tokenized = processor.tokenizer(
        transcription,
        truncation=True,
        max_length=448
    )

    return {
        "input_features": input_features,
        "labels": tokenized.input_ids,
        "source.language": "kik",
        "target.language": "kik",
    }

print("Preprocessing training data …")
train_data = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names, num_proc=None)

print("Preprocessing validation data …")
val_data   = valid_ds.map(prepare_dataset, remove_columns=valid_ds.column_names, num_proc=None)

print(f"Preprocessing done  —  train: {len(train_data)}  |  valid: {len(val_data)}")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
print("Data collator ready.")

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model.enable_input_require_grads()
model = get_peft_model(model, lora_config)
model.config.use_cache = False

print("LoRA applied!")
model.print_trainable_parameters()

class WhisperCTCAuxTrainer(transformers.Seq2SeqTrainer):
    """Seq2SeqTrainer that adds a CTC auxiliary loss on the encoder output."""

    def __init__(self, *args, ctc_weight: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctc_weight = ctc_weight

        # Build the CTC head and move it to the same device as the model.
        # vocab_size is read from the base model config (accessible through PEFT wrapper).
        vocab_size = self.model.config.vocab_size
        d_model    = self.model.config.d_model

        self.ctc_head    = nn.Linear(d_model, vocab_size).to(self.args.device)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        print(
            f"CTC auxiliary head: d_model={d_model}  vocab={vocab_size}  "
            f"λ={ctc_weight}"
        )

def _get_encoder_hidden(self, model, input_features):
        """
        Return the encoder's last hidden-state tensor [B, T, D].

        Works for both the raw PEFT wrapper and any nested attribute layout
        that Hugging Face may use depending on the version.
        """
        # Try the most common attribute paths in order.
        for attr_path in [
            "base_model.model.model.encoder",   # PEFT-wrapped WhisperForConditionalGeneration
            "base_model.model.encoder",          # some PEFT configs
            "model.encoder",                     # unwrapped
        ]:
            obj = model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                return obj(input_features).last_hidden_state
            except AttributeError:
                continue

        raise AttributeError(
            "Could not locate Whisper encoder through the model wrapper. "
            "Please update _get_encoder_hidden() for your transformers version."
        )

def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")   # shape [B, L], padded with -100

        # Standard seq2seq (cross-entropy) loss
        outputs       = model(**inputs)
        seq2seq_loss  = outputs.loss

        # CTC auxiliary loss 
        if labels is not None and self.ctc_weight > 0:
            try:
                # Encoder hidden states  [B, T, D]
                encoder_hidden = self._get_encoder_hidden(
                    model, inputs["input_features"]
                )

                # Project to vocabulary  [B, T, V]  →  [T, B, V] for CTCLoss
                ctc_logits = self.ctc_head(encoder_hidden.float())
                log_probs  = ctc_logits.log_softmax(-1).permute(1, 0, 2)

                B   = encoder_hidden.size(0)
                T   = encoder_hidden.size(1)
                dev = encoder_hidden.device

                # All frames are valid (no padding on the encoder side after
                # Whisper's fixed-length mel extraction)
                input_lengths = torch.full((B,), T, dtype=torch.long, device=dev)

                # Target lengths: count non-(-100) positions per sample
                valid_mask     = labels != -100
                target_lengths = valid_mask.sum(dim=1).to(dev)

                # Replace -100 with 0 (blank index) so CTCLoss doesn't crash
                targets = labels.masked_fill(~valid_mask, 0).to(dev)

                ctc_loss  = self.ctc_loss_fn(
                    log_probs, targets, input_lengths, target_lengths
                )
                total_loss = seq2seq_loss + self.ctc_weight * ctc_loss

                # Log both loss components so they appear in TensorBoard / wandb
                if self.state.global_step % self.args.logging_steps == 0:
                    self.log({
                        "loss/seq2seq": seq2seq_loss.item(),
                        "loss/ctc":     ctc_loss.item(),
                        "loss/total":   total_loss.item(),
                    })

            except Exception as exc:
                # CTC is auxiliary — never let it crash the whole training run.
                print(f"[WARN] CTC loss skipped this step: {exc}")
                total_loss = seq2seq_loss
        else:
            total_loss = seq2seq_loss

        return (total_loss, outputs) if return_outputs else total_loss

import evaluate
import salt.metrics

compute_metrics = salt.metrics.multilingual_eval_fn(
    val_data,
    [evaluate.load("wer"), evaluate.load("cer")],
    processor.tokenizer,
    log_first_N_predictions=5,
    speech_processor=processor,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()

output_dir = "./whisper-kikuyu-lora-finetuned"

training_args = transformers.Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=50,
    max_steps=500,
    weight_decay=0.01,
    gradient_checkpointing=True,
    bf16=True,
    eval_strategy="steps",
    eval_steps=50,
    predict_with_generate=True,
    generation_max_length=100,
    save_steps=50,
    save_total_limit=3,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_num_workers=2,
    remove_unused_columns=True,
    report_to="tensorboard",
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=5,
    early_stopping_threshold=0.0,
)

trainer = WhisperCTCAuxTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    processing_class=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
    ctc_weight=0.3,   # ← tune this: 0.1 (conservative) … 0.5 (aggressive)
)

print(" Starting training … ")
print(f"   Batch: rows {BATCH_START} → {BATCH_START + BATCH_SIZE}")
print(f"   Training samples:   {len(train_data)}")
print(f"   Validation samples: {len(val_data)}")

trainer.train()
print(" Training complete! ")
