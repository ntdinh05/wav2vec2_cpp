#!/usr/bin/env python3
"""Fine-tune wav2vec2 (from UXTD checkpoint) on TaL80 adult speech corpus."""

import gc
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import wandb
from datasets import Dataset, DatasetDict
from evaluate import load as load_metric
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
)

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
MODEL_DIR = "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2-uxtd-finetuned"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint-784")
CSV_PATH = "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wave2vec2_cpp/output/tal80_utterances_by_length.csv"
SPEAKERS_CSV = "/home/ultraspeech-dev/ultrasuite/TaL80/doc/speakers.csv"
OUTPUT_DIR = "./wav2vec2-tal80-finetuned"
TARGET_SR = 16000
SOURCE_SR = 48000  # TaL80 audio is 48kHz
MAX_AUDIO_SEC = 10  # truncate audio longer than this to fit in GPU memory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(OUTPUT_DIR):
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, "train_tal80.log")

logger = logging.getLogger("finetune_wav2vec2_tal80")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_formatter)
_file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
_file_handler.setFormatter(_formatter)

logger.addHandler(_stream_handler)
logger.addHandler(_file_handler)


def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = _log_uncaught_exception
logger.info("Logger initialized.")

# ──────────────────────────────────────────────────────────────────────
# CUDA check
# ──────────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This script requires a GPU to train. "
        "Check your PyTorch installation with: python -c 'import torch; print(torch.cuda.is_available())'"
    )
logger.info(
    "Using GPU: %s (CUDA %s)", torch.cuda.get_device_name(0), torch.version.cuda
)
logger.info("Log file: %s", os.path.abspath(LOG_PATH))


def log_cuda_memory(stage: str):
    if not torch.cuda.is_available():
        return
    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
    reserved_gb = torch.cuda.memory_reserved() / (1024**3)
    logger.info(
        "[CUDA] %s | allocated=%.2f GB reserved=%.2f GB",
        stage,
        allocated_gb,
        reserved_gb,
    )


# Speaker split ratio: ~70% train, ~15% dev, ~15% test (by speaker)
TRAIN_RATIO = 0.70
DEV_RATIO = 0.15

# ──────────────────────────────────────────────────────────────────────
# 1. Load CSV and create speaker splits
# ──────────────────────────────────────────────────────────────────────
logger.info("Loading utterance CSV...")
df = pd.read_csv(CSV_PATH)

# Derive wav paths from the txt filepath column
df["wav_path"] = df["filepath"].str.replace(r"\.txt$", ".wav", regex=True)

# Verify all wav files exist
missing = df[~df["wav_path"].apply(os.path.exists)]
if len(missing) > 0:
    logger.warning("%d wav files not found. Dropping them.", len(missing))
    df = df[df["wav_path"].apply(os.path.exists)].reset_index(drop=True)
logger.info("Total utterances with audio: %d", len(df))

# Create speaker-level train/dev/test splits
# Sort speakers for reproducibility, then split by ratio
speakers = sorted(df["speaker"].unique())
np.random.seed(42)
np.random.shuffle(speakers)

n_train = int(len(speakers) * TRAIN_RATIO)
n_dev = int(len(speakers) * DEV_RATIO)

train_speakers = set(speakers[:n_train])
dev_speakers = set(speakers[n_train : n_train + n_dev])
test_speakers = set(speakers[n_train + n_dev :])

logger.info(
    "Speaker splits: %d train, %d dev, %d test",
    len(train_speakers),
    len(dev_speakers),
    len(test_speakers),
)


def assign_split(speaker):
    if speaker in train_speakers:
        return "train"
    elif speaker in dev_speakers:
        return "dev"
    else:
        return "test"


df["split"] = df["speaker"].map(assign_split)
logger.info("Split distribution:\n%s", df["split"].value_counts().to_string())

# ──────────────────────────────────────────────────────────────────────
# W&B init (after splits are computed so we can log dataset stats)
# ──────────────────────────────────────────────────────────────────────
wandb.init(
    project="wav2vec2-tal80-finetune",
    name="tal80-from-uxtd-ckpt784",
    config={
        "base_checkpoint": "wav2vec2-uxtd-finetuned/checkpoint-784",
        "target_sr": TARGET_SR,
        "source_sr": SOURCE_SR,
        "max_audio_sec": MAX_AUDIO_SEC,
        "train_ratio": TRAIN_RATIO,
        "dev_ratio": DEV_RATIO,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "num_speakers": len(speakers),
        "num_train_speakers": len(train_speakers),
        "num_dev_speakers": len(dev_speakers),
        "num_test_speakers": len(test_speakers),
    },
)

# Log dataset split sizes
split_counts = df["split"].value_counts().to_dict()
for split_name, count in split_counts.items():
    wandb.config.update({f"num_{split_name}_utterances": count})

# Log audio duration distribution per split (from source SR, pre-resample)
logger.info("Computing audio duration stats for W&B...")
durations = []
for _, row in df.iterrows():
    try:
        info = sf.info(row["wav_path"])
        durations.append({
            "split": row["split"],
            "speaker": row["speaker"],
            "duration_sec": info.duration,
        })
    except Exception:
        pass
dur_df = pd.DataFrame(durations)

for split_name in ["train", "dev", "test"]:
    split_durs = dur_df[dur_df["split"] == split_name]["duration_sec"]
    if len(split_durs) == 0:
        continue
    wandb.config.update({
        f"{split_name}_total_hours": round(split_durs.sum() / 3600, 2),
        f"{split_name}_mean_dur_sec": round(split_durs.mean(), 2),
        f"{split_name}_max_dur_sec": round(split_durs.max(), 2),
    })

# Log duration histogram
wandb.log({
    "dataset/duration_histogram": wandb.Histogram(dur_df["duration_sec"].values, num_bins=50),
    "dataset/total_hours": round(dur_df["duration_sec"].sum() / 3600, 2),
})

# Drop utterances longer than MAX_AUDIO_SEC to avoid CTC label/audio mismatch.
# Audio gets truncated to MAX_AUDIO_SEC but labels stay full-length, causing
# label_length > input_length which makes CTC loss return inf.
logger.info("Filtering utterances longer than %ds...", MAX_AUDIO_SEC)
keep_mask = []
for wav_path in df["wav_path"]:
    info = sf.info(wav_path)
    keep_mask.append(info.duration <= MAX_AUDIO_SEC)
n_before = len(df)
df = df[keep_mask].reset_index(drop=True)
logger.info("Dropped %d utterances > %ds. Remaining: %d", n_before - len(df), MAX_AUDIO_SEC, len(df))

# ──────────────────────────────────────────────────────────────────────
# 2. Convert text prompts to IPA phonemes
# ──────────────────────────────────────────────────────────────────────
logger.info("Converting text to IPA phonemes...")
from phonemizer import phonemize
from phonemizer.separator import Separator

unique_utterances = df["utterance"].unique().tolist()
phoneme_map = {}
# Batch phonemize with phone-level separators
# phone=' ' separates individual phonemes, word='  ' (double space) separates words
phonemized = phonemize(
    unique_utterances,
    language="en-us",
    backend="espeak",
    strip=True,
    with_stress=False,
    language_switch="remove-flags",
    separator=Separator(phone=" ", word="  ", syllable=""),
)
for text, phones in zip(unique_utterances, phonemized):
    phoneme_map[text] = phones

df["phonemes"] = df["utterance"].map(phoneme_map)

# Drop any rows where phonemization failed or produced empty result
df = df[df["phonemes"].str.strip().str.len() > 0].reset_index(drop=True)
logger.info(
    "Sample phonemization: '%s' -> '%s'",
    df["utterance"].iloc[0],
    df["phonemes"].iloc[0],
)

# ──────────────────────────────────────────────────────────────────────
# 3. Load processor and model from UXTD checkpoint
# ──────────────────────────────────────────────────────────────────────
logger.info("Loading processor from %s, model from %s...", MODEL_DIR, CHECKPOINT_PATH)
# Load feature extractor and tokenizer separately, then combine into processor.
# The saved processor_config.json uses a nested format incompatible with newer transformers.
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(MODEL_DIR)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
# Reset backend to a plain string so save_pretrained() can JSON-serialize it.
# (Importing phonemizer replaces the string with a live EspeakBackend object.)
tokenizer.backend = "espeak"
model = Wav2Vec2ForCTC.from_pretrained(CHECKPOINT_PATH)

# The space token (id=392) exceeds vocab_size=392 in config. Fix by updating config.
actual_vocab_size = len(processor.tokenizer.get_vocab())  # 393
if model.config.vocab_size < actual_vocab_size:
    model.config.vocab_size = actual_vocab_size
    # Resize the lm_head to match
    old_weight = model.lm_head.weight.data
    old_bias = model.lm_head.bias.data
    model.lm_head = torch.nn.Linear(old_weight.shape[1], actual_vocab_size)
    model.lm_head.weight.data[: old_weight.shape[0]] = old_weight
    model.lm_head.bias.data[: old_bias.shape[0]] = old_bias
    logger.info("Resized lm_head from %d to %d", old_weight.shape[0], actual_vocab_size)

# Freeze the feature encoder (CNN layers)
model.freeze_feature_encoder()
logger.info("Feature encoder frozen.")

# Log model parameter stats
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
wandb.config.update({
    "total_params_M": round(total_params / 1e6, 1),
    "trainable_params_M": round(trainable_params / 1e6, 1),
    "frozen_params_M": round(frozen_params / 1e6, 1),
    "pct_trainable": round(100 * trainable_params / total_params, 1),
    "vocab_size": actual_vocab_size,
})
logger.info(
    "Model params: %.1fM total, %.1fM trainable (%.1f%%)",
    total_params / 1e6, trainable_params / 1e6, 100 * trainable_params / total_params,
)

# ──────────────────────────────────────────────────────────────────────
# 4. Build HuggingFace datasets
# ──────────────────────────────────────────────────────────────────────
logger.info("Building datasets...")

# Build phoneme-to-id mapping from the vocab
vocab = processor.tokenizer.get_vocab()
unk_id = vocab.get("<unk>", 3)
space_id = vocab.get(" ", len(vocab) - 1)  # word separator token


def phonemes_to_ids(phoneme_str):
    """Convert space-separated phoneme string to token IDs.
    Single space = phoneme boundary, double space = word boundary."""
    ids = []
    words = phoneme_str.split("  ")  # split by double space (word boundary)
    for wi, word in enumerate(words):
        if wi > 0:
            ids.append(space_id)  # word separator
        phones = word.split(" ")
        for phone in phones:
            if phone == "":
                continue
            if phone in vocab:
                ids.append(vocab[phone])
            else:
                ids.append(unk_id)
    return ids


# Verify tokenization works
sample_phones = df["phonemes"].iloc[0]
sample_ids = phonemes_to_ids(sample_phones)
decoded = processor.tokenizer.decode(sample_ids)
logger.info(
    "Tokenization check: '%s' -> %d tokens -> '%s'",
    sample_phones,
    len(sample_ids),
    decoded,
)


def prepare_labels(row):
    """Pre-tokenize phoneme labels only (cheap, small). Audio loaded lazily."""
    return {"labels": phonemes_to_ids(row["phonemes"])}


# Create train/dev/test datasets — only pre-compute label IDs, not audio
datasets = {}
for split_name in ["train", "dev", "test"]:
    split_df = df[df["split"] == split_name].reset_index(drop=True)
    if len(split_df) == 0:
        continue
    ds = Dataset.from_pandas(split_df[["wav_path", "phonemes"]])
    ds = ds.map(
        prepare_labels,
        remove_columns=["phonemes"],
        num_proc=1,
        desc=f"Tokenizing {split_name}",
    )
    datasets[split_name] = ds
    logger.info("  %s: %d examples", split_name, len(ds))

dataset_dict = DatasetDict(datasets)

# Log label length distribution and phoneme frequency to W&B
if "train" in datasets:
    train_label_lens = [len(ex["labels"]) for ex in datasets["train"]]
    wandb.log({
        "dataset/train_label_len_histogram": wandb.Histogram(train_label_lens, num_bins=40),
        "dataset/train_mean_label_len": np.mean(train_label_lens),
    })

    # Phoneme frequency in training set
    id_to_phone = {v: k for k, v in vocab.items()}
    phone_counts = {}
    for ex in datasets["train"]:
        for tid in ex["labels"]:
            phone = id_to_phone.get(tid, f"id_{tid}")
            phone_counts[phone] = phone_counts.get(phone, 0) + 1
    # Log as a bar chart table
    phone_table = wandb.Table(
        columns=["phoneme", "count"],
        data=sorted(phone_counts.items(), key=lambda x: -x[1]),
    )
    wandb.log({"dataset/phoneme_distribution": phone_table})
    logger.info("Unique phonemes in training labels: %d", len(phone_counts))


# ──────────────────────────────────────────────────────────────────────
# 5. Data collator
# ──────────────────────────────────────────────────────────────────────
@dataclass
class DataCollatorCTCWithPadding:
    """Collator that loads audio lazily from wav_path at batch time."""

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Load and resample audio on the fly
        input_features = []
        for f in features:
            audio, sr = sf.read(f["wav_path"])
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            inputs = self.processor(audio, sampling_rate=TARGET_SR, return_tensors=None)
            input_features.append({"input_values": inputs.input_values[0]})

        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding with -100 for CTC loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor)

# ──────────────────────────────────────────────────────────────────────
# 6. Evaluation metric (Phoneme Error Rate = WER on phoneme sequences)
# ──────────────────────────────────────────────────────────────────────
wer_metric = load_metric("wer")


def preprocess_logits_for_eval(logits, labels):
    """Reduce logits to argmax IDs before accumulation to avoid OOM."""
    return logits.argmax(dim=-1)


def compute_metrics(pred):
    pred_ids = pred.predictions  # already argmax'd by preprocess_logits_for_eval

    # Replace -100 in labels (padding) with pad_token_id
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    per = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"per": per}


# ──────────────────────────────────────────────────────────────────────
# 6b. Custom W&B callback for richer logging
# ──────────────────────────────────────────────────────────────────────
NUM_PREDICTION_SAMPLES = 10  # number of examples to log predictions for each epoch


class WandbPredictionCallback(TrainerCallback):
    """Logs sample predictions, GPU memory, and gradient norms to W&B."""

    def __init__(self, eval_dataset, processor, num_samples=NUM_PREDICTION_SAMPLES):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.num_samples = min(num_samples, len(eval_dataset))
        # Pick fixed sample indices for consistent comparison across epochs
        self.sample_indices = list(range(self.num_samples))

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Log sample predictions as a W&B table after each evaluation."""
        if model is None:
            return

        epoch = state.epoch or 0

        # Log GPU memory
        if torch.cuda.is_available():
            wandb.log({
                "gpu/allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu/reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "gpu/max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            }, step=state.global_step)

        # Build prediction table
        model.eval()
        table = wandb.Table(columns=[
            "epoch", "index", "ground_truth", "prediction", "label_len", "pred_len",
        ])

        for idx in self.sample_indices:
            sample = self.eval_dataset[idx]

            # Load audio
            audio, sr = sf.read(sample["wav_path"])
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

            inputs = self.processor(
                audio, sampling_rate=TARGET_SR, return_tensors="pt"
            )
            input_values = inputs.input_values.to(model.device)

            with torch.no_grad():
                logits = model(input_values).logits

            pred_ids = logits.argmax(dim=-1)[0].cpu().numpy()
            pred_str = self.processor.decode(pred_ids)

            label_ids = [i for i in sample["labels"] if i != -100]
            label_str = self.processor.tokenizer.decode(label_ids)

            table.add_data(
                round(epoch, 2), idx, label_str, pred_str,
                len(label_ids), len(pred_ids),
            )

        wandb.log({
            "predictions/samples": table,
        }, step=state.global_step)

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """Log gradient norm and learning rate explicitly."""
        if logs is None:
            return

        extra = {}
        if "grad_norm" in logs:
            extra["train/grad_norm"] = logs["grad_norm"]
        if "learning_rate" in logs:
            extra["train/learning_rate"] = logs["learning_rate"]
        if "loss" in logs:
            extra["train/loss"] = logs["loss"]

        if extra:
            wandb.log(extra, step=state.global_step)


# ──────────────────────────────────────────────────────────────────────
# 7. Training arguments
# ──────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=30,
    learning_rate=3e-5,  # lowered: 1e-4 caused fp16 overflow -> NaN
    max_grad_norm=1.0,  # clip large gradients to prevent inf/NaN
    warmup_steps=300,
    gradient_checkpointing=True,
    bf16=True,  # bf16 instead of fp16: CTC loss (~200) overflows fp16 max (65504) during backprop
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="per",
    greater_is_better=False,
    save_total_limit=3,
    dataloader_num_workers=2,
    report_to="wandb",
    remove_unused_columns=False,
    logging_dir=os.path.join(OUTPUT_DIR, "hf_logs"),
    logging_nan_inf_filter=False,  # surface NaN/Inf losses instead of silently filtering
    include_num_input_tokens_seen=True,
)

# ──────────────────────────────────────────────────────────────────────
# 8. Trainer
# ──────────────────────────────────────────────────────────────────────
eval_ds = dataset_dict.get("dev", dataset_dict.get("test"))
wandb_callback = WandbPredictionCallback(
    eval_dataset=eval_ds,
    processor=processor,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=eval_ds,
    data_collator=data_collator,
    processing_class=processor,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_eval,
    callbacks=[wandb_callback],
)

# ──────────────────────────────────────────────────────────────────────
# 9. Train!
# ──────────────────────────────────────────────────────────────────────
logger.info("Starting training...")
log_cuda_memory("before-train")
try:
    train_result = trainer.train()
    logger.info("Training completed. global_step=%s", train_result.global_step)
except Exception:
    logger.exception("Training failed during trainer.train()")
    raise

# Save the final model (trainer already has the fully-trained model)
log_cuda_memory("after-train-before-final-save")
torch.cuda.empty_cache()
gc.collect()

try:
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    logger.info("Model saved to %s", OUTPUT_DIR)
except Exception:
    logger.exception("Failed while saving final model/processor")
    raise

# ──────────────────────────────────────────────────────────────────────
# 10. Evaluate on test set
# ──────────────────────────────────────────────────────────────────────
if "test" in dataset_dict:
    # Free optimizer/scheduler state before eval to reclaim memory
    trainer.optimizer = None
    trainer.lr_scheduler = None
    gc.collect()
    torch.cuda.empty_cache()
    log_cuda_memory("before-test-eval")

    logger.info("Evaluating on test set...")
    try:
        results = trainer.evaluate(dataset_dict["test"])
        test_per = results.get("eval_per")
        if test_per is not None:
            logger.info("Test PER: %.4f", test_per)
            wandb.log({"test/per": test_per})
        else:
            logger.warning("`eval_per` not found in evaluation results: %s", results)
    except Exception:
        logger.exception("Failed during final test evaluation")
        raise

wandb.finish()
