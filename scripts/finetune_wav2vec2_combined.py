#!/usr/bin/env python3
"""Fine-tune wav2vec2 on combined UXTD + TaL80 datasets using PyTorch Lightning + WandB."""

import gc
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
)

import wandb
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
BASE_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"

UXTD_CSV = "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/tests/utterances_by_length.csv"
UXTD_SPEAKERS_PATH = "/home/ultraspeech-dev/ultrasuite/core-uxtd/doc/speakers"
UXTD_SOURCE_SR = 22050

TAL80_CSV = "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/output/tal80_utterances_by_length.csv"
TAL80_SOURCE_SR = 48000

TARGET_SR = 16000
MAX_AUDIO_SEC = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "wav2vec2-combined-finetuned-top-layer")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_DIR, "train_combined.log")

# Training hyperparameters
BATCH_SIZE = 2
GRAD_ACCUM = 8
MAX_EPOCHS = 15
LEARNING_RATE = 1e-3
WARMUP_STEPS = 100
NUM_WORKERS = 2

# TaL80 speaker split ratios
TRAIN_RATIO = 0.70
DEV_RATIO = 0.15

# ──────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger("finetune_combined")
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
        "CUDA is not available. This script requires a GPU. "
        "Check: python -c 'import torch; print(torch.cuda.is_available())'"
    )
torch.set_float32_matmul_precision("medium")
logger.info("Using GPU: %s (CUDA %s)", torch.cuda.get_device_name(0), torch.version.cuda)

# ──────────────────────────────────────────────────────────────────────
# 1. Load both datasets and create speaker splits
# ──────────────────────────────────────────────────────────────────────
logger.info("Loading UXTD utterances...")
uxtd_df = pd.read_csv(UXTD_CSV)
uxtd_df["wav_path"] = uxtd_df["filepath"].str.replace(r"\.txt$", ".wav", regex=True)
uxtd_df["dataset"] = "uxtd"
uxtd_df["source_sr"] = UXTD_SOURCE_SR

# UXTD speaker splits from file
uxtd_speaker_df = pd.read_csv(UXTD_SPEAKERS_PATH, sep="\t")
uxtd_speaker_to_split = dict(zip(uxtd_speaker_df["speaker_id"], uxtd_speaker_df["subset"]))
uxtd_df["split"] = uxtd_df["speaker"].map(uxtd_speaker_to_split)
uxtd_df = uxtd_df.dropna(subset=["split"]).reset_index(drop=True)
# Normalize split names: "dev" -> "dev"
uxtd_df["split"] = uxtd_df["split"].str.strip().str.lower()
logger.info("UXTD: %d utterances, splits: %s", len(uxtd_df), uxtd_df["split"].value_counts().to_dict())

logger.info("Loading TaL80 utterances...")
tal80_df = pd.read_csv(TAL80_CSV)
tal80_df["wav_path"] = tal80_df["filepath"].str.replace(r"\.txt$", ".wav", regex=True)
tal80_df["dataset"] = "tal80"
tal80_df["source_sr"] = TAL80_SOURCE_SR

# TaL80 speaker splits: 70/15/15 by speaker
tal80_speakers = sorted(tal80_df["speaker"].unique())
rng = np.random.RandomState(42)
rng.shuffle(tal80_speakers)

n_train = int(len(tal80_speakers) * TRAIN_RATIO)
n_dev = int(len(tal80_speakers) * DEV_RATIO)
tal80_train_speakers = set(tal80_speakers[:n_train])
tal80_dev_speakers = set(tal80_speakers[n_train : n_train + n_dev])
tal80_test_speakers = set(tal80_speakers[n_train + n_dev :])


def assign_tal80_split(speaker):
    if speaker in tal80_train_speakers:
        return "train"
    elif speaker in tal80_dev_speakers:
        return "dev"
    return "test"


tal80_df["split"] = tal80_df["speaker"].map(assign_tal80_split)
logger.info("TaL80: %d utterances, splits: %s", len(tal80_df), tal80_df["split"].value_counts().to_dict())

# Combine both datasets
df = pd.concat([uxtd_df, tal80_df], ignore_index=True)

# Verify all wav files exist
missing = df[~df["wav_path"].apply(os.path.exists)]
if len(missing) > 0:
    logger.warning("%d wav files not found. Dropping them.", len(missing))
    df = df[df["wav_path"].apply(os.path.exists)].reset_index(drop=True)
logger.info("Combined total utterances with audio: %d", len(df))

# Filter utterances longer than MAX_AUDIO_SEC
logger.info("Filtering utterances longer than %ds...", MAX_AUDIO_SEC)
keep_mask = []
for wav_path in df["wav_path"]:
    try:
        info = sf.info(wav_path)
        keep_mask.append(info.duration <= MAX_AUDIO_SEC)
    except Exception:
        keep_mask.append(False)
n_before = len(df)
df = df[keep_mask].reset_index(drop=True)
logger.info("Dropped %d utterances > %ds. Remaining: %d", n_before - len(df), MAX_AUDIO_SEC, len(df))

# ──────────────────────────────────────────────────────────────────────
# 2. Phonemize text
# ──────────────────────────────────────────────────────────────────────
logger.info("Converting text to IPA phonemes...")
from phonemizer import phonemize
from phonemizer.separator import Separator

unique_utterances = df["utterance"].unique().tolist()
phonemized = phonemize(
    unique_utterances,
    language="en-us",
    backend="espeak",
    strip=True,
    with_stress=False,
    language_switch="remove-flags",
    separator=Separator(phone=" ", word="  ", syllable=""),
)
phoneme_map = dict(zip(unique_utterances, phonemized))
df["phonemes"] = df["utterance"].map(phoneme_map)
df = df[df["phonemes"].str.strip().str.len() > 0].reset_index(drop=True)
logger.info("Sample: '%s' -> '%s'", df["utterance"].iloc[0], df["phonemes"].iloc[0])

# ──────────────────────────────────────────────────────────────────────
# 3. Load processor (needed for vocab before Lightning module)
# ──────────────────────────────────────────────────────────────────────
logger.info("Loading processor from %s...", BASE_MODEL)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL)
tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(BASE_MODEL)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
tokenizer.backend = "espeak"

vocab = processor.tokenizer.get_vocab()
unk_id = vocab.get("<unk>", 3)
space_id = vocab.get(" ", len(vocab) - 1)
actual_vocab_size = len(vocab)  # 393


def phonemes_to_ids(phoneme_str):
    """Convert space-separated phoneme string to token IDs."""
    ids = []
    words = phoneme_str.split("  ")
    for wi, word in enumerate(words):
        if wi > 0:
            ids.append(space_id)
        phones = word.split(" ")
        for phone in phones:
            if phone == "":
                continue
            if phone in vocab:
                ids.append(vocab[phone])
            else:
                ids.append(unk_id)
    return ids


# Pre-tokenize labels
df["label_ids"] = df["phonemes"].apply(phonemes_to_ids)

# Log split+dataset breakdown
for split_name in ["train", "dev", "test"]:
    split_data = df[df["split"] == split_name]
    ds_counts = split_data["dataset"].value_counts().to_dict()
    logger.info("  %s: %d total (%s)", split_name, len(split_data), ds_counts)


# ──────────────────────────────────────────────────────────────────────
# 4. W&B init
# ──────────────────────────────────────────────────────────────────────
logger.info("Initializing W&B...")

# Compute duration stats before W&B init
durations = []
for _, row in df.iterrows():
    try:
        info = sf.info(row["wav_path"])
        durations.append({"split": row["split"], "dataset": row["dataset"], "duration_sec": info.duration})
    except Exception:
        pass
dur_df = pd.DataFrame(durations)

wandb.init(
    project="wav2vec2-combined-finetune",
    name="combined-uxtd-tal80-top-layer",
    config={
        "base_model": BASE_MODEL,
        "target_sr": TARGET_SR,
        "max_audio_sec": MAX_AUDIO_SEC,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "precision": "bf16-mixed",
        "gradient_clip_val": 1.0,
        "balancing": "weighted_sampling",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "uxtd_utterances": len(df[df["dataset"] == "uxtd"]),
        "tal80_utterances": len(df[df["dataset"] == "tal80"]),
        "total_utterances": len(df),
        "uxtd_source_sr": UXTD_SOURCE_SR,
        "tal80_source_sr": TAL80_SOURCE_SR,
        "num_tal80_speakers": len(tal80_speakers),
        "num_tal80_train_speakers": len(tal80_train_speakers),
        "num_tal80_dev_speakers": len(tal80_dev_speakers),
        "num_tal80_test_speakers": len(tal80_test_speakers),
    },
)

# Log split sizes per dataset
for split_name in ["train", "dev", "test"]:
    split_data = df[df["split"] == split_name]
    wandb.config.update({
        f"num_{split_name}_total": len(split_data),
        f"num_{split_name}_uxtd": len(split_data[split_data["dataset"] == "uxtd"]),
        f"num_{split_name}_tal80": len(split_data[split_data["dataset"] == "tal80"]),
    })

# Log duration stats
for split_name in ["train", "dev", "test"]:
    split_durs = dur_df[dur_df["split"] == split_name]["duration_sec"]
    if len(split_durs) == 0:
        continue
    wandb.config.update({
        f"{split_name}_total_hours": round(split_durs.sum() / 3600, 2),
        f"{split_name}_mean_dur_sec": round(split_durs.mean(), 2),
        f"{split_name}_max_dur_sec": round(split_durs.max(), 2),
    })

wandb.log({
    "dataset/duration_histogram": wandb.Histogram(dur_df["duration_sec"].values, num_bins=50),
    "dataset/total_hours": round(dur_df["duration_sec"].sum() / 3600, 2),
})

# Log per-dataset duration histograms
for ds_name in ["uxtd", "tal80"]:
    ds_durs = dur_df[dur_df["dataset"] == ds_name]["duration_sec"]
    if len(ds_durs) > 0:
        wandb.log({f"dataset/{ds_name}_duration_histogram": wandb.Histogram(ds_durs.values, num_bins=50)})

# Log phoneme frequency from training set
train_df = df[df["split"] == "train"]
id_to_phone = {v: k for k, v in vocab.items()}
phone_counts = {}
for label_ids in train_df["label_ids"]:
    for tid in label_ids:
        phone = id_to_phone.get(tid, f"id_{tid}")
        phone_counts[phone] = phone_counts.get(phone, 0) + 1

phone_table = wandb.Table(
    columns=["phoneme", "count"],
    data=sorted(phone_counts.items(), key=lambda x: -x[1]),
)
wandb.log({"dataset/phoneme_distribution": phone_table})
logger.info("Unique phonemes in training labels: %d", len(phone_counts))

# Log label length distribution
train_label_lens = [len(ids) for ids in train_df["label_ids"]]
wandb.log({
    "dataset/train_label_len_histogram": wandb.Histogram(train_label_lens, num_bins=40),
    "dataset/train_mean_label_len": np.mean(train_label_lens),
})


# ──────────────────────────────────────────────────────────────────────
# 5. PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────
class SpeechPhonemeDataset(Dataset):
    """Stores metadata + pre-tokenized labels. Audio loaded lazily in collator."""

    def __init__(self, dataframe: pd.DataFrame):
        self.wav_paths = dataframe["wav_path"].tolist()
        self.label_ids = dataframe["label_ids"].tolist()
        self.source_srs = dataframe["source_sr"].tolist()
        self.datasets = dataframe["dataset"].tolist()

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        return {
            "wav_path": self.wav_paths[idx],
            "labels": self.label_ids[idx],
            "source_sr": self.source_srs[idx],
            "dataset": self.datasets[idx],
        }


def collate_fn(features: List[Dict]) -> Dict[str, torch.Tensor]:
    """Load audio lazily, resample, extract features, pad batch."""
    input_features = []
    for f in features:
        audio, sr = sf.read(f["wav_path"])
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors=None)
        input_features.append({"input_values": inputs.input_values[0]})

    label_features = [{"input_ids": f["labels"]} for f in features]

    batch = processor.pad(input_features, padding=True, return_tensors="pt")
    labels_batch = processor.pad(labels=label_features, padding=True, return_tensors="pt")

    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
    batch["labels"] = labels

    return batch


# ──────────────────────────────────────────────────────────────────────
# 6. Lightning DataModule
# ──────────────────────────────────────────────────────────────────────
class CombinedSpeechDataModule(pl.LightningDataModule):
    def __init__(self, dataframe: pd.DataFrame, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
        super().__init__()
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        train_df = self.dataframe[self.dataframe["split"] == "train"].reset_index(drop=True)
        val_df = self.dataframe[self.dataframe["split"] == "dev"].reset_index(drop=True)
        test_df = self.dataframe[self.dataframe["split"] == "test"].reset_index(drop=True)

        self.train_dataset = SpeechPhonemeDataset(train_df)
        self.val_dataset = SpeechPhonemeDataset(val_df)
        self.test_dataset = SpeechPhonemeDataset(test_df)

        # Compute per-sample weights for balanced sampling across datasets
        datasets_col = train_df["dataset"].values
        uxtd_count = (datasets_col == "uxtd").sum()
        tal80_count = (datasets_col == "tal80").sum()

        # Each dataset gets equal total weight: weight_per_sample = 1/(2*count_in_dataset)
        weights = np.zeros(len(train_df), dtype=np.float64)
        if uxtd_count > 0:
            weights[datasets_col == "uxtd"] = 1.0 / (2.0 * uxtd_count)
        if tal80_count > 0:
            weights[datasets_col == "tal80"] = 1.0 / (2.0 * tal80_count)

        self.sample_weights = torch.from_numpy(weights).double()
        logger.info(
            "Weighted sampler: UXTD weight=%.6f (%d samples), TaL80 weight=%.6f (%d samples)",
            1.0 / (2.0 * uxtd_count) if uxtd_count else 0,
            uxtd_count,
            1.0 / (2.0 * tal80_count) if tal80_count else 0,
            tal80_count,
        )

    def train_dataloader(self):
        sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.train_dataset),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )


# ──────────────────────────────────────────────────────────────────────
# 7. Lightning Module
# ──────────────────────────────────────────────────────────────────────
class Wav2Vec2FineTuner(pl.LightningModule):
    def __init__(
        self,
        model_name: str = BASE_MODEL,
        learning_rate: float = LEARNING_RATE,
        warmup_steps: int = WARMUP_STEPS,
        vocab_size: int = actual_vocab_size,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

        # Load pretrained model
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # Fix vocab size (space token id=392 exceeds config vocab_size=392)
        if self.model.config.vocab_size < vocab_size:
            self.model.config.vocab_size = vocab_size
            old_weight = self.model.lm_head.weight.data
            old_bias = self.model.lm_head.bias.data
            self.model.lm_head = torch.nn.Linear(old_weight.shape[1], vocab_size)
            self.model.lm_head.weight.data[: old_weight.shape[0]] = old_weight
            self.model.lm_head.bias.data[: old_bias.shape[0]] = old_bias
            logger.info("Resized lm_head from %d to %d", old_weight.shape[0], vocab_size)

        # Freeze ALL parameters, then unfreeze only lm_head (top linear layer)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

        # Log param counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "Model params: %.1fM total, %.1fM trainable (%.1f%%)",
            total_params / 1e6,
            trainable_params / 1e6,
            100 * trainable_params / total_params,
        )
        wandb.config.update({
            "total_params_M": round(total_params / 1e6, 1),
            "trainable_params_M": round(trainable_params / 1e6, 1),
            "frozen_params_M": round((total_params - trainable_params) / 1e6, 1),
            "pct_trainable": round(100 * trainable_params / total_params, 1),
            "vocab_size": vocab_size,
        })

        # Validation step outputs (accumulated for PER computation)
        self.val_pred_ids = []
        self.val_label_ids = []
        self.test_pred_ids = []
        self.test_label_ids = []

    def forward(self, input_values, labels=None):
        return self.model(input_values=input_values, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(input_values=batch["input_values"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(input_values=batch["input_values"], labels=batch["labels"])
        loss = outputs.loss
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        pred_ids = outputs.logits.argmax(dim=-1).cpu().numpy()
        label_ids = batch["labels"].cpu().numpy()
        self.val_pred_ids.extend(pred_ids.tolist())
        self.val_label_ids.extend(label_ids.tolist())

    def on_validation_epoch_end(self):
        per = self._compute_per(self.val_pred_ids, self.val_label_ids)
        self.log("val/per", per, prog_bar=True, sync_dist=True)
        logger.info("Epoch %s — val/per: %.4f", self.current_epoch, per)

        # Log GPU memory
        if torch.cuda.is_available():
            wandb.log({
                "gpu/allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu/reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "gpu/max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            })

        self.val_pred_ids.clear()
        self.val_label_ids.clear()

    def test_step(self, batch, batch_idx):
        outputs = self(input_values=batch["input_values"], labels=batch["labels"])
        loss = outputs.loss
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)

        pred_ids = outputs.logits.argmax(dim=-1).cpu().numpy()
        label_ids = batch["labels"].cpu().numpy()
        self.test_pred_ids.extend(pred_ids.tolist())
        self.test_label_ids.extend(label_ids.tolist())

    def on_test_epoch_end(self):
        per = self._compute_per(self.test_pred_ids, self.test_label_ids)
        self.log("test/per", per, sync_dist=True)
        logger.info("Test PER: %.4f", per)
        self.test_pred_ids.clear()
        self.test_label_ids.clear()

    def _compute_per(self, pred_ids_list, label_ids_list):
        """Compute Phoneme Error Rate using the processor's decoder."""
        from evaluate import load as load_metric

        wer_metric = load_metric("wer")

        # Replace -100 padding with pad_token_id for decoding
        pad_id = processor.tokenizer.pad_token_id
        cleaned_labels = []
        for label_ids in label_ids_list:
            cleaned_labels.append([pad_id if lid == -100 else lid for lid in label_ids])

        pred_str = processor.batch_decode(pred_ids_list)
        label_str = processor.batch_decode(cleaned_labels, group_tokens=False)

        # Filter out empty references to avoid division by zero
        valid = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not valid:
            return 1.0
        pred_str, label_str = zip(*valid)

        per = wer_metric.compute(predictions=list(pred_str), references=list(label_str))
        return per

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        # Linear warmup then linear decay
        from transformers import get_linear_schedule_with_warmup

        # Estimate total training steps
        # This is approximate — Lightning's trainer has the exact count
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# ──────────────────────────────────────────────────────────────────────
# 8. Prediction logging callback
# ──────────────────────────────────────────────────────────────────────
NUM_PREDICTION_SAMPLES = 10


class PredictionLoggingCallback(pl.Callback):
    """Log sample predictions as a W&B table after each validation epoch."""

    def __init__(self, val_dataset: SpeechPhonemeDataset, num_samples: int = NUM_PREDICTION_SAMPLES):
        super().__init__()
        self.val_dataset = val_dataset
        self.num_samples = min(num_samples, len(val_dataset))
        self.sample_indices = list(range(self.num_samples))

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: Wav2Vec2FineTuner):
        model = pl_module.model
        model.eval()

        table = wandb.Table(columns=["epoch", "index", "ground_truth", "prediction", "label_len", "pred_len"])

        for idx in self.sample_indices:
            sample = self.val_dataset[idx]

            audio, sr = sf.read(sample["wav_path"])
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

            inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
            input_values = inputs.input_values.to(pl_module.device)

            with torch.no_grad():
                logits = model(input_values).logits

            pred_ids = logits.argmax(dim=-1)[0].cpu().numpy()
            pred_str = processor.decode(pred_ids)

            label_ids = [i for i in sample["labels"] if i != -100]
            label_str = processor.tokenizer.decode(label_ids)

            table.add_data(
                trainer.current_epoch,
                idx,
                label_str,
                pred_str,
                len(label_ids),
                len(pred_ids),
            )

        wandb.log({"predictions/samples": table})


# ──────────────────────────────────────────────────────────────────────
# 9. Main training
# ──────────────────────────────────────────────────────────────────────
def main():
    # Create data module
    data_module = CombinedSpeechDataModule(df)
    data_module.setup()

    # Create model
    model = Wav2Vec2FineTuner()

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename="combined-{epoch:02d}-{val_per:.4f}",
        monitor="val/per",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    prediction_callback = PredictionLoggingCallback(val_dataset=data_module.val_dataset)

    # Logger
    wandb_logger = WandbLogger(
        project="wav2vec2-combined-finetune",
        experiment=wandb.run,  # reuse the already-initialized run
        log_model=False,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accumulate_grad_batches=GRAD_ACCUM,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, lr_monitor, prediction_callback],
        logger=wandb_logger,
        log_every_n_steps=50,
        val_check_interval=1.0,  # validate every epoch
        enable_checkpointing=True,
        deterministic=False,
        accelerator="gpu",
        devices=1,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, datamodule=data_module)

    # Save final model + processor
    logger.info("Saving final model to %s...", OUTPUT_DIR)
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info("Loading best checkpoint: %s (val/per=%.4f)", best_model_path, checkpoint_callback.best_model_score)
        best_model = Wav2Vec2FineTuner.load_from_checkpoint(best_model_path)
        best_model.model.save_pretrained(OUTPUT_DIR)
    else:
        model.model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    logger.info("Model + processor saved to %s", OUTPUT_DIR)

    # Test evaluation
    logger.info("Running test evaluation...")
    trainer.test(model, datamodule=data_module, ckpt_path="best")

    wandb.finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()
