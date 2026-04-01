#!/usr/bin/env python3
"""Fine-tune facebook/wav2vec2-lv-60-espeak-cv-ft on UXTD child speech corpus."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Dataset, DatasetDict
from evaluate import load as load_metric
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
CSV_PATH = "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wave2vec2_cpp/tests/utterances_by_length.csv"
SPEAKERS_PATH = "/home/ultraspeech-dev/ultrasuite/core-uxtd/doc/speakers"
OUTPUT_DIR = "./wav2vec2-uxtd-finetuned"
TARGET_SR = 16000
SOURCE_SR = 22050
MAX_AUDIO_SEC = 10  # truncate audio longer than this to fit in GPU memory

# ──────────────────────────────────────────────────────────────────────
# 1. Load CSV and speaker splits
# ──────────────────────────────────────────────────────────────────────
print("Loading utterance CSV...")
df = pd.read_csv(CSV_PATH)

# Derive wav paths from the txt filepath column
df["wav_path"] = df["filepath"].str.replace(r"\.txt$", ".wav", regex=True)

# Verify all wav files exist
missing = df[~df["wav_path"].apply(os.path.exists)]
if len(missing) > 0:
    print(f"WARNING: {len(missing)} wav files not found. Dropping them.")
    df = df[df["wav_path"].apply(os.path.exists)].reset_index(drop=True)
print(f"Total utterances with audio: {len(df)}")

# Load speaker split info
speaker_df = pd.read_csv(SPEAKERS_PATH, sep="\t")
speaker_to_split = dict(zip(speaker_df["speaker_id"], speaker_df["subset"]))

# Map each utterance to its split
df["split"] = df["speaker"].map(speaker_to_split)
# Drop any utterances whose speaker isn't in the speakers file
df = df.dropna(subset=["split"]).reset_index(drop=True)
print(f"Split distribution:\n{df['split'].value_counts()}")

# ──────────────────────────────────────────────────────────────────────
# 2. Convert text prompts to IPA phonemes
# ──────────────────────────────────────────────────────────────────────
print("Converting text to IPA phonemes...")
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
print(
    f"Sample phonemization: '{df['utterance'].iloc[0]}' -> '{df['phonemes'].iloc[0]}'"
)

# ──────────────────────────────────────────────────────────────────────
# 3. Load processor and model
# ──────────────────────────────────────────────────────────────────────
print("Loading processor and model...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

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
    print(f"Resized lm_head from {old_weight.shape[0]} to {actual_vocab_size}")

# Freeze the feature encoder (CNN layers)
model.freeze_feature_encoder()
print("Feature encoder frozen.")

# ──────────────────────────────────────────────────────────────────────
# 4. Build HuggingFace datasets
# ──────────────────────────────────────────────────────────────────────
print("Building datasets...")

# Build phoneme-to-id mapping from the vocab
# Phonemizer output uses ' ' between phonemes within a word, '  ' between words
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
print(
    f"Tokenization check: '{sample_phones}' -> {len(sample_ids)} tokens -> '{decoded}'"
)


def prepare_example(row):
    """Load audio, resample, and tokenize phoneme labels."""
    audio, sr = sf.read(row["wav_path"])
    # Resample to 16kHz
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    # Truncate to max length to avoid OOM
    max_samples = int(MAX_AUDIO_SEC * TARGET_SR)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    # Process audio through wav2vec2 feature extractor
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors=None)

    # Tokenize phoneme labels using greedy longest-match
    label_ids = phonemes_to_ids(row["phonemes"])

    return {
        "input_values": inputs.input_values[0],
        "labels": label_ids,
    }


# Create train/dev/test datasets
datasets = {}
for split_name in ["train", "dev", "test"]:
    split_df = df[df["split"] == split_name].reset_index(drop=True)
    if len(split_df) == 0:
        continue
    ds = Dataset.from_pandas(split_df[["wav_path", "phonemes"]])
    ds = ds.map(
        prepare_example,
        remove_columns=["wav_path", "phonemes"],
        num_proc=1,  # soundfile isn't always multiprocess-safe
        desc=f"Processing {split_name}",
    )
    datasets[split_name] = ds
    print(f"  {split_name}: {len(ds)} examples")

dataset_dict = DatasetDict(datasets)


# ──────────────────────────────────────────────────────────────────────
# 5. Data collator
# ──────────────────────────────────────────────────────────────────────
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Separate input_values and labels
        input_features = [{"input_values": f["input_values"]} for f in features]
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


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 in labels (padding) with pad_token_id
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    per = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"per": per}


# ──────────────────────────────────────────────────────────────────────
# 7. Training arguments
# ──────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=30,
    learning_rate=3e-4,
    warmup_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="per",
    greater_is_better=False,
    save_total_limit=3,
    dataloader_num_workers=4,
    report_to="none",
    remove_unused_columns=False,
)

# ──────────────────────────────────────────────────────────────────────
# 8. Trainer
# ──────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict.get("dev", dataset_dict.get("test")),
    data_collator=data_collator,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

# ──────────────────────────────────────────────────────────────────────
# 9. Train!
# ──────────────────────────────────────────────────────────────────────
print("\nStarting training...")
trainer.train()

# Save the final model
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")

# ──────────────────────────────────────────────────────────────────────
# 10. Evaluate on test set
# ──────────────────────────────────────────────────────────────────────
if "test" in dataset_dict:
    print("\nEvaluating on test set...")
    results = trainer.evaluate(dataset_dict["test"])
    print(f"Test PER: {results['eval_per']:.4f}")
