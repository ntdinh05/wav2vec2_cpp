#!/usr/bin/env python3
"""Benchmark original vs fine-tuned wav2vec2 models on UXTD child speech.

Compares phoneme error rate (PER) and inference time for:
  - Original: facebook/wav2vec2-lv-60-espeak-cv-ft (ONNX)
  - UXTD fine-tuned: fine-tuned on UXTD child speech (ONNX)
  - TaL80 fine-tuned: fine-tuned on TaL80 adult speech from UXTD checkpoint (ONNX)

Outputs per-utterance and summary CSVs.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import onnxruntime as ort
from jiwer import wer

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

ORIGINAL_ONNX = os.path.join(PROJECT_DIR, "onnx_models", "wav2vec2_original.onnx")
UXTD_ONNX = os.path.join(PROJECT_DIR, "onnx_models", "wav2vec2_uxtd.onnx")
TAL80_ONNX = os.path.join(PROJECT_DIR, "onnx_models", "wav2vec2_tal80.onnx")
CSV_PATH = os.path.join(PROJECT_DIR, "tests", "utterances_by_length.csv")
SPEAKERS_PATH = "/home/ultraspeech-dev/ultrasuite/core-uxtd/doc/speakers"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

TARGET_SR = 16000
MAX_AUDIO_SEC = 10

# ── Load processor and phonemizer ────────────────────────────────────
from transformers import Wav2Vec2Processor
from phonemizer import phonemize
from phonemizer.separator import Separator

print("Loading processor...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

# ── Load data ────────────────────────────────────────────────────────
print("Loading utterance data...")
df = pd.read_csv(CSV_PATH)
df["wav_path"] = df["filepath"].str.replace(r"\.txt$", ".wav", regex=True)

# Verify wav files exist
df = df[df["wav_path"].apply(os.path.exists)].reset_index(drop=True)

# Load speaker splits
speaker_df = pd.read_csv(SPEAKERS_PATH, sep="\t")
speaker_to_split = dict(zip(speaker_df["speaker_id"], speaker_df["subset"]))
df["split"] = df["speaker"].map(speaker_to_split)
df = df.dropna(subset=["split"]).reset_index(drop=True)

# Generate IPA phoneme references
print("Generating IPA phoneme references...")
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
df["ref_phonemes"] = df["utterance"].map(phoneme_map)

print(f"Total utterances: {len(df)}")
print(f"Split distribution: {df['split'].value_counts().to_dict()}")

# ── ONNX inference function ──────────────────────────────────────────
def run_inference(session, audio_path):
    """Load audio, preprocess, run ONNX inference, return decoded string and time."""
    audio, sr = sf.read(audio_path)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    max_samples = int(MAX_AUDIO_SEC * TARGET_SR)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="np")

    t0 = time.perf_counter()
    logits = session.run(None, {"input_values": inputs.input_values})[0]
    t1 = time.perf_counter()

    pred_ids = np.argmax(logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)[0]

    return pred_str, t1 - t0


def compute_per(ref, hyp):
    """Compute phoneme error rate between two phoneme strings."""
    # Normalize: collapse whitespace for fair comparison
    ref_norm = " ".join(ref.split())
    hyp_norm = " ".join(hyp.split())
    if not ref_norm:
        return 1.0 if hyp_norm else 0.0
    return wer(ref_norm, hyp_norm)


# ── Run benchmark ────────────────────────────────────────────────────
# Use CUDA if available, fall back to CPU
available_providers = ort.get_available_providers()
if "CUDAExecutionProvider" in available_providers:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("Using CUDA GPU for inference")
else:
    providers = ["CPUExecutionProvider"]
    print("WARNING: CUDAExecutionProvider not available, falling back to CPU")

model_configs = [("original", ORIGINAL_ONNX), ("uxtd_finetuned", UXTD_ONNX), ("tal80_finetuned", TAL80_ONNX)]

results = []

# Load one model at a time to avoid GPU OOM from holding all 3 (~1.2GB each)
for model_name, onnx_path in model_configs:
    if not os.path.exists(onnx_path):
        print(f"WARNING: {onnx_path} not found, skipping {model_name}")
        continue

    print(f"\nLoading ONNX session: {model_name} ({os.path.basename(onnx_path)})...")
    session = ort.InferenceSession(onnx_path, providers=providers)

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  [{model_name}] Processing utterance {idx + 1}/{len(df)}...")

        try:
            hyp, inf_time = run_inference(session, row["wav_path"])
            per = compute_per(row["ref_phonemes"], hyp)
            error = ""
        except Exception as e:
            hyp, inf_time, per = "", 0.0, 1.0
            error = str(e)

        results.append({
            "model": model_name,
            "speaker": row["speaker"],
            "split": row["split"],
            "filename": row["filename"],
            "utterance": row["utterance"],
            "ref_phonemes": row["ref_phonemes"],
            "hyp_phonemes": hyp,
            "n_ref": len(row["ref_phonemes"].split()),
            "per": round(per, 4),
            "inference_time_s": round(inf_time, 4),
            "error": error,
        })

    # Release GPU memory before loading next model
    del session

results_df = pd.DataFrame(results)

# ── Per-utterance CSV ────────────────────────────────────────────────
per_utt_path = os.path.join(OUTPUT_DIR, "benchmark_original_vs_finetuned.csv")
results_df.to_csv(per_utt_path, index=False)
print(f"\nPer-utterance results: {per_utt_path}")

# ── Summary CSV ──────────────────────────────────────────────────────
summary_rows = []
for model_name in results_df["model"].unique():
    for split in ["train", "dev", "test", "all"]:
        if split == "all":
            subset = results_df[results_df["model"] == model_name]
        else:
            subset = results_df[(results_df["model"] == model_name) & (results_df["split"] == split)]
        if len(subset) == 0:
            continue
        summary_rows.append({
            "model": model_name,
            "split": split,
            "n_utterances": len(subset),
            "mean_per": round(subset["per"].mean(), 4),
            "median_per": round(subset["per"].median(), 4),
            "std_per": round(subset["per"].std(), 4),
            "min_per": round(subset["per"].min(), 4),
            "max_per": round(subset["per"].max(), 4),
            "mean_inference_time_s": round(subset["inference_time_s"].mean(), 4),
        })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, "benchmark_original_vs_finetuned_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Summary results: {summary_path}")

# ── Print summary to console ─────────────────────────────────────────
print("\n" + "=" * 70)
print("BENCHMARK SUMMARY: Original vs UXTD vs TaL80")
print("=" * 70)
print(summary_df.to_string(index=False))
print("=" * 70)
