#!/usr/bin/env python3
"""4-model benchmark: Base · UXTD · TaL80 · Combined (UXTD+TaL80).

All four models run via ONNX Runtime.
Test sets: UXTD test split + TaL80 test split.

Outputs (in output/):
  benchmark_4models_summary.csv       — Mean/Median/Std PER per model per dataset
  benchmark_4models_per_utterance.csv — Per-utterance PER for all combinations
"""

import gc
import json
import os

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
import torch
from jiwer import wer
from phonemizer import phonemize
from phonemizer.separator import Separator
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
)

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR    = os.path.dirname(PROJECT_DIR)

ORIGINAL_ONNX     = os.path.join(PROJECT_DIR, "onnx_models", "wav2vec2_original.onnx")
UXTD_TOP_ONNX     = os.path.join(PROJECT_DIR, "onnx_models", "wav2vec2_uxtd_top_layer.onnx")
COMBINED_TOP_ONNX = os.path.join(PROJECT_DIR, "onnx_models", "wav2vec2_combined_top_layer.onnx")

BASE_PROCESSOR_PATH     = "facebook/wav2vec2-lv-60-espeak-cv-ft"
UXTD_TOP_PROCESSOR_PATH = os.path.join(REPO_DIR, "wav2vec2-uxtd-finetuned-top-layer")

UXTD_CSV       = os.path.join(PROJECT_DIR, "tests",  "utterances_by_length.csv")
UXTD_SPEAKERS  = "/home/ultraspeech-dev/ultrasuite/core-uxtd/doc/speakers"
TAL80_CSV      = os.path.join(PROJECT_DIR, "output", "tal80_utterances_by_length.csv")
TAL80_SPEAKER_MAP = os.path.join(PROJECT_DIR, "output", "speaker_map.json")
OUTPUT_DIR     = os.path.join(PROJECT_DIR, "output")

TARGET_SR     = 16000
MAX_AUDIO_SEC = 10

# ── Model configs: (display_name, onnx_path, processor_path) ──────────
MODEL_CONFIGS = [
    ("Base (LV60)",          ORIGINAL_ONNX,    BASE_PROCESSOR_PATH),
    ("UXTD top-layer",       UXTD_TOP_ONNX,    UXTD_TOP_PROCESSOR_PATH),
    ("Combined top-layer",   COMBINED_TOP_ONNX, BASE_PROCESSOR_PATH),
]


# ── Processor loading ─────────────────────────────────────────────────
def load_processor(path):
    """Load Wav2Vec2Processor, with fallback for nested processor_config.json."""
    try:
        return Wav2Vec2Processor.from_pretrained(path)
    except TypeError:
        fe  = Wav2Vec2FeatureExtractor.from_pretrained(path)
        tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(path)
        return Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)


# ── Dataset helpers ───────────────────────────────────────────────────
def load_uxtd_test():
    df = pd.read_csv(UXTD_CSV)
    df["wav_path"] = df["filepath"].str.replace(r"\.txt$", ".wav", regex=True)
    df = df[df["wav_path"].apply(os.path.exists)].reset_index(drop=True)
    speaker_df = pd.read_csv(UXTD_SPEAKERS, sep="\t")
    split_map  = dict(zip(speaker_df["speaker_id"], speaker_df["subset"]))
    df["split"] = df["speaker"].map(split_map)
    df = df[df["split"] == "test"].reset_index(drop=True)
    return df, sorted(df["speaker"].unique().tolist())


def load_tal80_test():
    df = pd.read_csv(TAL80_CSV)
    df["wav_path"] = df["filepath"].str.replace(r"\.txt$", ".wav", regex=True)
    df = df[df["wav_path"].apply(os.path.exists)].reset_index(drop=True)
    with open(TAL80_SPEAKER_MAP) as f:
        speaker_map = json.load(f)
    test_speakers = sorted(speaker_map["test"])
    df = df[df["speaker"].isin(test_speakers)].reset_index(drop=True)
    return df, test_speakers


def add_ref_phonemes(df):
    unique_utts = df["utterance"].unique().tolist()
    phonemized  = phonemize(
        unique_utts,
        language="en-us",
        backend="espeak",
        strip=True,
        with_stress=False,
        language_switch="remove-flags",
        separator=Separator(phone=" ", word="  ", syllable=""),
    )
    pmap = dict(zip(unique_utts, phonemized))
    df["ref_phonemes"] = df["utterance"].map(pmap)
    return df.dropna(subset=["ref_phonemes"]).reset_index(drop=True)


# ── Audio / inference / PER ───────────────────────────────────────────

# Placeholder that stands in for the word-boundary space token (vocab id 392).
# Using a printable string ensures jiwer treats it as a real token, so
# insertions/deletions of word boundaries are counted in PER.
_WORD_BOUNDARY = "▁"


def load_audio(path):
    audio, sr = sf.read(path)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    max_samples = int(MAX_AUDIO_SEC * TARGET_SR)
    return audio[:max_samples] if len(audio) > max_samples else audio


def run_onnx(session, proc, audio_path):
    """Return a 1-D numpy array of argmax token IDs (pre-CTC-collapse)."""
    audio  = load_audio(audio_path)
    inputs = proc(audio, sampling_rate=TARGET_SR, return_tensors="np")
    logits = session.run(None, {"input_values": inputs.input_values})[0]
    return np.argmax(logits[0], axis=-1)  # shape (T,)


def _ref_to_token_str(ref_phonemes: str) -> str:
    """Convert phonemizer output to a space-separated token string.

    Phonemizer uses single-space between phones and double-space between
    words.  We map each word boundary to _WORD_BOUNDARY so it is counted
    as an explicit edit target by jiwer, instead of being silently dropped
    by str.split().
    """
    tokens = []
    for i, word in enumerate(ref_phonemes.strip().split("  ")):
        if i > 0:
            tokens.append(_WORD_BOUNDARY)
        tokens.extend(p for p in word.split(" ") if p)
    return " ".join(tokens)


def _pred_ids_to_token_str(pred_ids_1d: np.ndarray, proc) -> str:
    """CTC-decode a 1-D argmax array to a space-separated token string.

    The word-boundary space token (vocab id 392, string value " ") is
    mapped to _WORD_BOUNDARY so it is counted by jiwer on equal footing
    with phoneme tokens.
    """
    vocab      = proc.tokenizer.get_vocab()
    pad_id     = proc.tokenizer.pad_token_id
    id_to_tok  = {v: k for k, v in vocab.items()}
    special    = {proc.tokenizer.pad_token, proc.tokenizer.unk_token,
                  proc.tokenizer.bos_token, proc.tokenizer.eos_token}
    special.discard(None)

    tokens = []
    prev   = None
    for raw_id in pred_ids_1d:
        tid = int(raw_id)
        if tid == prev:          # CTC collapse: skip consecutive repeats
            continue
        prev = tid
        if tid == pad_id:        # blank / padding token
            continue
        tok = id_to_tok.get(tid, proc.tokenizer.unk_token)
        if tok in special:
            continue
        tokens.append(_WORD_BOUNDARY if tok == " " else tok)

    return " ".join(tokens)


def compute_per(ref_phonemes: str, pred_ids_1d: np.ndarray, proc) -> tuple:
    """Return (per_float, ref_token_str, hyp_token_str).

    Working at the token level (rather than using proc.batch_decode +
    str.split) means word-boundary tokens are counted as explicit edit
    targets instead of being silently absorbed by whitespace collapsing.
    """
    ref_str = _ref_to_token_str(ref_phonemes)
    hyp_str = _pred_ids_to_token_str(pred_ids_1d, proc)

    if not ref_str:
        return (1.0 if hyp_str else 0.0), ref_str, hyp_str
    if not hyp_str:
        return 1.0, ref_str, hyp_str
    return wer(ref_str, hyp_str), ref_str, hyp_str


# ── Main ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading UXTD test split...")
    uxtd_df, uxtd_speakers = load_uxtd_test()
    print(f"  {len(uxtd_df)} utterances, {len(uxtd_speakers)} speakers")

    print("Loading TaL80 test split...")
    tal80_df, tal80_speakers = load_tal80_test()
    print(f"  {len(tal80_df)} utterances, {len(tal80_speakers)} speakers")

    print("Generating IPA phoneme references...")
    uxtd_df  = add_ref_phonemes(uxtd_df)
    tal80_df = add_ref_phonemes(tal80_df)

    datasets = [("UXTD", uxtd_df), ("TaL80", tal80_df)]

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("ONNX: using CUDAExecutionProvider")
    else:
        providers = ["CPUExecutionProvider"]
        print("WARNING: ONNX falling back to CPU")

    all_results = []
    # corpus_tokens[(model_name, dataset_name)] = {"refs": [...], "hyps": [...]}
    # Each entry is a space-separated token string for one utterance.
    corpus_tokens: dict = {}

    for model_name, onnx_path, proc_path in MODEL_CONFIGS:
        if not os.path.exists(onnx_path):
            print(f"\nWARNING: {onnx_path} not found — skipping {model_name}")
            continue

        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        print(f"  Loading processor...")
        proc = load_processor(proc_path)

        print(f"  Loading ONNX session...")
        try:
            session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            print(f"  CUDA load failed ({e}), falling back to CPU...")
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        for dataset_name, test_df in datasets:
            print(f"\n  Running on {dataset_name} test ({len(test_df)} utterances)...")

            key = (model_name, dataset_name)
            corpus_tokens[key] = {"refs": [], "hyps": []}

            for i, (_, row) in enumerate(test_df.iterrows()):
                if i % 200 == 0:
                    print(f"    [{dataset_name}] {i + 1}/{len(test_df)}...")

                try:
                    pred_ids = run_onnx(session, proc, row["wav_path"])
                    per, ref_str, hyp_str = compute_per(row["ref_phonemes"], pred_ids, proc)
                except Exception as e:
                    per, ref_str, hyp_str = 1.0, "", ""
                    print(f"    Error on {row['wav_path']}: {e}")

                corpus_tokens[key]["refs"].append(ref_str)
                corpus_tokens[key]["hyps"].append(hyp_str)
                all_results.append({
                    "model":     model_name,
                    "dataset":   dataset_name,
                    "speaker":   row.get("speaker", ""),
                    "utterance": row.get("utterance", ""),
                    "per":       per,
                })

        del session, proc
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    results_df = pd.DataFrame(all_results)

    # ── Per-utterance CSV ─────────────────────────────────────────────
    utt_path = os.path.join(OUTPUT_DIR, "benchmark_4models_per_utterance.csv")
    results_df.to_csv(utt_path, index=False)
    print(f"\nPer-utterance results saved: {utt_path}")

    # ── Summary CSV ───────────────────────────────────────────────────
    summary_rows = []
    for model_name, _, _ in MODEL_CONFIGS:
        for dataset_name in ["UXTD", "TaL80"]:
            subset = results_df[
                (results_df["model"] == model_name) &
                (results_df["dataset"] == dataset_name)
            ]
            if subset.empty:
                continue

            # Corpus-level PER: total edit distance / total reference tokens
            # across all utterances — comparable to the metric logged during training.
            key = (model_name, dataset_name)
            ct  = corpus_tokens.get(key, {"refs": [], "hyps": []})
            paired_refs = [r for r, h in zip(ct["refs"], ct["hyps"]) if r]
            paired_hyps = [h for r, h in zip(ct["refs"], ct["hyps"]) if r]
            corpus_per  = round(wer(paired_refs, paired_hyps), 4) if paired_refs else float("nan")

            summary_rows.append({
                "Model":        model_name,
                "Test Set":     dataset_name,
                "N":            len(subset),
                "Corpus PER":   corpus_per,
                "Mean PER":     round(subset["per"].mean(), 4),
                "Median PER":   round(subset["per"].median(), 4),
                "Std PER":      round(subset["per"].std(), 4),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "benchmark_4models_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    print("\n" + "=" * 75)
    print("4-MODEL BENCHMARK SUMMARY")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
