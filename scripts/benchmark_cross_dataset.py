#!/usr/bin/env python3
"""Cross-dataset benchmark: 3 models × 2 test sets (UXTD + TaL80).

Produces three CSV files in output/:
  1. benchmark_summary_cross_dataset.csv  — Mean/Median PER per model per dataset
  2. benchmark_test_speakers.csv          — Test speakers for each dataset
  3. benchmark_example_predictions.csv    — Force-aligned word-level predictions
"""

import gc
import json
import os
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
REPO_DIR = os.path.dirname(PROJECT_DIR)

ORIGINAL_ONNX = os.path.join(PROJECT_DIR, "onnx_models", "wav2vec2_original.onnx")
UXTD_ONNX = os.path.join(PROJECT_DIR, "onnx_models", "wav2vec2_uxtd.onnx")
TAL80_ONNX = os.path.join(PROJECT_DIR, "onnx_models", "wav2vec2_tal80.onnx")

ORIGINAL_PROCESSOR = "facebook/wav2vec2-lv-60-espeak-cv-ft"
UXTD_PROCESSOR = os.path.join(REPO_DIR, "wav2vec2-uxtd-finetuned")
TAL80_PROCESSOR = os.path.join(REPO_DIR, "wav2vec2-tal80-finetuned")

UXTD_CSV = os.path.join(PROJECT_DIR, "tests", "utterances_by_length.csv")
UXTD_SPEAKERS = "/home/ultraspeech-dev/ultrasuite/core-uxtd/doc/speakers"

TAL80_CSV = os.path.join(PROJECT_DIR, "output", "tal80_utterances_by_length.csv")
TAL80_SPEAKER_MAP = os.path.join(PROJECT_DIR, "output", "speaker_map.json")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

TARGET_SR = 16000
MAX_AUDIO_SEC = 10

MODEL_CONFIGS = [
    ("Wav2Vec2 LV60", ORIGINAL_ONNX, ORIGINAL_PROCESSOR),
    ("Wav2Vec2 LV60 (UXTD)", UXTD_ONNX, UXTD_PROCESSOR),
    ("Wav2Vec2 LV60 (TaL80)", TAL80_ONNX, TAL80_PROCESSOR),
]

# ── Load phonemizer ─────────────────────────────────────────────────
from transformers import Wav2Vec2Processor
from phonemizer import phonemize
from phonemizer.separator import Separator


# ── Helper: load UXTD test split ────────────────────────────────────
def load_uxtd_test():
    df = pd.read_csv(UXTD_CSV)
    df["wav_path"] = df["filepath"].str.replace(r"\.txt$", ".wav", regex=True)
    df = df[df["wav_path"].apply(os.path.exists)].reset_index(drop=True)

    speaker_df = pd.read_csv(UXTD_SPEAKERS, sep="\t")
    speaker_to_split = dict(zip(speaker_df["speaker_id"], speaker_df["subset"]))
    df["split"] = df["speaker"].map(speaker_to_split)
    df = df[df["split"] == "test"].reset_index(drop=True)
    test_speakers = sorted(df["speaker"].unique().tolist())
    return df, test_speakers


# ── Helper: load TaL80 test split ───────────────────────────────────
def load_tal80_test():
    df = pd.read_csv(TAL80_CSV)
    df["wav_path"] = df["filepath"].str.replace(r"\.txt$", ".wav", regex=True)
    df = df[df["wav_path"].apply(os.path.exists)].reset_index(drop=True)

    with open(TAL80_SPEAKER_MAP) as f:
        speaker_map = json.load(f)
    test_speakers = sorted(speaker_map["test"])
    df = df[df["speaker"].isin(test_speakers)].reset_index(drop=True)
    return df, test_speakers


# ── Phonemize references ────────────────────────────────────────────
def phonemize_text(text):
    """Phonemize a single text string."""
    result = phonemize(
        [text],
        language="en-us",
        backend="espeak",
        strip=True,
        with_stress=False,
        language_switch="remove-flags",
        separator=Separator(phone=" ", word="  ", syllable=""),
    )
    return result[0]


def phonemize_words(text):
    """Phonemize each word individually, return list of (word, phonemes)."""
    words = text.split()
    word_phonemes = []
    for w in words:
        ph = phonemize_text(w)
        word_phonemes.append((w, ph))
    return word_phonemes


def add_ref_phonemes(df):
    unique_utts = df["utterance"].unique().tolist()
    phonemized = phonemize(
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
    df = df.dropna(subset=["ref_phonemes"]).reset_index(drop=True)
    return df


# ── Audio loading ────────────────────────────────────────────────────
def load_audio(audio_path):
    """Load and resample audio to TARGET_SR, truncate to MAX_AUDIO_SEC."""
    audio, sr = sf.read(audio_path)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    max_samples = int(MAX_AUDIO_SEC * TARGET_SR)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio


# ── Inference ────────────────────────────────────────────────────────
def run_inference(session, proc, audio_path):
    """Run ONNX inference, return decoded string, logits, and time."""
    audio = load_audio(audio_path)
    inputs = proc(audio, sampling_rate=TARGET_SR, return_tensors="np")

    t0 = time.perf_counter()
    logits = session.run(None, {"input_values": inputs.input_values})[0]
    t1 = time.perf_counter()

    pred_ids = np.argmax(logits, axis=-1)
    pred_str = proc.batch_decode(pred_ids)[0]
    return pred_str, logits[0], t1 - t0


def clean_prediction(hyp):
    """Strip <unk> and other special tokens from prediction."""
    tokens = hyp.split()
    cleaned = [t for t in tokens if t not in ("<unk>", "<pad>", "<s>", "</s>")]
    return " ".join(cleaned)


def compute_per(ref, hyp):
    ref_norm = " ".join(ref.split())
    hyp_norm = " ".join(clean_prediction(hyp).split())
    if not ref_norm:
        return 1.0 if hyp_norm else 0.0
    return wer(ref_norm, hyp_norm)


# ── CTC Force Alignment ─────────────────────────────────────────────
def ctc_force_align(logits, token_ids, blank_id=0):
    """Viterbi CTC force alignment.

    Args:
        logits: (T, V) numpy array of raw logits
        token_ids: list of target token IDs (without blanks)
        blank_id: CTC blank token ID

    Returns:
        List of (frame_start, frame_end, token_id) for each target token.
    """
    T, V = logits.shape
    log_probs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)

    # Build CTC target: blank, tok0, blank, tok1, blank, ...
    S = 2 * len(token_ids) + 1
    target = []
    for tid in token_ids:
        target.append(blank_id)
        target.append(tid)
    target.append(blank_id)

    # Viterbi DP
    NEG_INF = -1e30
    dp = np.full((T, S), NEG_INF, dtype=np.float64)
    bt = np.full((T, S), -1, dtype=np.int32)

    # Init: can start at blank (s=0) or first token (s=1)
    dp[0, 0] = log_probs[0, target[0]]
    if S > 1:
        dp[0, 1] = log_probs[0, target[1]]

    for t in range(1, T):
        for s in range(S):
            # Stay in same state
            best = dp[t - 1, s]
            best_s = s

            # Transition from previous state
            if s > 0 and dp[t - 1, s - 1] > best:
                best = dp[t - 1, s - 1]
                best_s = s - 1

            # Skip blank if current and two-back are different non-blank tokens
            if s > 1 and target[s] != blank_id and target[s] != target[s - 2]:
                if dp[t - 1, s - 2] > best:
                    best = dp[t - 1, s - 2]
                    best_s = s - 2

            dp[t, s] = best + log_probs[t, target[s]]
            bt[t, s] = best_s

    # Backtrace from best final state (last blank or last token)
    if dp[T - 1, S - 1] >= dp[T - 1, S - 2]:
        s = S - 1
    else:
        s = S - 2

    path = []
    for t in range(T - 1, -1, -1):
        path.append((t, s))
        s = bt[t, s]
    path.reverse()

    # Extract token spans (skip blanks)
    alignments = []
    current_token_idx = None
    start_frame = None

    for t, s in path:
        if s % 2 == 1:  # Non-blank position
            tok_idx = s // 2
            if tok_idx != current_token_idx:
                if current_token_idx is not None:
                    alignments.append((start_frame, t - 1, token_ids[current_token_idx]))
                current_token_idx = tok_idx
                start_frame = t
        else:
            if current_token_idx is not None:
                alignments.append((start_frame, t - 1, token_ids[current_token_idx]))
                current_token_idx = None
                start_frame = None

    if current_token_idx is not None:
        alignments.append((start_frame, T - 1, token_ids[current_token_idx]))

    return alignments


def force_align_words(logits, proc, word_phonemes):
    """Force-align model logits to words, return per-word predicted phonemes.

    Args:
        logits: (T, V) numpy array
        proc: Wav2Vec2Processor for this model
        word_phonemes: list of (word, phoneme_string) from phonemize_words()

    Returns:
        list of (word, gt_phonemes, predicted_phonemes) tuples
    """
    tokenizer = proc.tokenizer

    # Build full target token sequence and track word boundaries
    all_token_ids = []
    word_boundaries = []  # (start_idx, end_idx) into all_token_ids for each word

    for word, phonemes in word_phonemes:
        phone_list = phonemes.split()
        start = len(all_token_ids)
        for ph in phone_list:
            tid = tokenizer.convert_tokens_to_ids(ph)
            if tid is not None and tid != tokenizer.unk_token_id:
                all_token_ids.append(tid)
        end = len(all_token_ids)
        word_boundaries.append((start, end))

    if not all_token_ids:
        return [(w, ph, "") for w, ph in word_phonemes]

    # Run force alignment
    blank_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    alignments = ctc_force_align(logits, all_token_ids, blank_id=blank_id)

    # Map alignments back to words
    # alignments[i] corresponds to all_token_ids[i]
    result = []
    align_idx = 0

    for i, (word, gt_ph) in enumerate(word_phonemes):
        w_start, w_end = word_boundaries[i]
        n_tokens = w_end - w_start

        if n_tokens == 0:
            result.append((word, gt_ph, ""))
            continue

        # Collect frames for this word's tokens
        word_frames_start = None
        word_frames_end = None

        for j in range(n_tokens):
            if align_idx < len(alignments):
                fs, fe, _ = alignments[align_idx]
                if word_frames_start is None:
                    word_frames_start = fs
                word_frames_end = fe
                align_idx += 1

        if word_frames_start is not None:
            # Decode this word's frames from the actual logits (greedy)
            word_logits = logits[word_frames_start:word_frames_end + 1]
            word_pred_ids = np.argmax(word_logits, axis=-1)

            # CTC decode: collapse repeats, remove blanks
            decoded_ids = []
            prev_id = None
            for pid in word_pred_ids:
                if pid != blank_id and pid != prev_id:
                    decoded_ids.append(int(pid))
                prev_id = pid

            # Convert to phoneme strings
            pred_phones = tokenizer.convert_ids_to_tokens(decoded_ids)
            pred_phones = [p for p in pred_phones if p not in ("<unk>", "<pad>", "<s>", "</s>")]
            pred_str = " ".join(pred_phones)
            result.append((word, gt_ph, pred_str))
        else:
            result.append((word, gt_ph, ""))

    return result


# ── Main ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load test splits
    print("Loading UXTD test split...")
    uxtd_df, uxtd_test_speakers = load_uxtd_test()
    print(f"  UXTD test: {len(uxtd_df)} utterances, {len(uxtd_test_speakers)} speakers")

    print("Loading TaL80 test split...")
    tal80_df, tal80_test_speakers = load_tal80_test()
    print(f"  TaL80 test: {len(tal80_df)} utterances, {len(tal80_test_speakers)} speakers")

    # Generate phoneme references
    print("Generating IPA phoneme references...")
    uxtd_df = add_ref_phonemes(uxtd_df)
    tal80_df = add_ref_phonemes(tal80_df)

    # ── CSV 2: Test speakers ────────────────────────────────────────
    speaker_rows = []
    for s in uxtd_test_speakers:
        speaker_rows.append({"Dataset": "UXTD", "Speaker ID": s})
    for s in tal80_test_speakers:
        speaker_rows.append({"Dataset": "TaL80", "Speaker ID": s})
    speakers_df = pd.DataFrame(speaker_rows)
    speakers_path = os.path.join(OUTPUT_DIR, "benchmark_test_speakers.csv")
    speakers_df.to_csv(speakers_path, index=False)
    print(f"\nTest speakers saved: {speakers_path}")

    # ── ONNX provider setup ─────────────────────────────────────────
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("Using CUDA GPU for inference")
    else:
        providers = ["CPUExecutionProvider"]
        print("WARNING: CUDAExecutionProvider not available, falling back to CPU")

    datasets = [("UXTD", uxtd_df), ("TaL80", tal80_df)]

    # Example utterances for CSV 3 (first from each dataset)
    uxtd_example = uxtd_df.iloc[0]
    tal80_example = tal80_df.iloc[0]
    examples = [("UXTD", uxtd_example), ("TaL80", tal80_example)]
    example_rows = []

    # Pre-compute per-word phonemes for examples
    example_word_phonemes = {}
    for ds_name, ex_row in examples:
        example_word_phonemes[ds_name] = phonemize_words(ex_row["utterance"])
        print(f"\nExample ({ds_name}): \"{ex_row['utterance']}\"")
        for w, ph in example_word_phonemes[ds_name]:
            print(f"  {w} → {ph}")

    # Collect all per-utterance results for summary
    all_results = []

    # Run each model on both test sets
    for model_name, onnx_path, proc_path in MODEL_CONFIGS:
        if not os.path.exists(onnx_path):
            print(f"WARNING: {onnx_path} not found, skipping {model_name}")
            continue

        # Free GPU memory before loading next model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"\nLoading processor: {model_name}...")
        proc = Wav2Vec2Processor.from_pretrained(proc_path)

        print(f"Loading ONNX session: {model_name}...")
        try:
            session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            print(f"  CUDA load failed ({e}), falling back to CPU...")
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        # Run on example utterances (force alignment)
        for ds_name, ex_row in examples:
            print(f"  Force-aligning example: {ds_name} - {ex_row['utterance']}")
            try:
                _, logits, _ = run_inference(session, proc, ex_row["wav_path"])
                word_results = force_align_words(logits, proc, example_word_phonemes[ds_name])
                for word, gt_ph, pred_ph in word_results:
                    example_rows.append({
                        "Dataset": ds_name,
                        "WAV File": os.path.basename(ex_row["wav_path"]),
                        "Text": ex_row["utterance"],
                        "Word": word,
                        "GT Phonemes (IPA)": gt_ph,
                        "Model (finetuned on)": model_name,
                        "Prediction (IPA)": pred_ph,
                    })
            except Exception as e:
                print(f"    Force alignment error: {e}")

        # Run full benchmark on both test sets
        for dataset_name, test_df in datasets:
            print(f"  Running {model_name} on {dataset_name} test ({len(test_df)} utterances)...")

            for idx, row in test_df.iterrows():
                if idx % 100 == 0:
                    print(f"    [{dataset_name}] Processing {idx + 1}/{len(test_df)}...")

                try:
                    hyp, _, inf_time = run_inference(session, proc, row["wav_path"])
                    per = compute_per(row["ref_phonemes"], hyp)
                except Exception as e:
                    hyp, inf_time, per = "", 0.0, 1.0
                    print(f"    Error: {e}")

                all_results.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "per": per,
                })

        del session
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    results_df = pd.DataFrame(all_results)

    # ── CSV 1: Summary ──────────────────────────────────────────────
    summary_rows = []
    for model_name, _, _ in MODEL_CONFIGS:
        for dataset_name in ["UXTD", "TaL80"]:
            subset = results_df[
                (results_df["model"] == model_name) & (results_df["dataset"] == dataset_name)
            ]
            if len(subset) == 0:
                continue
            summary_rows.append({
                "Model (finetuned on)": model_name,
                "Test Split": dataset_name,
                "Mean PER": round(subset["per"].mean(), 4),
                "Median PER": round(subset["per"].median(), 4),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "benchmark_summary_cross_dataset.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    # ── CSV 3: Example predictions ──────────────────────────────────
    example_df = pd.DataFrame(example_rows)
    example_path = os.path.join(OUTPUT_DIR, "benchmark_example_predictions.csv")
    example_df.to_csv(example_path, index=False)
    print(f"Example predictions saved: {example_path}")

    # ── Print summary to console ────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-DATASET BENCHMARK SUMMARY (test split only)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    print("\nForce-aligned example predictions:")
    print(example_df.to_string(index=False))


if __name__ == "__main__":
    main()
