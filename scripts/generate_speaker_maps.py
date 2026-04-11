#!/usr/bin/env python3
"""Generate speaker_map.json files for UXTD and TaL80 to ensure consistent test splits.

This ensures the benchmark uses the EXACT SAME test speakers that were used during training,
preventing data leakage.
"""

import json
import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(PROJECT_DIR)

# ─────────────────────────────────────────────────────────────────────
# UXTD: Use speakers file splits
# ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("UXTD SPEAKER MAP")
print("=" * 70)

UXTD_SPEAKERS_PATH = "/home/ultraspeech-dev/ultrasuite/core-uxtd/doc/speakers"
uxtd_speakers_df = pd.read_csv(UXTD_SPEAKERS_PATH, sep="\t")

print(f"Loaded {len(uxtd_speakers_df)} speakers from {UXTD_SPEAKERS_PATH}")
print("\nUXTD Speaker splits:")
print(uxtd_speakers_df["subset"].value_counts())

uxtd_speaker_map = {
    "train": sorted(uxtd_speakers_df[uxtd_speakers_df["subset"] == "train"]["speaker_id"].tolist()),
    "val": sorted(uxtd_speakers_df[uxtd_speakers_df["subset"] == "dev"]["speaker_id"].tolist()),
    "test": sorted(uxtd_speakers_df[uxtd_speakers_df["subset"] == "test"]["speaker_id"].tolist()),
}

uxtd_map_path = os.path.join(PROJECT_DIR, "output", "uxtd_speaker_map.json")
os.makedirs(os.path.dirname(uxtd_map_path), exist_ok=True)
with open(uxtd_map_path, "w") as f:
    json.dump(uxtd_speaker_map, f, indent=2)
print(f"\n✓ Saved UXTD speaker map to: {uxtd_map_path}")
print(f"  Train: {len(uxtd_speaker_map['train'])} speakers")
print(f"  Val:   {len(uxtd_speaker_map['val'])} speakers")
print(f"  Test:  {len(uxtd_speaker_map['test'])} speakers")

# ─────────────────────────────────────────────────────────────────────
# TaL80: Use seed 42 random split (SAME as training)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TaL80 SPEAKER MAP (using seed 42 - same as training)")
print("=" * 70)

TAL80_CSV = os.path.join(PROJECT_DIR, "output", "tal80_utterances_by_length.csv")
tal80_df = pd.read_csv(TAL80_CSV)
tal80_speakers = sorted(tal80_df["speaker"].unique())

# Use EXACT SAME split as training scripts (seed 42, 70/15/15)
TRAIN_RATIO = 0.70
DEV_RATIO = 0.15

rng = np.random.RandomState(42)
shuffled_speakers = tal80_speakers.copy()
rng.shuffle(shuffled_speakers)

n_train = int(len(shuffled_speakers) * TRAIN_RATIO)
n_dev = int(len(shuffled_speakers) * DEV_RATIO)

tal80_speaker_map = {
    "train": sorted(shuffled_speakers[:n_train]),
    "val": sorted(shuffled_speakers[n_train : n_train + n_dev]),
    "test": sorted(shuffled_speakers[n_train + n_dev :]),
}

# Save new speaker map
tal80_new_map_path = os.path.join(PROJECT_DIR, "output", "tal80_speaker_map_training.json")
with open(tal80_new_map_path, "w") as f:
    json.dump(tal80_speaker_map, f, indent=2)
print(f"\n✓ Saved TaL80 speaker map (TRAINING) to: {tal80_new_map_path}")
print(f"  Train: {len(tal80_speaker_map['train'])} speakers")
print(f"  Val:   {len(tal80_speaker_map['val'])} speakers")
print(f"  Test:  {len(tal80_speaker_map['test'])} speakers")

# Compare with existing speaker_map.json
print("\n" + "=" * 70)
print("COMPARISON: Existing speaker_map.json vs Training split")
print("=" * 70)

existing_map_path = os.path.join(PROJECT_DIR, "output", "speaker_map.json")
with open(existing_map_path) as f:
    existing_map = json.load(f)

print(f"\nExisting speaker_map.json:")
print(f"  Train: {len(existing_map['train'])} speakers")
print(f"  Val:   {len(existing_map['val'])} speakers")
print(f"  Test:  {len(existing_map['test'])} speakers")

print(f"\nTraining split (seed 42):")
print(f"  Train: {len(tal80_speaker_map['train'])} speakers")
print(f"  Val:   {len(tal80_speaker_map['val'])} speakers")
print(f"  Test:  {len(tal80_speaker_map['test'])} speakers")

# Check overlap
existing_test = set(existing_map["test"])
training_test = set(tal80_speaker_map["test"])
overlap = existing_test & training_test
missing_in_existing = training_test - existing_test
extra_in_existing = existing_test - training_test

print(f"\nTest set overlap analysis:")
print(f"  Speakers in BOTH test sets: {len(overlap)}")
print(f"  Speakers in training split but NOT in existing map: {len(missing_in_existing)}")
if missing_in_existing:
    print(f"    {sorted(missing_in_existing)}")
print(f"  Speakers in existing map but NOT in training split: {len(extra_in_existing)}")
if extra_in_existing:
    print(f"    {sorted(extra_in_existing)}")

if existing_test == training_test:
    print("\n  ✓ TEST SETS ARE IDENTICAL - No data leakage!")
else:
    print("\n  ⚠️  TEST SETS DIFFER - DATA LEAKAGE DETECTED!")
    print("     Benchmark may have tested on training speakers!")

# ─────────────────────────────────────────────────────────────────────
# Export test split utterances as CSV
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXPORTING TEST SPLIT CSV FILES")
print("=" * 70)

# UXTD test split
uxtd_csv_path = os.path.join(PROJECT_DIR, "tests", "utterances_by_length.csv")
uxtd_all_df = pd.read_csv(uxtd_csv_path)
uxtd_test_df = uxtd_all_df[uxtd_all_df["speaker"].isin(uxtd_speaker_map["test"])].reset_index(drop=True)
uxtd_test_csv = os.path.join(PROJECT_DIR, "output", "uxtd_test_split.csv")
uxtd_test_df.to_csv(uxtd_test_csv, index=False)
print(f"\n✓ UXTD test split: {uxtd_test_csv}")
print(f"  {len(uxtd_test_df)} utterances, {len(uxtd_speaker_map['test'])} speakers")

# TaL80 test split (TRAINING-BASED, not existing map)
tal80_test_df = tal80_df[tal80_df["speaker"].isin(tal80_speaker_map["test"])].reset_index(drop=True)
tal80_test_csv = os.path.join(PROJECT_DIR, "output", "tal80_test_split_training.csv")
tal80_test_df.to_csv(tal80_test_csv, index=False)
print(f"\n✓ TaL80 test split (TRAINING): {tal80_test_csv}")
print(f"  {len(tal80_test_df)} utterances, {len(tal80_speaker_map['test'])} speakers")

# TaL80 test split (EXISTING MAP)
tal80_existing_test_df = tal80_df[tal80_df["speaker"].isin(existing_map["test"])].reset_index(drop=True)
tal80_existing_test_csv = os.path.join(PROJECT_DIR, "output", "tal80_test_split_existing.csv")
tal80_existing_test_df.to_csv(tal80_existing_test_csv, index=False)
print(f"\n✓ TaL80 test split (EXISTING): {tal80_existing_test_csv}")
print(f"  {len(tal80_existing_test_df)} utterances, {len(existing_map['test'])} speakers")

print("\n" + "=" * 70)
print("DATA LEAKAGE ANALYSIS COMPLETE")
print("=" * 70)
if existing_test == training_test:
    print("\n✓ NO DATA LEAKAGE: Benchmark uses correct test speakers")
else:
    print("\n⚠️  DATA LEAKAGE DETECTED!")
    print("   The benchmark may have tested models on speakers used in training.")
    print("   Use tal80_test_split_training.csv for accurate evaluation.")
