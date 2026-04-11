#!/usr/bin/env python3
"""Export the combined (UXTD+TaL80) Lightning checkpoint to ONNX.

Loads best checkpoint (epoch 27), reconstructs Wav2Vec2ForCTC,
then exports via optimum to wav2vec2_cpp/onnx_models/wav2vec2_combined.onnx.
"""

import os
import shutil
import tempfile

import torch
from optimum.onnxruntime import ORTModelForCTC
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
)

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR    = os.path.dirname(PROJECT_DIR)

CKPT_PATH  = os.path.join(REPO_DIR, "wav2vec2-combined-finetuned",
                          "combined-epoch=27-val_per=0.0000.ckpt")
ONNX_OUT   = os.path.join(PROJECT_DIR, "onnx_models", "wav2vec2_combined.onnx")
BASE_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"

# ── Step 1: load checkpoint ───────────────────────────────────────────
print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
state_dict = {
    k[len("model."):]: v
    for k, v in ckpt["state_dict"].items()
    if k.startswith("model.")
}
print(f"  Extracted {len(state_dict)} tensors from state_dict")

# ── Step 2: build HuggingFace model ───────────────────────────────────
print(f"Loading base processor from {BASE_MODEL}...")
processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
vocab_size = len(processor.tokenizer.get_vocab())  # 393
print(f"  Vocab size: {vocab_size}")

print(f"Loading base model from {BASE_MODEL}...")
model = Wav2Vec2ForCTC.from_pretrained(BASE_MODEL)

# Resize lm_head to match combined training (393 tokens)
if model.config.vocab_size < vocab_size:
    old_w = model.lm_head.weight.data
    old_b = model.lm_head.bias.data
    model.lm_head = torch.nn.Linear(old_w.shape[1], vocab_size)
    model.lm_head.weight.data[: old_w.shape[0]] = old_w
    model.lm_head.bias.data[: old_b.shape[0]] = old_b
    model.config.vocab_size = vocab_size
    print(f"  Resized lm_head: {old_w.shape[0]} → {vocab_size}")

model.load_state_dict(state_dict, strict=True)
model.eval()
print("  State dict loaded successfully")

# ── Step 3: save as HuggingFace model to temp dir ─────────────────────
print("Saving to temporary HuggingFace directory...")
with tempfile.TemporaryDirectory() as tmpdir:
    model.save_pretrained(tmpdir)
    processor.save_pretrained(tmpdir)

    # ── Step 4: export to ONNX via optimum ───────────────────────────
    print("Exporting to ONNX via optimum (this may take a few minutes)...")
    ort_model = ORTModelForCTC.from_pretrained(tmpdir, export=True)

    onnx_save_dir = os.path.join(tmpdir, "onnx_export")
    ort_model.save_pretrained(onnx_save_dir)

    # optimum saves as <dir>/model.onnx — copy to our target path
    src = os.path.join(onnx_save_dir, "model.onnx")
    os.makedirs(os.path.dirname(ONNX_OUT), exist_ok=True)
    shutil.copy2(src, ONNX_OUT)

print(f"\nONNX model saved: {ONNX_OUT}")
print(f"  Size: {os.path.getsize(ONNX_OUT) / 1e6:.1f} MB")
print("Done.")
