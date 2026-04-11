#!/usr/bin/env python3
"""Export the UXTD top-layer fine-tuned model to ONNX."""

import os
import shutil
import tempfile

from optimum.onnxruntime import ORTModelForCTC
from transformers import Wav2Vec2Processor

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR    = os.path.dirname(PROJECT_DIR)

MODEL_DIR = os.path.join(REPO_DIR, "wav2vec2-uxtd-finetuned-top-layer")
ONNX_OUT  = os.path.join(PROJECT_DIR, "onnx_models", "wav2vec2_uxtd_top_layer.onnx")

print(f"Loading model from: {MODEL_DIR}")
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)

print("Exporting to ONNX via optimum (this may take a few minutes)...")
ort_model = ORTModelForCTC.from_pretrained(MODEL_DIR, export=True)

with tempfile.TemporaryDirectory() as tmpdir:
    ort_model.save_pretrained(tmpdir)
    src = os.path.join(tmpdir, "model.onnx")
    os.makedirs(os.path.dirname(ONNX_OUT), exist_ok=True)
    shutil.copy2(src, ONNX_OUT)

print(f"\nONNX model saved: {ONNX_OUT}")
print(f"  Size: {os.path.getsize(ONNX_OUT) / 1e6:.1f} MB")
print("Done.")
