# CLAUDE.md

This file provides guidance to Claude Code when working inside `wav2vec2_cpp/`.

## Directory Overview

```
wav2vec2_cpp/
├── src/          C++ real-time inference engine
├── scripts/      Python fine-tuning, export, and benchmark scripts
├── app/          Qt6/QML GUI + PySide6 bridge
├── onnx_models/  ONNX model files (~1.2 GB each)
├── tests/        TIMIT WAV test files + UXTD utterance CSV
├── vocab/        IPA phoneme vocab (42 tokens)
├── libs/         Bundled C++ deps (miniaudio, nlohmann/json, ONNX Runtime)
└── output/       Benchmark CSV results
```

## Build & Run (C++)

```bash
mkdir -p build && cd build && cmake .. && make

./build/auto_resample      # real-time microphone inference
./build/chunk_experiment   # batch WAV chunk-size experiment → output/experiment_results_raw.csv
```

C++17 required. ONNX Runtime v1.19.2 is bundled in `libs/onnxruntime/`.

## Python Scripts

All scripts live in `scripts/`. Run from `wav2vec2_cpp/` unless noted.

```bash
pip install -r scripts/requirements.txt
```

### Fine-tuning

| Script | Purpose |
|--------|---------|
| `finetune_wav2vec2_uxtd.py` | Train only lm_head on UXTD child speech (top-layer linear probing) |

- **Base model:** `facebook/wav2vec2-lv-60-espeak-cv-ft` (XLS-R-300M, IPA CTC)
- **Strategy:** All layers frozen except `lm_head` (~403K trainable params). Preserves pre-trained representations.
- **Config:** lr=1e-3, 15 epochs, 100 warmup steps, bf16, batch 2 × 8 grad accum
- **Output:** `../wav2vec2-uxtd-finetuned-top-layer/` (safetensors + processor)
- **Logging:** WandB project `wav2vec2-uxtd-top-layer-finetuning`
- **Requires:** GPU (CUDA), `espeak-ng` for phonemizer

### ONNX Export

| Script | Input | Output |
|--------|-------|--------|
| `export_uxtd_top_layer_to_onnx.py` | `../wav2vec2-uxtd-finetuned-top-layer/` | `onnx_models/wav2vec2_uxtd_top_layer.onnx` |
| `export_combined_to_onnx.py` | Lightning `.ckpt` | `onnx_models/wav2vec2_combined.onnx` |

### Benchmarking

| Script | Models compared |
|--------|----------------|
| `benchmark_original_vs_finetuned.py` | `wav2vec2_original.onnx` vs `wav2vec2_uxtd_top_layer.onnx` |

Results → `output/benchmark_original_vs_finetuned.csv` (per-utterance) and `..._summary.csv`.

## Key Paths

- ONNX models: `onnx_models/` (`wav2vec2_original.onnx`, `wav2vec2_uxtd_top_layer.onnx`, etc.)
- Vocab: `vocab/vocab.json` (42 IPA phoneme tokens)
- Test audio: `tests/TEST_DR1_FAKS0_SA*/` (TIMIT WAV, 16-bit mono 16 kHz)
- UXTD utterances: `tests/utterances_by_length.csv`
- Speaker splits: `/home/ultraspeech-dev/ultrasuite/core-uxtd/doc/speakers`

## Implementation Notes

- **Vocab fix:** Space token (id=392) exceeds `vocab_size=392` in the base model config. All fine-tuning and export scripts resize `lm_head` from 392→393 to fix this.
- **Audio normalization:** Per-chunk zero-mean, unit-variance with epsilon 1e-5.
- **CTC decoding:** Argmax → collapse consecutive duplicates → skip special tokens → replace `|`/`▁` with space.
- **Speaker-based splits:** UXTD train/dev/test splits are by speaker (no speaker overlap) using the official split file.
