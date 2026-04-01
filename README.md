# Wav2Vec2 Speech-to-Phoneme Recognition

Speech-to-phoneme recognition system built around the XLS-R-300M wav2vec2 model. The project has evolved through three phases: a C++ real-time inference engine, a Qt6/QML GUI, and a fine-tuning pipeline for adapting the model to child and adult speech corpora.

## Background

### Phase 1: C++ Real-Time Inference Engine

The initial goal was to run wav2vec2 phoneme recognition in real time from a microphone using C++ and ONNX Runtime. The base model (`facebook/wav2vec2-lv-60-espeak-cv-ft`) was exported to ONNX and paired with a miniaudio-based audio capture loop. A chunk-size experiment tool was also built to determine the minimum viable audio segment for inference (400 samples, due to the wav2vec2 convolutional receptive field).

### Phase 2: Qt6/QML GUI

A desktop GUI was added using Qt6/QML with a PySide6 Python bridge. The app spawns the C++ `auto_resample` binary as a subprocess and parses its tagged output (`TR:` lines) to display live transcriptions.

### Phase 3: Fine-Tuning (Current Work)

The current focus is adapting the base model to domain-specific speech using CTC fine-tuning:

1. **UXTD fine-tuning** — Fine-tuned on the UXTD child speech corpus to improve phoneme recognition for children's speech. Text prompts are converted to IPA phonemes via `phonemizer` (espeak backend, no stress marks). The CNN feature encoder is frozen; only the transformer and CTC head are trained.

2. **TaL80 fine-tuning** — Continued fine-tuning from the UXTD checkpoint on the TaL80 adult speech corpus (48 kHz source audio, resampled to 16 kHz). Includes fixes for CTC NaN loss (filtering utterances >10s), bf16 training, lower learning rate (3e-5), and gradient clipping.

3. **3-model benchmark** — A benchmark script compares Phoneme Error Rate (PER) and inference time across the original, UXTD-finetuned, and TaL80-finetuned models using ONNX inference on the UXTD test set.

## Prerequisites

- C++17 compiler
- CMake
- Python 3 (for scripts and GUI)
- Qt6 + PySide6 (for GUI only)
- GPU + CUDA (for fine-tuning)
- `espeak-ng` (for phonemizer)

## Install Python Dependencies

```bash
pip install -r scripts/requirements.txt
```

Core packages: `torch`, `transformers`, `onnxruntime`, `optimum`, `soundfile`, `numpy`.

Fine-tuning packages: `datasets`, `evaluate`, `phonemizer`, `librosa`, `pandas`, `wandb`.

## Build (C++ Targets)

```bash
mkdir -p build && cd build
cmake ..
make
```

### CMake Targets

| Target | Description |
|---|---|
| `auto_resample` | Real-time microphone capture + inference |
| `chunk_experiment` | Batch WAV processing at various chunk sizes |
| `InferenceRunner` | Qt6 GUI app (only built if Qt6 is found) |

## Run

```bash
# Real-time microphone inference
./build/auto_resample

# Batch chunk-size experiment (results -> output/experiment_results_raw.csv)
./build/chunk_experiment

# GUI (requires Qt6 + PySide6)
cd app && python main.py
```

## Fine-Tuning

### UXTD Child Speech

```bash
python scripts/finetune_wav2vec2_uxtd.py
```

- **Base model:** `facebook/wav2vec2-lv-60-espeak-cv-ft` (IPA phoneme CTC, 393 vocab tokens)
- **Dataset:** UXTD child speech corpus. Utterances from `tests/utterances_by_length.csv`; speaker splits from UltraSuite docs.
- **Audio:** 22050 Hz source, resampled to 16 kHz. Max duration 10s.
- **Training:** batch 2 x 8 grad accum, lr 3e-4, 500 warmup steps, 30 epochs, fp16, gradient checkpointing. Feature encoder frozen.
- **Metric:** Phoneme Error Rate (PER) via WER on phoneme sequences. Best model by lowest dev PER.
- **Output:** `wav2vec2-uxtd-finetuned/`

### TaL80 Adult Speech

```bash
python scripts/finetune_wav2vec2_tal80.py
```

- **Base model:** UXTD fine-tuned checkpoint (`wav2vec2-uxtd-finetuned/checkpoint-784`)
- **Dataset:** TaL80 adult speech corpus. 48 kHz source audio, resampled to 16 kHz.
- **Training:** bf16, lr 3e-5, gradient clipping. Utterances >10s filtered to prevent NaN loss.
- **Output:** `scripts/wav2vec2-tal80-finetuned/`
- **Logging:** Weights & Biases (dataset stats, sample predictions, GPU memory, grad norms)

### Benchmarking

```bash
python scripts/benchmark_original_vs_finetuned.py
```

Compares original, UXTD-finetuned, and TaL80-finetuned models (ONNX) on the UXTD test set. Outputs per-utterance and summary CSVs to `output/`.

## Architecture

- **`src/realtime_autoresample.cpp`** — Real-time engine. Captures audio via miniaudio, buffers with mutex, runs ONNX inference every ~1s.
- **`src/chunk_experiment.cpp`** — Batch WAV processing at chunk sizes 50–5000ms.
- **`app/`** — Qt6/QML GUI with PySide6 Python bridge.
- **`scripts/finetune_wav2vec2_uxtd.py`** — UXTD child speech fine-tuning pipeline.
- **`scripts/finetune_wav2vec2_tal80.py`** — TaL80 adult speech fine-tuning (from UXTD checkpoint).
- **`scripts/benchmark_original_vs_finetuned.py`** — 3-model PER and latency comparison.
- **`scripts/export_to_onnx.py`** — Model export to ONNX format.

## Key Paths

| Path | Description |
|---|---|
| `onnx_output/model.onnx` | Base ONNX model (~1.2 GB) |
| `onnx_models/` | All ONNX models (original, UXTD, TaL80) for benchmarking |
| `vocab/vocab.json` | Vocabulary (42 IPA phoneme tokens) |
| `libs/` | Bundled libraries (miniaudio, nlohmann/json, ONNX Runtime v1.19.2) |
| `tests/` | TIMIT test WAV files and UXTD utterance CSV |

## Implementation Notes

- Audio normalization is done per-chunk: zero-mean, unit-variance with epsilon 1e-5.
- CTC greedy decoding: argmax over logits, collapse consecutive duplicates, skip special tokens (`[PAD]`, `<pad>`, `<s>`, `</s>`, `<unk>`), replace `|` with space.
- The base model has a known off-by-one where the space token (id=392) exceeds `vocab_size` (392); fine-tuning scripts detect and fix this by resizing `lm_head`.
- Linux is the primary platform; macOS and Windows are supported.
