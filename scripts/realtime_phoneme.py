"""
Real-time phoneme extraction from microphone using:
  - Model : facebook/wav2vec2-lv-60-espeak-cv-ft
  - Decoder: pyctcdecode (BeamSearch CTC)

Usage:
    python realtime_phoneme.py [--chunk-duration SECONDS] [--device DEVICE_INDEX]
"""

import argparse
import queue
import sys
import threading

import numpy as np
import sounddevice as sd
import torch
from pyctcdecode.decoder import build_ctcdecoder
from transformers import Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2Processor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SAMPLE_RATE = 16_000  # model expects 16 kHz
BLOCK_DURATION = 0.1  # seconds per sounddevice callback block


def build_decoder(model_name: str):
    """Build a pyctcdecode BeamSearchDecoder from the model vocabulary."""
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(model_name)
    vocab_dict = tokenizer.get_vocab()
    labels = [tok for tok, _ in sorted(vocab_dict.items(), key=lambda x: x[1])]
    labels[tokenizer.pad_token_id] = ""
    return build_ctcdecoder(labels)


def process_loop(
    audio_queue: queue.Queue, processor, model, decoder, chunk_samples: int
):
    """Consume audio from the queue, run inference, and print phonemes."""
    buffer = np.array([], dtype=np.float32)

    while True:
        try:
            chunk = audio_queue.get(timeout=1.5)
        except queue.Empty:
            continue

        buffer = np.concatenate([buffer, chunk])

        while len(buffer) >= chunk_samples:
            audio_chunk = buffer[:chunk_samples]
            buffer = buffer[chunk_samples:]

            inputs = processor(
                audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            with torch.no_grad():
                logits = model(inputs.input_values).logits  # (1, T, vocab)

            phonemes = decoder.decode(logits[0].numpy())
            if phonemes.strip():
                print(f"Phonemes: {phonemes}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Real-time phoneme extractor")
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=1.5,
        help="Audio chunk duration in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Sounddevice input device index (default: system default)",
    )
    args = parser.parse_args()

    chunk_samples = int(SAMPLE_RATE * args.chunk_duration)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading {MODEL_NAME} …", flush=True)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.\n", flush=True)

    decoder = build_decoder(MODEL_NAME)

    # ------------------------------------------------------------------
    # Audio pipeline
    # ------------------------------------------------------------------
    audio_queue: queue.Queue = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    proc_thread = threading.Thread(
        target=process_loop,
        args=(audio_queue, processor, model, decoder, chunk_samples),
        daemon=True,
    )
    proc_thread.start()

    print(f"Listening … (chunk={args.chunk_duration}s, device={args.device})")
    print("Press Ctrl+C to stop.\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
        device=args.device,
        callback=audio_callback,
    ):
        try:
            while True:
                sd.sleep(200)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
