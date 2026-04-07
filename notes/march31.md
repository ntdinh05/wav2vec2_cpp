Report: TaL80 Fine-tuning Session

  Objective

  Fine-tune wav2vec2 (from UXTD child speech checkpoint) on the TaL80 adult speech corpus for phoneme recognition.

  Timeline of Work

  1. Initial Setup & W&B Integration
  - Added Weights & Biases logging to finetune_wav2vec2_tal80.py
  - Logged dataset stats (duration histograms, phoneme distribution, split sizes), model params (315.8M total, 311.6M trainable), GPU
  memory, sample predictions per epoch, gradient norms, and learning rate

  2. Run #1 — AttributeError
  - Crash on startup: torch.cuda.get_device_properties(0).total_mem doesn't exist
  - Fix: Changed to .total_memory

  3. Run #2 — NaN Loss + Serialization Crash (fp16, lr=1e-4)
  - Loss diverged: 196 -> 94 -> 89 -> inf -> nan (step 200 onward)
  - Crashed at epoch 1 checkpoint save: TypeError: Object of type EspeakBackend is not JSON serializable
  - Serialization fix: tokenizer.backend = "espeak" — importing phonemizer replaced the string attribute with a live Python object
  - NaN diagnosis (initial): Suspected fp16 overflow (max 65504) from high CTC loss. Switched fp16=True to bf16=True, lowered LR from 1e-4
  to 3e-5, added max_grad_norm=1.0

  4. Run #3 — Still NaN (bf16, lr=3e-5)
  - Same pattern: loss 126 -> 116 -> nan at step 200
  - Proved the issue was NOT dtype or learning rate

  5. Root Cause Found — CTC Label/Audio Length Mismatch
  - The data collator truncated audio to 10s but kept full-length phoneme labels
  - 502 utterances (3%) were >10s (up to 60s). A 50s utterance truncated to 10s had 646 phoneme labels but only 500 output frames
  - CTC loss returns inf when label_length > input_length (alignment is mathematically impossible)
  - Once any batch produces inf loss, gradients become nan and corrupt model weights permanently

  6. Final Fix
  - Added pre-training filter: drop all utterances >10s from the dataframe (line 214-224)
  - Removed the now-unnecessary audio truncation from the data collator
  - Removed leftover truncation from the W&B prediction callback

  Summary of All Changes to finetune_wav2vec2_tal80.py

  ┌───────────────────────────────────────────┬──────────────────────────────────────────────────────────────┐
  │                  Change                   │                             Why                              │
  ├───────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ tokenizer.backend = "espeak"              │ Prevent EspeakBackend serialization crash on checkpoint save │
  ├───────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ Filter utterances >10s before training    │ Prevent CTC inf loss from label/audio length mismatch        │
  ├───────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ Removed audio truncation in data collator │ No longer needed after filtering                             │
  ├───────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ fp16=True -> bf16=True                    │ Larger dynamic range, safer for high CTC losses              │
  ├───────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ learning_rate 1e-4 -> 3e-5                │ More conservative for sequential fine-tuning                 │
  ├───────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ max_grad_norm=1.0                         │ Clip gradient spikes                                         │
  ├───────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ logging_nan_inf_filter=False              │ Surface NaN/Inf instead of silently filtering                │
  └───────────────────────────────────────────┴──────────────────────────────────────────────────────────────┘

  Also Done

  - Pushed wave2vec2_cpp/ to GitHub (ntdinh05/wave2vec2_cpp, commit 99f1fe7)
  - Updated .gitignore to exclude __pycache__/, wandb/, *.pyc, *.log
  - Drafted content for 2 Google Slides summarizing the fine-tuning approach
