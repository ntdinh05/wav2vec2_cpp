 Plan: Combined UXTD+TaL80 Fine-tuning with PyTorch Lightning + WandB

 Context

 The project currently has two separate fine-tuning scripts (UXTD child
 speech, TaL80 adult speech) using HuggingFace Trainer. The goal is to
 create a single training script that combines both datasets and uses
 PyTorch Lightning instead, with WandB logging. This trains from the
 original facebook/wav2vec2-lv-60-espeak-cv-ft base model.

 Branch

 - Create branch feature/combined-lightning-training from up-to-date main

 New file

 - scripts/finetune_wav2vec2_combined.py — single self-contained script

 Implementation

 1. Data Loading

 - Load both CSVs:
   - UXTD: wav2vec2_cpp/tests/utterances_by_length.csv (1,138 utterances,
  22050 Hz)
   - TaL80: wav2vec2_cpp/output/tal80_utterances_by_length.csv (16,315
 utterances, 48000 Hz)
 - Derive wav_path from filepath column (.txt → .wav), verify existence
 - Add dataset column ("uxtd" / "tal80") and source_sr column (22050 /
 48000)
 - Speaker splits:
   - UXTD: Load from
 /home/ultraspeech-dev/ultrasuite/core-uxtd/doc/speakers (tab-separated,
 has speaker_id → subset mapping)
   - TaL80: Compute 70/15/15 split by speaker with seed=42 (same logic as
  existing tal80 script)
 - Filter utterances > 10s
 - Phonemize all unique utterances with espeak (same pipeline:
 phonemize() with Separator(phone=" ", word="  ", syllable=""))

 2. PyTorch Lightning DataModule

 - CombinedSpeechDataModule(pl.LightningDataModule):
   - setup(): Build train/val/test splits from combined DataFrame, create
  Dataset objects with pre-tokenized labels
   - train_dataloader(): Use WeightedRandomSampler — compute per-sample
 weights so UXTD and TaL80 are drawn with equal probability per batch
   - val_dataloader() / test_dataloader(): Standard sequential loaders
   - Custom collate function (lazy audio loading + resampling in
 collator, same as TaL80 script pattern):
       - Load audio via soundfile.read()
     - Resample to 16kHz via librosa.resample()
     - Process through Wav2Vec2Processor
     - Pad inputs and labels, mask label padding with -100

 3. PyTorch Lightning Module

 - Wav2Vec2FineTuner(pl.LightningModule):
   - __init__(): Load
 Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft"),
 apply vocab size fix (392→393), freeze feature encoder
   - forward(): Pass input_values and labels through model, return CTC
 loss + logits
   - training_step(): Forward pass, log train/loss, train/lr
   - validation_step(): Accumulate predictions and labels
   - on_validation_epoch_end(): Compute PER via jiwer/evaluate, log
 val/per
   - test_step() / on_test_epoch_end(): Same as validation, log test/per
   - configure_optimizers(): AdamW with lr=3e-4, linear warmup (500
 steps) + linear decay via get_linear_schedule_with_warmup

 4. WandB Integration

 - WandbLogger from pytorch_lightning.loggers
   - Project: "wav2vec2-combined-finetune"
   - Log config: dataset stats, GPU info, model params, split counts,
 source datasets
 - Log dataset duration histograms, phoneme distributions at init
 - Custom WandbPredictionCallback(pl.Callback): log sample prediction
 tables each validation epoch
 - Log GPU memory stats

 5. Trainer Setup

 - pl.Trainer with:
   - max_epochs=30
   - accumulate_grad_batches=8
   - precision="bf16-mixed" (CTC numerical stability)
   - gradient_clip_val=1.0
   - callbacks: ModelCheckpoint (monitor val/per, save top 3),
 EarlyStopping (optional), LearningRateMonitor
   - logger=WandbLogger(...)
 - After training: save best model + processor, run test evaluation,
 wandb.finish()

 Key files referenced

 - scripts/finetune_wav2vec2_tal80.py — TaL80 patterns (lazy loading,
 bf16, WandB callback, vocab fix)
 - scripts/finetune_wav2vec2_uxtd.py — UXTD patterns (speaker splits from
  file, base model loading)
 - scripts/requirements.txt — will need pytorch-lightning and wandb added

 Verification

 - User will run the script manually on GPU
 - Check WandB dashboard for logged metrics, histograms, prediction
 tables
 - Verify both UXTD and TaL80 data appears in training (log counts per
 dataset per batch)
