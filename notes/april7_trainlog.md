(.venv) ultraspeech-dev@ultraspeech-dev-msi:~/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts$ python finetune_wav2vec2_combined.py
2026-04-06 13:29:18,284 | INFO | Logger initialized.
2026-04-06 13:29:18,377 | INFO | Using GPU: NVIDIA GeForce RTX 5070 Laptop GPU (CUDA 12.8)
2026-04-06 13:29:18,377 | INFO | Loading UXTD utterances...
2026-04-06 13:29:18,386 | INFO | UXTD: 1138 utterances, splits: {'train': 778, 'test': 252, 'dev': 108}
2026-04-06 13:29:18,386 | INFO | Loading TaL80 utterances...
2026-04-06 13:29:18,410 | INFO | TaL80: 16315 utterances, splits: {'train': 11226, 'test': 2628, 'dev': 2461}
2026-04-06 13:29:18,431 | INFO | Combined total utterances with audio: 17453
2026-04-06 13:29:18,431 | INFO | Filtering utterances longer than 10s...
2026-04-06 13:29:18,902 | INFO | Dropped 552 utterances > 10s. Remaining: 16901
2026-04-06 13:29:18,902 | INFO | Converting text to IPA phonemes...
2026-04-06 13:29:20,054 | INFO | Sample: 'thank you scissors helicopter bridge' -> 'θ æ ŋ k  j uː  s ɪ s ɚ z  h ɛ l ɪ k ɑː p t ɚ  b ɹ ɪ dʒ'
2026-04-06 13:29:20,054 | INFO | Loading processor from facebook/wav2vec2-lv-60-espeak-cv-ft...
2026-04-06 13:29:45,051 | INFO |   train: 11618 total ({'tal80': 10878, 'uxtd': 740})
2026-04-06 13:29:45,052 | INFO |   dev: 2474 total ({'tal80': 2369, 'uxtd': 105})
2026-04-06 13:29:45,053 | INFO |   test: 2809 total ({'tal80': 2566, 'uxtd': 243})
2026-04-06 13:29:45,053 | INFO | Initializing W&B...
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/ultraspeech-dev/.netrc.
wandb: Currently logged in as: ntdinh05 (ntdinh05-university-at-buffalo) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in /home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wandb/run-20260406_132946-ckdnjba0
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run combined-uxtd-tal80-lightning
wandb: ⭐️ View project at https://wandb.ai/ntdinh05-university-at-buffalo/wav2vec2-combined-finetune
wandb: 🚀 View run at https://wandb.ai/ntdinh05-university-at-buffalo/wav2vec2-combined-finetune/runs/ckdnjba0
2026-04-06 13:29:47,858 | INFO | Unique phonemes in training labels: 63
2026-04-06 13:29:47,866 | INFO | Weighted sampler: UXTD weight=0.000676 (740 samples), TaL80 weight=0.000046 (10878 samples)
2026-04-06 13:29:48,310 | INFO | Model params: 315.8M total, 311.6M trainable (98.7%)
Using bfloat16 Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
`Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
2026-04-06 13:29:48,319 | INFO | Starting training...
2026-04-06 13:29:48,419 | INFO | Weighted sampler: UXTD weight=0.000676 (740 samples), TaL80 weight=0.000046 (10878 samples)
/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:881: Checkpoint directory /home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned exists and is not empty.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading `train_dataloader` to estimate number of stepping batches.
/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/model_summary/model_summary.py:242: Precision bf16-mixed is not supported by the model summary.  Estimated model size in MB will not be accurate. Using 32 bits instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type           ┃ Params ┃ Mode ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━┩
│ 0 │ model │ Wav2Vec2ForCTC │  315 M │ eval │     0 │
└───┴───────┴────────────────┴────────┴──────┴───────┘
Trainable params: 311 M
Non-trainable params: 4.2 M
Total params: 315 M
Total estimated model params size (MB): 1.3 K
Modules in train mode: 0
Modules in eval mode: 409
Total FLOPs: 0
Sanity Checking ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2/2 0:00:00 • 0:00:00 38.27it/s  2026-04-06 13:30:39,089 | INFO | Epoch 0 — val/per: 0.2679
/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/fit_loop.py:534: Found 409 module(s) in eval
mode at the start of training. This may lead to unexpected behavior during training. If this is intentional, you can ignore this warning.
Epoch 0/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:13:20 • 0:00:00 7.22it/s v_num: jba0 train/loss_step: 1.3282026-04-06 13:45:50,360 | INFO | Epoch 0 — val/per: 0.1027
Epoch 0/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:13:20 • 0:00:00 7.22it/s v_num: jba0 train/loss_step: 1.328 val/loss: 16.017 val/per: 0.103 train/loss_epoch: 37.178Epoch 0, global step 727: 'val/per' reached 0.10268 (best 0.10268), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=00-val_per=0.0000.ckpt' as top 3
Epoch 1/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:42 • 0:00:00 7.82it/s v_num: jba0 train/loss_step: 9.049 val/loss: 16.017 val/per: 0.103 train/loss_epoch: 37.1782026-04-06 14:00:42,883 | INFO | Epoch 1 — val/per: 0.0977
Epoch 1, global step 1454: 'val/per' reached 0.09775 (best 0.09775), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=01-val_per=0.0000.ckpt' as top 3
Epoch 2/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:44 • 0:00:00 7.55it/s v_num: jba0 train/loss_step: 0.181 val/loss: 15.079 val/per: 0.098 train/loss_epoch: 16.1142026-04-06 14:15:35,431 | INFO | Epoch 2 — val/per: 0.1001
Epoch 2/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:44 • 0:00:00 7.55it/s v_num: jba0 train/loss_step: 0.181 val/loss: 14.883 val/per: 0.100 train/loss_epoch: 14.365Epoch 2, global step 2181: 'val/per' reached 0.10006 (best 0.09775), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=02-val_per=0.0000.ckpt' as top 3
Epoch 3/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:46 • 0:00:00 7.50it/s v_num: jba0 train/loss_step: 0.164 val/loss: 14.883 val/per: 0.100 train/loss_epoch: 14.3652026-04-06 14:30:27,079 | INFO | Epoch 3 — val/per: 0.0939
Epoch 3, global step 2908: 'val/per' reached 0.09387 (best 0.09387), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=03-val_per=0.0000.ckpt' as top 3
Epoch 4/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:44 • 0:00:00 7.52it/s v_num: jba0 train/loss_step: 0.968 val/loss: 13.896 val/per: 0.094 train/loss_epoch: 13.1202026-04-06 14:45:19,231 | INFO | Epoch 4 — val/per: 0.1061
Epoch 4, global step 3635: 'val/per' was not in top 3
Epoch 5/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:49 • 0:00:00 7.50it/s v_num: jba0 train/loss_step: 0.025 val/loss: 15.211 val/per: 0.106 train/loss_epoch: 12.3032026-04-06 15:00:09,690 | INFO | Epoch 5 — val/per: 0.0956
Epoch 5, global step 4362: 'val/per' reached 0.09561 (best 0.09387), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=05-val_per=0.0000.ckpt' as top 3
Epoch 6/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:45 • 0:00:00 7.57it/s v_num: jba0 train/loss_step: 0.133 val/loss: 14.965 val/per: 0.096 train/loss_epoch: 12.1282026-04-06 15:15:01,936 | INFO | Epoch 6 — val/per: 0.0951
Epoch 6, global step 5089: 'val/per' reached 0.09508 (best 0.09387), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=06-val_per=0.0000.ckpt' as top 3
Epoch 7/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:44 • 0:00:00 7.53it/s v_num: jba0 train/loss_step: 8.283 val/loss: 13.746 val/per: 0.095 train/loss_epoch: 11.6992026-04-06 15:29:53,443 | INFO | Epoch 7 — val/per: 0.0940
Epoch 7, global step 5816: 'val/per' reached 0.09402 (best 0.09387), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=07-val_per=0.0000.ckpt' as top 3
Epoch 8/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:45 • 0:00:00 7.62it/s v_num: jba0 train/loss_step: 0.032 val/loss: 14.952 val/per: 0.094 train/loss_epoch: 10.8102026-04-06 15:44:46,072 | INFO | Epoch 8 — val/per: 0.0891
Epoch 8/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:45 • 0:00:00 7.62it/s v_num: jba0 train/loss_step: 0.032 val/loss: 14.840 val/per: 0.089 train/loss_epoch: 10.322Epoch 8, global step 6543: 'val/per' reached 0.08907 (best 0.08907), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=08-val_per=0.0000.ckpt' as top 3
Epoch 9/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:47 • 0:00:00 7.50it/s v_num: jba0 train/loss_step: 26.661 val/loss: 14.840 val/per: 0.089 train/loss_epoch: 10.3222026-04-06 15:59:51,853 | INFO | Epoch 9 — val/per: 0.0918
Epoch 9, global step 7270: 'val/per' reached 0.09178 (best 0.08907), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=09-val_per=0.0000.ckpt' as top 3
Epoch 10/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:45 • 0:00:00 7.55it/s v_num: jba0 train/loss_step: 0.013 val/loss: 15.124 val/per: 0.092 train/loss_epoch: 10.1742026-04-06 16:14:44,694 | INFO | Epoch 10 — val/per: 0.1015
Epoch 10/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:45 • 0:00:00 7.55it/s v_num: jba0 train/loss_step: 0.013 val/loss: 15.988 val/per: 0.101 train/loss_epoch: 9.465Epoch 10, global step 7997: 'val/per' was not in top 3
Epoch 11/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:47 • 0:00:00 7.50it/s v_num: jba0 train/loss_step: 0.008 val/loss: 15.988 val/per: 0.101 train/loss_epoch: 9.4652026-04-06 16:29:32,743 | INFO | Epoch 11 — val/per: 0.0897
Epoch 11, global step 8724: 'val/per' reached 0.08965 (best 0.08907), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=11-val_per=0.0000.ckpt' as top 3
Epoch 12/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:47 • 0:00:00 7.43it/s v_num: jba0 train/loss_step: 0.002 val/loss: 14.347 val/per: 0.090 train/loss_epoch: 8.8792026-04-06 16:44:36,969 | INFO | Epoch 12 — val/per: 0.0884
Epoch 12, global step 9451: 'val/per' reached 0.08844 (best 0.08844), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=12-val_per=0.0000.ckpt' as top 3
Epoch 13/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:46 • 0:00:00 7.41it/s v_num: jba0 train/loss_step: 6.474 val/loss: 14.507 val/per: 0.088 train/loss_epoch: 8.6632026-04-06 16:59:30,311 | INFO | Epoch 13 — val/per: 0.0877
Epoch 13, global step 10178: 'val/per' reached 0.08774 (best 0.08774), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=13-val_per=0.0000.ckpt' as top 3
Epoch 14/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:45 • 0:00:00 7.59it/s v_num: jba0 train/loss_step: 0.002 val/loss: 14.926 val/per: 0.088 train/loss_epoch: 7.0842026-04-06 17:14:34,666 | INFO | Epoch 14 — val/per: 0.0848
Epoch 14, global step 10905: 'val/per' reached 0.08485 (best 0.08485), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=14-val_per=0.0000.ckpt' as top 3
Epoch 15/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:44 • 0:00:00 7.54it/s v_num: jba0 train/loss_step: 0.016 val/loss: 14.656 val/per: 0.085 train/loss_epoch: 6.9922026-04-06 17:29:26,217 | INFO | Epoch 15 — val/per: 0.0821
Epoch 15, global step 11632: 'val/per' reached 0.08214 (best 0.08214), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=15-val_per=0.0000.ckpt' as top 3
Epoch 16/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:46 • 0:00:00 7.59it/s v_num: jba0 train/loss_step: 0.001 val/loss: 14.618 val/per: 0.082 train/loss_epoch: 6.6802026-04-06 17:44:29,022 | INFO | Epoch 16 — val/per: 0.0818
Epoch 16/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:46 • 0:00:00 7.59it/s v_num: jba0 train/loss_step: 0.001 val/loss: 15.021 val/per: 0.082 train/loss_epoch: 6.074Epoch 16, global step 12359: 'val/per' reached 0.08176 (best 0.08176), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=16-val_per=0.0000.ckpt' as top 3
Epoch 17/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:44 • 0:00:00 7.64it/s v_num: jba0 train/loss_step: 0.002 val/loss: 15.021 val/per: 0.082 train/loss_epoch: 6.0742026-04-06 17:59:18,271 | INFO | Epoch 17 — val/per: 0.0809
Epoch 17, global step 13086: 'val/per' reached 0.08091 (best 0.08091), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=17-val_per=0.0000.ckpt' as top 3
Epoch 18/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:44 • 0:00:00 7.54it/s v_num: jba0 train/loss_step: 0.002 val/loss: 14.934 val/per: 0.081 train/loss_epoch: 5.0472026-04-06 18:14:07,352 | INFO | Epoch 18 — val/per: 0.0810
Epoch 18/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:44 • 0:00:00 7.54it/s v_num: jba0 train/loss_step: 0.002 val/loss: 15.024 val/per: 0.081 train/loss_epoch: 4.723Epoch 18, global step 13813: 'val/per' reached 0.08102 (best 0.08091), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=18-val_per=0.0000.ckpt' as top 3
Epoch 19/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:44 • 0:00:00 7.63it/s v_num: jba0 train/loss_step: 0.001 val/loss: 15.024 val/per: 0.081 train/loss_epoch: 4.7232026-04-06 18:28:58,559 | INFO | Epoch 19 — val/per: 0.0799
Epoch 19, global step 14540: 'val/per' reached 0.07995 (best 0.07995), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=19-val_per=0.0000.ckpt' as top 3
Epoch 20/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:46 • 0:00:00 7.49it/s v_num: jba0 train/loss_step: 0.004 val/loss: 15.996 val/per: 0.080 train/loss_epoch: 4.5172026-04-06 18:43:49,888 | INFO | Epoch 20 — val/per: 0.0800
Epoch 20/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:46 • 0:00:00 7.49it/s v_num: jba0 train/loss_step: 0.004 val/loss: 15.277 val/per: 0.080 train/loss_epoch: 3.779Epoch 20, global step 15267: 'val/per' reached 0.08004 (best 0.07995), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=20-val_per=0.0000.ckpt' as top 3
Epoch 21/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:46 • 0:00:00 7.58it/s v_num: jba0 train/loss_step: 0.002 val/loss: 15.277 val/per: 0.080 train/loss_epoch: 3.7792026-04-06 18:58:43,767 | INFO | Epoch 21 — val/per: 0.0790
Epoch 21, global step 15994: 'val/per' reached 0.07898 (best 0.07898), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=21-val_per=0.0000.ckpt' as top 3
Epoch 22/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:44 • 0:00:00 7.63it/s v_num: jba0 train/loss_step: 0.000 val/loss: 16.610 val/per: 0.079 train/loss_epoch: 2.7592026-04-06 19:13:35,439 | INFO | Epoch 22 — val/per: 0.0781
Epoch 22/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:44 • 0:00:00 7.63it/s v_num: jba0 train/loss_step: 0.000 val/loss: 17.310 val/per: 0.078 train/loss_epoch: 2.289Epoch 22, global step 16721: 'val/per' reached 0.07812 (best 0.07812), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=22-val_per=0.0000.ckpt' as top 3
Epoch 23/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:45 • 0:00:00 7.57it/s v_num: jba0 train/loss_step: 0.001 val/loss: 17.310 val/per: 0.078 train/loss_epoch: 2.2892026-04-06 19:28:27,775 | INFO | Epoch 23 — val/per: 0.0777
Epoch 23, global step 17448: 'val/per' reached 0.07768 (best 0.07768), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=23-val_per=0.0000.ckpt' as top 3
Epoch 24/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:45 • 0:00:00 7.50it/s v_num: jba0 train/loss_step: 0.000 val/loss: 17.537 val/per: 0.078 train/loss_epoch: 1.9942026-04-06 19:43:20,186 | INFO | Epoch 24 — val/per: 0.0774
Epoch 24, global step 18175: 'val/per' reached 0.07744 (best 0.07744), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=24-val_per=0.0000.ckpt' as top 3
Epoch 25/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:44 • 0:00:00 7.72it/s v_num: jba0 train/loss_step: 0.000 val/loss: 18.211 val/per: 0.077 train/loss_epoch: 1.6822026-04-06 19:58:09,675 | INFO | Epoch 25 — val/per: 0.0780
Epoch 25, global step 18902: 'val/per' reached 0.07801 (best 0.07744), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=25-val_per=0.0000.ckpt' as top 3
Epoch 26/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:45 • 0:00:00 7.58it/s v_num: jba0 train/loss_step: 0.000 val/loss: 18.909 val/per: 0.078 train/loss_epoch: 1.1942026-04-06 20:13:02,438 | INFO | Epoch 26 — val/per: 0.0780
Epoch 26, global step 19629: 'val/per' reached 0.07798 (best 0.07744), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=26-val_per=0.0000.ckpt' as top 3
Epoch 27/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:47 • 0:00:00 7.57it/s v_num: jba0 train/loss_step: 0.001 val/loss: 19.797 val/per: 0.078 train/loss_epoch: 1.0032026-04-06 20:28:08,047 | INFO | Epoch 27 — val/per: 0.0771
Epoch 27, global step 20356: 'val/per' reached 0.07712 (best 0.07712), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=27-val_per=0.0000.ckpt' as top 3
Epoch 28/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:47 • 0:00:00 7.79it/s v_num: jba0 train/loss_step: 0.000 val/loss: 20.101 val/per: 0.077 train/loss_epoch: 0.7452026-04-06 20:43:03,163 | INFO | Epoch 28 — val/per: 0.0772
Epoch 28, global step 21083: 'val/per' reached 0.07723 (best 0.07712), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=28-val_per=0.0000.ckpt' as top 3
Epoch 29/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:43 • 0:00:00 7.70it/s v_num: jba0 train/loss_step: 0.000 val/loss: 21.122 val/per: 0.077 train/loss_epoch: 0.5482026-04-06 20:57:53,026 | INFO | Epoch 29 — val/per: 0.0773
Epoch 29, global step 21810: 'val/per' reached 0.07732 (best 0.07712), saving model to '/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=29-val_per=0.0000.ckpt' as top 3
Epoch 29/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:43 • 0:00:00 7.70it/s v_num: jba0 train/loss_step: 0.000 val/loss: 21.809 val/per: 0.077 train/loss_epoch: 0.435`Trainer.fit` stopped: `max_epochs=30` reached.
Epoch 29/29 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5809/5809 0:12:43 • 0:00:00 7.70it/s v_num: jba0 train/loss_step: 0.000 val/loss: 21.809 val/per: 0.077 train/loss_epoch: 0.435
2026-04-06 20:58:00,932 | INFO | Saving final model to /home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned...
2026-04-06 20:58:00,933 | INFO | Loading best checkpoint: /home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/wav2vec2-combined-finetuned/combined-epoch=27-val_per=0.0000.ckpt (val/per=0.0771)
Traceback (most recent call last):
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/finetune_wav2vec2_combined.py", line 747, in <module>
    main()
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/finetune_wav2vec2_combined.py", line 731, in main
    best_model = Wav2Vec2FineTuner.load_from_checkpoint(best_model_path)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/model_helpers.py", line 130, in wrapper
    return self.method(cls_type, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/pytorch_lightning/core/module.py", line 1797, in load_from_checkpoint
    loaded = _load_from_checkpoint(
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/pytorch_lightning/core/saving.py", line 65, in _load_from_checkpoint
    checkpoint = pl_load(checkpoint_path, map_location=map_location, weights_only=weights_only)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/lightning_fabric/utilities/cloud_io.py", line 73, in _load
    return torch.load(
           ^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 1521, in load
    return _load(
           ^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 2119, in _load
    result = unpickler.load()
             ^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/_weights_only_unpickler.py", line 532, in load
    self.append(self.persistent_load(pid))
                ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 2083, in persistent_load
    typed_storage = load_tensor(
                    ^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 2049, in load_tensor
    wrap_storage = restore_location(storage, location)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 1871, in restore_location
    result = default_restore_location(storage, location)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 698, in default_restore_location
    result = fn(storage, location)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 637, in _deserialize
    return obj.to(device=device)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/storage.py", line 291, in to
    return _to(self, device, non_blocking)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/_utils.py", line 101, in _to
    untyped_storage = torch.UntypedStorage(self.size(), device=device)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 7.51 GiB of which 40.75 MiB is free. Process 9500 has 148.91 MiB memory in use. Including non-PyTorch memory, this process has 7.01 GiB memory in use. Of the allocated memory 6.81 GiB is allocated by PyTorch, and 17.19 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
2026-04-06 20:58:03,234 | CRITICAL | Uncaught exception
Traceback (most recent call last):
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/finetune_wav2vec2_combined.py", line 747, in <module>
    main()
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/wav2vec2_cpp/scripts/finetune_wav2vec2_combined.py", line 731, in main
    best_model = Wav2Vec2FineTuner.load_from_checkpoint(best_model_path)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/model_helpers.py", line 130, in wrapper
    return self.method(cls_type, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/pytorch_lightning/core/module.py", line 1797, in load_from_checkpoint
    loaded = _load_from_checkpoint(
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/pytorch_lightning/core/saving.py", line 65, in _load_from_checkpoint
    checkpoint = pl_load(checkpoint_path, map_location=map_location, weights_only=weights_only)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/lightning_fabric/utilities/cloud_io.py", line 73, in _load
    return torch.load(
           ^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 1521, in load
    return _load(
           ^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 2119, in _load
    result = unpickler.load()
             ^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/_weights_only_unpickler.py", line 532, in load
    self.append(self.persistent_load(pid))
                ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 2083, in persistent_load
    typed_storage = load_tensor(
                    ^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 2049, in load_tensor
    wrap_storage = restore_location(storage, location)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 1871, in restore_location
    result = default_restore_location(storage, location)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 698, in default_restore_location
    result = fn(storage, location)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/serialization.py", line 637, in _deserialize
    return obj.to(device=device)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/storage.py", line 291, in to
    return _to(self, device, non_blocking)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ultraspeech-dev/Documents/ultraspeech-dev/wav2vec2-standalone-testing/.venv/lib/python3.12/site-packages/torch/_utils.py", line 101, in _to
    untyped_storage = torch.UntypedStorage(self.size(), device=device)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 7.51 GiB of which 40.75 MiB is free. Process 9500 has 148.91 MiB memory in use. Including non-PyTorch memory, this process has 7.01 GiB memory in use. Of the allocated memory 6.81 GiB is allocated by PyTorch, and 17.19 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
wandb:
wandb: 🚀 View run combined-uxtd-tal80-lightning at: https://wandb.ai/ntdinh05-university-at-buffalo/wav2vec2-combined-finetune/runs/ckdnjba0
wandb: Find logs at: wandb/run-20260406_132946-ckdnjba0/logs
