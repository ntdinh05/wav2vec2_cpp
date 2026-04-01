#!/usr/bin/env python3
"""
Phoneme Recognition Benchmark on UltraSuite core-uxtd dataset.

Methodology
-----------
Ground truth
  g2p_en (CMUdict-backed) converts each utterance's text transcript to ARPABET.
  Stress markers are stripped (AH0 → AH) and the sequence is lowercased so it
  matches the TIMIT token set. Word-boundary spaces are kept as a separator but
  not counted as phoneme tokens.

Models tested (in the order listed below)
  1. vitouphy/wav2vec2-xls-r-300m-timit-phoneme   – XLS-R 300M, TIMIT ARPABET
  2. facebook/wav2vec2-lv-60-espeak-cv-ft           – LibriLight, espeak IPA
  3. bookbot/wav2vec2-ljspeech-gruut                – 94M, Gruut IPA
  4. facebook/wav2vec2-xlsr-53-espeak-cv-ft         – XLSR-53, espeak IPA multilingual
  5. allosaurus (pip install allosaurus)            – PHOIBLE IPA, 2000+ languages

All model outputs are normalised to ARPABET via the IPA→ARPABET table in this
file so every model is scored against the same reference space.

Metric  PER (Phoneme Error Rate)
  PER = (Substitutions + Deletions + Insertions) / N_reference
  Computed with jiwer.wer() treating each phoneme as a "word".
  Lower is better; 0 = perfect match.

Audio preprocessing
  All files in core-uxtd are at 22 050 Hz; wav2vec2 models require 16 000 Hz.
  Resampling is done with torchaudio.functional.resample() in float32.

Outputs
  wave2vec2_cpp/output/phoneme_benchmark_per_utterance.csv  – one row per (model, utterance)
  wave2vec2_cpp/output/phoneme_benchmark_summary.csv        – per-model aggregate stats
"""

import csv
import os
import re
import time
import traceback

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from g2p_en import G2p
from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2PhonemeCTCTokenizer,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_INPUT  = os.path.join(os.path.dirname(__file__), "../tests/utterances_by_length.csv")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "../output")
OUT_UTTE   = os.path.join(OUT_DIR, "phoneme_benchmark_per_utterance.csv")
OUT_SUMM   = os.path.join(OUT_DIR, "phoneme_benchmark_summary.csv")

TARGET_SR  = 16_000
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# ARPABET stress-strip  (AH0 / AH1 / AH2  →  AH)
# ---------------------------------------------------------------------------
_STRESS_RE = re.compile(r"\d$")

def strip_stress(tok: str) -> str:
    return _STRESS_RE.sub("", tok)

# ---------------------------------------------------------------------------
# IPA → ARPABET  (covers espeak-ng and Gruut IPA, and allosaurus PHOIBLE)
# ---------------------------------------------------------------------------
IPA_TO_ARPABET: dict[str, str] = {
    # Vowels
    "ɑ":  "aa", "ɑː": "aa",
    "æ":  "ae",
    "ə":  "ax", "ɐ":  "ax",
    "aʊ": "aw",
    "aɪ": "ay",
    "ɛ":  "eh", "e":  "eh",
    "ɝ":  "er", "ɚ":  "er", "ɜ": "er",
    "eɪ": "ey",
    "ɪ":  "ih",
    "i":  "iy", "iː": "iy",
    "oʊ": "ow", "o":  "ow",
    "ɔɪ": "oy",
    "ʊ":  "uh",
    "u":  "uw", "uː": "uw",
    "ʌ":  "ah",
    "ɔ":  "ao",
    # Consonants
    "b":  "b",
    "tʃ": "ch",
    "d":  "d",
    "ð":  "dh",
    "ɾ":  "dx",
    "f":  "f",
    "ɡ":  "g", "g":  "g",
    "h":  "hh",
    "dʒ": "jh",
    "k":  "k",
    "l":  "l",
    "m":  "m",
    "n":  "n",
    "ŋ":  "ng",
    "p":  "p",
    "ɹ":  "r", "r":  "r",
    "s":  "s",
    "ʃ":  "sh",
    "t":  "t",
    "θ":  "th",
    "v":  "v",
    "w":  "w",
    "j":  "y",
    "z":  "z",
    "ʒ":  "zh",
    # Additional allosaurus / PHOIBLE symbols
    "ʔ":  "t",   # glottal stop → closest plosive
    "x":  "k",   # velar fricative
    "ç":  "sh",  # palatal fricative
    "β":  "v",
    "χ":  "k",
    "ʁ":  "r",
    "ɬ":  "l",
    "ts": "s",
    "pf": "f",
}

# Normalise TIMIT-specific phonemes to the common set
TIMIT_NORM: dict[str, str] = {
    "hv":  "hh",
    "axr": "er",
    "ao":  "aa",
    "ax":  "ah",
    "ix":  "ih",
    "el":  "l",
    "em":  "m",
    "en":  "n",
    "nx":  "n",
    "eng": "ng",
}
TIMIT_SKIP = {"h#", "dcl", "gcl", "kcl", "pcl", "tcl", "bcl", "epi", "q", "pau", "sil"}

def ipa_to_arpabet(ipa_str: str) -> list[str]:
    """
    Convert an IPA string to a list of ARPABET tokens.
    Tries greedy 2-char matching first, then single-char.
    Silently drops characters not in the table (e.g. length marks, nasalisation).
    """
    tokens = []
    i = 0
    s = ipa_str.strip()
    while i < len(s):
        two = s[i:i+2]
        one = s[i:i+1]
        if two in IPA_TO_ARPABET:
            tokens.append(IPA_TO_ARPABET[two])
            i += 2
        elif one in IPA_TO_ARPABET:
            tokens.append(IPA_TO_ARPABET[one])
            i += 1
        else:
            i += 1  # skip unknown symbol
    return tokens

# ---------------------------------------------------------------------------
# G2P reference
# ---------------------------------------------------------------------------
_g2p = G2p()

def text_to_arpabet(text: str) -> list[str]:
    """Convert utterance text → cleaned ARPABET list via g2p_en + CMUdict."""
    raw = _g2p(text.lower())
    result = []
    for tok in raw:
        if tok == " ":
            continue  # word boundary – don't count
        clean = strip_stress(tok).lower()
        if not clean.isalpha():
            continue
        clean = TIMIT_NORM.get(clean, clean)
        if clean in TIMIT_SKIP:
            continue
        result.append(clean)
    return result

# ---------------------------------------------------------------------------
# PER calculation
# ---------------------------------------------------------------------------
def compute_per(ref: list[str], hyp: list[str]) -> float:
    """Phoneme Error Rate using jiwer (phoneme-as-word trick)."""
    if not ref:
        return float("nan")
    ref_str = " ".join(ref)
    hyp_str = " ".join(hyp) if hyp else ""
    return jiwer.wer(ref_str, hyp_str)

# ---------------------------------------------------------------------------
# Audio loading + resampling
# ---------------------------------------------------------------------------
def load_audio(wav_path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Load wav, resample to target_sr, return 1-D float32 tensor."""
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
    return waveform.squeeze(0)  # (T,)

# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------
SPECIAL_HF = {"[PAD]", "<pad>", "<s>", "</s>", "<unk>", "|", "[UNK]"}

def _hf_ctc_decode_timit(logits: np.ndarray, processor) -> list[str]:
    """CTC greedy decode for TIMIT/ARPABET model (vitouphy)."""
    ids = np.argmax(logits[0], axis=-1)
    tokens = []
    prev = -1
    for idx in ids:
        if int(idx) == prev:
            continue
        prev = int(idx)
        tok = processor.decode([int(idx)]).strip()
        if tok in SPECIAL_HF or tok == "":
            continue
        tok = TIMIT_NORM.get(tok, tok)
        if tok in TIMIT_SKIP:
            continue
        tokens.append(tok)
    return tokens

def _hf_ctc_decode_ipa(logits: np.ndarray, processor) -> list[str]:
    """CTC greedy decode for IPA models, then convert to ARPABET."""
    ids = np.argmax(logits[0], axis=-1)
    # collapse repeats, skip PAD
    prev = -1
    ipa_chars = []
    for idx in ids:
        if int(idx) == prev:
            continue
        prev = int(idx)
        tok = processor.decode([int(idx)]).strip()
        if tok in SPECIAL_HF or tok == "":
            continue
        ipa_chars.append(tok)
    # The IPA tokenizer emits full IPA symbols as individual tokens
    tokens = []
    for ch in ipa_chars:
        arpa = ipa_to_arpabet(ch)
        tokens.extend(arpa)
    return tokens


class TimitModel:
    # Despite the "timit" name, this model's vocab is IPA (not ARPABET strings).
    name = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
    short = "vitouphy-timit"

    def load(self):
        self.processor = Wav2Vec2Processor.from_pretrained(self.name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.name).eval().to(DEVICE)

    def predict(self, waveform: torch.Tensor) -> list[str]:
        inputs = self.processor(waveform.numpy(), sampling_rate=TARGET_SR,
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits.cpu().numpy()
        # Vocab is IPA; use the IPA decoder path
        return _hf_ctc_decode_ipa(logits, self.processor.tokenizer)


class LibriLightIpaModel:
    name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    short = "facebook-lv60-espeak"

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.name).eval().to(DEVICE)

    def predict(self, waveform: torch.Tensor) -> list[str]:
        inputs = self.processor(waveform.numpy(), sampling_rate=TARGET_SR,
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits.cpu().numpy()
        return _hf_ctc_decode_ipa(logits, self.processor.tokenizer)


class BookbotGruutModel:
    name = "bookbot/wav2vec2-ljspeech-gruut"
    short = "bookbot-ljspeech-gruut"

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.name).eval().to(DEVICE)

    def predict(self, waveform: torch.Tensor) -> list[str]:
        inputs = self.processor(waveform.numpy(), sampling_rate=TARGET_SR,
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits.cpu().numpy()
        return _hf_ctc_decode_ipa(logits, self.processor.tokenizer)


class Xlsr53EspeakModel:
    name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    short = "facebook-xlsr53-espeak"

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.name).eval().to(DEVICE)

    def predict(self, waveform: torch.Tensor) -> list[str]:
        inputs = self.processor(waveform.numpy(), sampling_rate=TARGET_SR,
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits.cpu().numpy()
        return _hf_ctc_decode_ipa(logits, self.processor.tokenizer)


class AllosaurusModel:
    name = "allosaurus"
    short = "allosaurus"

    def load(self):
        from allosaurus.app import read_recognizer
        self.recognizer = read_recognizer()
        # allosaurus needs a temp file; we import tempfile lazily
        import tempfile
        self._tmpdir = tempfile.mkdtemp()

    def predict(self, waveform: torch.Tensor) -> list[str]:
        import tempfile, soundfile as sf2
        # Write 16kHz wav to a temp file, allosaurus reads from disk
        tmp = os.path.join(self._tmpdir, "tmp.wav")
        sf2.write(tmp, waveform.numpy(), TARGET_SR)
        raw = self.recognizer.recognize(tmp, lang_id="eng")
        # raw is an IPA string like "h ɛ l oʊ" (space-separated phones)
        tokens = []
        for ph in raw.split():
            tokens.extend(ipa_to_arpabet(ph))
        return tokens


ALL_MODELS = [
    TimitModel(),
    LibriLightIpaModel(),
    BookbotGruutModel(),
    Xlsr53EspeakModel(),
    AllosaurusModel(),
]

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_INPUT)
    print(f"Loaded {len(df)} utterances from CSV.")
    print(f"Running on device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

    utte_rows = []   # per-utterance detail
    for model_obj in ALL_MODELS:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_obj.name}")
        try:
            t0 = time.time()
            model_obj.load()
            load_time = time.time() - t0
            print(f"  Loaded in {load_time:.1f}s")
        except Exception as e:
            print(f"  FAILED to load: {e}")
            traceback.print_exc()
            continue

        for i, row in df.iterrows():
            wav_path = row["filepath"].replace(".txt", ".wav")
            utterance_text = row["utterance"]
            speaker = row["speaker"]
            filename = row["filename"]

            if not os.path.exists(wav_path):
                print(f"  [SKIP] {wav_path} not found")
                utte_rows.append({
                    "model": model_obj.short,
                    "speaker": speaker,
                    "filename": filename,
                    "utterance": utterance_text,
                    "ref_phonemes": "",
                    "hyp_phonemes": "",
                    "n_ref": 0,
                    "per": float("nan"),
                    "inference_time_s": float("nan"),
                    "error": "wav not found",
                })
                continue

            # Ground truth phonemes
            ref = text_to_arpabet(utterance_text)

            try:
                waveform = load_audio(wav_path)
                t_start = time.time()
                hyp = model_obj.predict(waveform)
                inf_time = time.time() - t_start
                per = compute_per(ref, hyp)
                err = ""
            except Exception as e:
                hyp = []
                inf_time = float("nan")
                per = float("nan")
                err = str(e)
                print(f"  [ERR] {speaker}/{filename}: {e}")

            utte_rows.append({
                "model": model_obj.short,
                "speaker": speaker,
                "filename": filename,
                "utterance": utterance_text,
                "ref_phonemes": " ".join(ref),
                "hyp_phonemes": " ".join(hyp),
                "n_ref": len(ref),
                "per": round(per, 4) if not np.isnan(per) else float("nan"),
                "inference_time_s": round(inf_time, 4) if not np.isnan(inf_time) else float("nan"),
                "error": err,
            })

            if (i + 1) % 50 == 0:
                print(f"  [{model_obj.short}] {i+1}/{len(df)} done")

        # free GPU memory / large objects
        try:
            del model_obj.model
        except AttributeError:
            pass
        try:
            del model_obj.recognizer
        except AttributeError:
            pass
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # -----------------------------------------------------------------
    # Write per-utterance CSV
    # -----------------------------------------------------------------
    detail_df = pd.DataFrame(utte_rows)
    detail_df.to_csv(OUT_UTTE, index=False)
    print(f"\nPer-utterance results → {OUT_UTTE}")

    # -----------------------------------------------------------------
    # Compute summary per model
    # -----------------------------------------------------------------
    summary_rows = []
    for model_short in detail_df["model"].unique():
        sub = detail_df[detail_df["model"] == model_short].copy()
        valid = sub.dropna(subset=["per"])
        n_ok  = len(valid)
        n_err = sub["error"].astype(bool).sum()
        avg_per  = valid["per"].mean()
        med_per  = valid["per"].median()
        avg_time = valid["inference_time_s"].mean()
        summary_rows.append({
            "model": model_short,
            "n_utterances": len(sub),
            "n_scored": n_ok,
            "n_errors": n_err,
            "mean_per": round(avg_per, 4),
            "median_per": round(med_per, 4),
            "mean_inference_time_s": round(avg_time, 4),
        })

    summ_df = pd.DataFrame(summary_rows).sort_values("mean_per")
    summ_df.to_csv(OUT_SUMM, index=False)
    print(f"Summary results         → {OUT_SUMM}")
    print("\n" + summ_df.to_string(index=False))


if __name__ == "__main__":
    main()
