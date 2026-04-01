import os
import re

import librosa
import torch
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ==========================================
# 1. IPA to ARPABET Converter Function
# ==========================================
def convert_ipa_to_arpabet(ipa_text):
    ipa_to_arpa = {
        # Vowels & Diphthongs
        "iː": "IY",
        "i": "IY",
        "ɪ": "IH",
        "eɪ": "EY",
        "e": "EY",
        "ɛ": "EH",
        "æ": "AE",
        "ɑː": "AA",
        "ɑ": "AA",
        "ɔ": "AO",
        "ɒ": "AO",
        "oʊ": "OW",
        "əʊ": "OW",
        "ʊ": "UH",
        "uː": "UW",
        "u": "UW",
        "ʌ": "AH",
        "ə": "AH",
        "ɚ": "ER",
        "ɝ": "ER",
        "ɜː": "ER",
        "ɜ": "ER",
        "aɪ": "AY",
        "aʊ": "AW",
        "ɔɪ": "OY",
        # Consonants
        "tʃ": "CH",
        "dʒ": "JH",
        "p": "P",
        "b": "B",
        "t": "T",
        "d": "D",
        "k": "K",
        "ɡ": "G",
        "g": "G",
        "m": "M",
        "n": "N",
        "ŋ": "NG",
        "f": "F",
        "v": "V",
        "θ": "TH",
        "ð": "DH",
        "s": "S",
        "z": "Z",
        "ʃ": "SH",
        "ʒ": "ZH",
        "h": "HH",
        "l": "L",
        "r": "R",
        "ɹ": "R",
        "j": "Y",
        "w": "W",
        # Modifiers and Wav2Vec Boundaries
        "ˈ": "",
        "ˌ": "",
        "ː": "",  # Strip stress and length marks
        "|": " ",
        " ": " ",  # Convert Wav2Vec word boundaries to spaces
    }

    # Sort by length descending so multi-char symbols (e.g. tʃ) match before single chars
    sorted_ipa = sorted(ipa_to_arpa.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(key) for key in sorted_ipa))

    def replace_match(match):
        val = ipa_to_arpa[match.group(0)]
        if val.isalpha():
            return val + " "
        return val

    raw_arpa = pattern.sub(replace_match, ipa_text)
    clean_arpa = re.sub(r"[^A-Z ]", "", raw_arpa)
    return re.sub(r"\s+", " ", clean_arpa).strip()


# ==========================================
# 2. Setup Model, Processor, and Decoder
# ==========================================
model_id = "facebook/wav2vec2-lv-60-espeak-cv-ft"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab = sorted((value, key) for key, value in vocab_dict.items())
labels = [x[1] for x in sorted_vocab]

decoder = build_ctcdecoder(
    labels,
    kenlm_model_path=None,
    alpha=0.5,
    beta=1.0,
)


# ==========================================
# 3. Transcription Function
# ==========================================
def transcribe(audio_path):
    # Load and preprocess audio (Must be 16kHz)
    speech, _ = librosa.load(audio_path, sr=16000)
    input_values = processor(
        speech, return_tensors="pt", sampling_rate=16000
    ).input_values

    # Get Logits
    with torch.no_grad():
        logits = model(input_values).logits[0]

    # Decode using pyctcdecode
    beam_search_output = decoder.decode(logits.numpy())
    return beam_search_output

# ==========================================
# 4. TIMIT Word-Phoneme Alignment
# ==========================================
_TIMIT_CLOSURES = {"dcl", "tcl", "bcl", "kcl", "gcl", "pcl"}
_TIMIT_SILENCE  = {"h#", "epi", "q"}
_TIMIT_TO_ARPABET = {
    "hv": "HH",
    "dcl": "D",  "tcl": "T",  "bcl": "B",  "kcl": "K",  "gcl": "G",  "pcl": "P",
    "dx":  "DX",
    "axr": "ER", "ax":  "AH", "ax-h": "AH", "ix": "IH",
    "em":  "M",  "en":  "N",  "eng": "NG",  "el": "L",
    "sh":  "SH", "ch":  "CH", "jh":  "JH", "th": "TH", "dh": "DH", "zh": "ZH",
    "iy":  "IY", "ih":  "IH", "eh":  "EH", "ae": "AE", "aa": "AA",
    "aw":  "AW", "ay":  "AY", "ah":  "AH", "ao": "AO", "oy": "OY",
    "ow":  "OW", "uh":  "UH", "uw":  "UW", "ux": "UW", "er": "ER", "ey": "EY",
    "p":   "P",  "b":   "B",  "t":   "T",  "d":  "D",  "k":  "K",  "g":  "G",
    "m":   "M",  "n":   "N",  "ng":  "NG",
    "f":   "F",  "v":   "V",  "s":   "S",  "z":  "Z",  "hh": "HH", "h":  "HH",
    "l":   "L",  "r":   "R",  "y":   "Y",  "w":  "W",
}


def align_phonemes_to_words(wrd_path, phn_path):
    """Read TIMIT .WRD and .PHN files; return list of (word, [ARPABET, ...]).

    Stop closures (dcl/kcl/…) and silence labels are dropped. Each phoneme
    is assigned to the word whose sample range contains the phoneme midpoint.
    """
    words, phonemes = [], []
    with open(wrd_path) as f:
        for line in f:
            s, e, w = line.split()
            words.append((int(s), int(e), w))
    with open(phn_path) as f:
        for line in f:
            s, e, p = line.split()
            if p not in _TIMIT_CLOSURES and p not in _TIMIT_SILENCE:
                phonemes.append((int(s), int(e), p))

    result = []
    for ws, we, word in words:
        word_phns = []
        for ps, pe, phn in phonemes:
            if ws <= (ps + pe) / 2 < we:
                arpa = _TIMIT_TO_ARPABET.get(phn)
                if arpa:
                    word_phns.append(arpa)
        result.append((word, word_phns))
    return result


# ==========================================
# 5. Execution
# ==========================================
if __name__ == "__main__":
    audio_file = "../tests/DR1/FAKS0/SA1.WAV.wav"

    # Step 1: Get the IPA string from the model
    ipa_result = transcribe(audio_file)
    print(f"Phonetic Transcription (IPA): {ipa_result}")

    # Step 2: Convert to ARPABET
    arpabet_result = convert_ipa_to_arpabet(ipa_result)
    print(f"ARPABET Transcription:        {arpabet_result}")

    # Step 3: Load and display ground-truth word-aligned phonemes
    base = os.path.splitext(os.path.splitext(audio_file)[0])[0]  # strip .WAV.wav
    wrd_path = base + ".WRD"
    phn_path = base + ".PHN"
    if os.path.exists(wrd_path) and os.path.exists(phn_path):
        alignment = align_phonemes_to_words(wrd_path, phn_path)
        print("\nGround truth word alignment:")
        for word, phns in alignment:
            print(f"  {word:<12} {' '.join(phns)}")
