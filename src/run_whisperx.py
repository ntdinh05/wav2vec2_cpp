import torch

# Monkeypatch torch.load to default weights_only=False for pyside/pyannote compatibility
original_load = torch.load
def strict_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = strict_load

import whisperx
import os
from g2p_en import G2p

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Robustly find the audio file relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up from src/ to wave2vec2_cpp/ then to tests/
audio_file = os.path.join(script_dir, "..", "tests", "DR1", "FAKS0", "SA1.WAV.wav")

if not os.path.exists(audio_file):
    print(f"Error: Audio file not found at {audio_file}")
    exit(1)

# Initialize G2P
g2p = G2p()

# 1. Transcribe with Whisper (gets the words)
compute_type = "float16" if device == "cuda" else "int8"
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
result = model.transcribe(audio_file)

# 2. Align (matches phonemes to words)
# Note: The default English alignment model aligns characters (graphemes), not phonemes.
# To align true phonemes, you would need a G2P step and a phoneme-based alignment model.
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_file, device, return_char_alignments=True)

# 3. Clean Output
print(f"\n{'='*60}")
print(f"TRANSCRIPTION & PHONEMES (g2p_en)")
print(f"Note: Phonemes are generated via G2P from the transcribed words.")
print(f"{'='*60}")

for segment in result_aligned["segments"]:
    start = segment.get("start", 0)
    end = segment.get("end", 0)
    print(f"\nSegment [{start:.3f} - {end:.3f}]: {segment.get('text', '').strip()}")
    
    if "words" in segment:
        print(f"  {'Phonemes'.ljust(25)} {'Word'.ljust(15)} {'Time Range'}")
        print(f"  {'-'*25} {'-'*15} {'-'*20}")
        
        for word_info in segment["words"]:
            w_text = word_info.get("word", "")
            w_start = word_info.get("start", 0)
            w_end = word_info.get("end", 0)
            
            # Convert word to phonemes using G2P
            # p.isdigit() check removes stress numbers (IY1 -> IY) for cleaner output
            phonemes_raw = g2p(w_text)
            phonemes = [p for p in phonemes_raw if p.strip() and p not in ["'", ",", "."]]
            # Strip stress digits if present (e.g. 'AA1' -> 'AA')
            phonemes_clean = ["".join(filter(str.isalpha, p)) for p in phonemes]
            phonemes_str = " ".join(phonemes_clean)
            
            print(f"  {phonemes_str.ljust(25)} {w_text.ljust(15)} [{w_start:.3f} - {w_end:.3f}]")
            
    print("-" * 60)