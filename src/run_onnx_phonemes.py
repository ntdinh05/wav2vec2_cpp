import json
import numpy as np
import soundfile as sf
import onnxruntime as ort
import os

# --- Constants ---
SAMPLE_RATE = 16000
# Wav2Vec2 generally has a stride of 20ms (320 samples at 16kHz)
# The output frame rate is 50Hz.
MODEL_STRIDE_MS = 20 
MODEL_STRIDE_SAMPLES = 320

DR1_PATH = "../tests/DR1"
VOCAB_PATH = "../vocab/vocab.json"
MODEL_PATH = "../onnx_output/model.onnx"
AUDIO_FILE = "../tests/DR1/FAKS0/SA1.WAV.wav"

SPECIAL_TOKENS = {"[PAD]", "<pad>", "<s>", "</s>", "<unk>"}
SPACE_TOKENS = {"|", "<|space|>"}

# IPA → ARPABET (TIMIT notation) - reused from chunk_experiment.py
IPA_TO_ARPABET = {
    "\u0251": "aa", "\u00e6": "ae", "\u0259": "ax", "a\u028a": "aw", "a\u026a": "ay",
    "b": "b", "\u02a7": "ch", "d": "d", "\u00f0": "dh", "\u027e": "dx",
    "\u025b": "eh", "\u025d": "er", "e\u026a": "ey", "f": "f", "g": "g",
    "h": "hh", "\u026a": "ih", "i": "iy", "\u02a4": "jh", "k": "k",
    "l": "l", "m": "m", "n": "n", "\u014b": "ng", "o\u028a": "ow",
    "\u0254\u026a": "oy", "p": "p", "\u0279": "r", "s": "s", "\u0283": "sh",
    "t": "t", "\u03b8": "th", "\u028a": "uh", "u": "uw", "v": "v",
    "w": "w", "j": "y", "z": "z",
}

def load_vocab(path):
    with open(path) as f:
        data = json.load(f)
    return {token_id: token for token, token_id in data.items()}

def normalize(samples):
    mean = np.mean(samples)
    stdev = np.std(samples)
    epsilon = 1e-5
    return (samples - mean) / (stdev + epsilon)

def ctc_decode_with_timestamps(ids, vocab, stride_ms=20):
    """
    Decodes CTC output and calculates start/end times.
    """
    phonemes = []
    prev_idx = -1
    
    # Group consecutive identical tokens
    current_token_group = []
    
    for frame_idx, token_id in enumerate(ids):
        token_id = int(token_id)
        
        # If character changes or we're at the end
        if token_id != prev_idx:
            if current_token_group:
                # Process the previous group
                token_val = vocab.get(prev_idx, "")
                if token_val not in SPECIAL_TOKENS:
                    # Determine time range based on frames
                    # Start of the segment
                    start_frame = current_token_group[0]
                    # End of the segment (inclusive)
                    end_frame = current_token_group[-1]
                    
                    # Convert to seconds
                    start_time = start_frame * stride_ms / 1000.0
                    end_time = (end_frame + 1) * stride_ms / 1000.0
                    
                    # Map to phoneme
                    display_phoneme = token_val
                    if token_val in SPACE_TOKENS:
                        display_phoneme = "|"
                    else:
                        display_phoneme = IPA_TO_ARPABET.get(token_val, token_val)
                    
                    phonemes.append({
                        "phoneme": display_phoneme,
                        "start": start_time,
                        "end": end_time,
                        "score": 1.0 # Greedy decoding implies max confidence
                    })
            
            # Start new group
            current_token_group = [frame_idx]
            prev_idx = token_id
        else:
            # Continue group
            current_token_group.append(frame_idx)
            
    # Process final group
    if current_token_group:
        token_val = vocab.get(prev_idx, "")
        if token_val not in SPECIAL_TOKENS:
            start_frame = current_token_group[0]
            end_frame = current_token_group[-1]
            start_time = start_frame * stride_ms / 1000.0
            end_time = (end_frame + 1) * stride_ms / 1000.0
            
            display_phoneme = token_val
            if token_val in SPACE_TOKENS:
                display_phoneme = "|"
            else:
                display_phoneme = IPA_TO_ARPABET.get(token_val, token_val)

            phonemes.append({
                "phoneme": display_phoneme,
                "start": start_time,
                "end": end_time
            })
            
    return phonemes

def main():
    # Setup paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(script_dir, VOCAB_PATH)
    model_path = os.path.join(script_dir, MODEL_PATH)
    audio_path = os.path.join(script_dir, AUDIO_FILE)

    print(f"Loading model from: {model_path}")
    vocab = load_vocab(vocab_path)
    session = ort.InferenceSession(model_path)
    
    print(f"Loading audio from: {audio_path}")
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    
    # Resample if needed (simple check)
    if sr != SAMPLE_RATE:
        print(f"Warning: Audio sample rate {sr} != {SAMPLE_RATE}")
        # In a real app, integrate a resampler here. 
        # For now, we assume the test files are correct.

    # Normalize
    normalized_audio = normalize(audio)
    input_tensor = normalized_audio.reshape(1, -1).astype(np.float32)

    # Run Inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    outputs = session.run([output_name], {input_name: input_tensor})
    logits = outputs[0] # Shape: [batch, frames, vocab_size]
    
    # Argmax greedy decoding
    ids = np.argmax(logits[0], axis=-1)
    
    # Decode with timestamps
    results = ctc_decode_with_timestamps(ids, vocab, stride_ms=MODEL_STRIDE_MS)

    # Print nicely
    print(f"\n{'='*50}")
    print(f"ONNX MODEL OUTPUT (Phonemes)")
    print(f"{'='*50}")
    print(f"{'Phoneme'.ljust(10)} {'Start'.ljust(10)} {'End'.ljust(10)}")
    print(f"{'-'*10} {'-'*10} {'-'*10}")
    
    for res in results:
        ph = res['phoneme']
        # Show space as distinct char if desired, or skip
        if ph == "|":
            print(f"{'<SPACE>'.ljust(10)} {str(res['start']).ljust(10)} {str(res['end']).ljust(10)}")
        else:
            print(f"{ph.ljust(10)} {f'{res['start']:.3f}'.ljust(10)} {f'{res['end']:.3f}'.ljust(10)}")
            
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
