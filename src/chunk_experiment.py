import json
import csv
import glob
import os
import difflib
import numpy as np
import soundfile as sf
import onnxruntime as ort

SAMPLE_RATE = 16000
MIN_CHUNK_SAMPLES = 400  # wav2vec2 convolutional receptive field
CHUNK_LENGTHS_MS = [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000]

# DR1 folder contains all speakers and utterances
DR1_PATH = "../tests/DR1"
OUTPUT_CSV_PATH = "../output/experiment_results_raw.csv"
VOCAB_PATH = "../vocab/vocab.json"
MODEL_PATH = "../onnx_output/model.onnx"

SPECIAL_TOKENS = {"[PAD]", "<pad>", "<s>", "</s>", "<unk>"}
SPACE_TOKENS = {"|", "<|space|>"}

# IPA → ARPABET (TIMIT notation)
IPA_TO_ARPABET = {
    "\u0251": "aa",     # ɑ
    "\u00e6": "ae",     # æ
    "\u0259": "ax",     # ə
    "a\u028a": "aw",    # aʊ
    "a\u026a": "ay",    # aɪ
    "b": "b",
    "\u02a7": "ch",     # ʧ
    "d": "d",
    "\u00f0": "dh",     # ð
    "\u027e": "dx",     # ɾ
    "\u025b": "eh",     # ɛ
    "\u025d": "er",     # ɝ
    "e\u026a": "ey",    # eɪ
    "f": "f",
    "g": "g",
    "h": "hh",
    "\u026a": "ih",     # ɪ
    "i": "iy",
    "\u02a4": "jh",     # ʤ
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "\u014b": "ng",     # ŋ
    "o\u028a": "ow",    # oʊ
    "\u0254\u026a": "oy",  # ɔɪ
    "p": "p",
    "\u0279": "r",      # ɹ
    "s": "s",
    "\u0283": "sh",     # ʃ
    "t": "t",
    "\u03b8": "th",     # θ
    "\u028a": "uh",     # ʊ
    "u": "uw",
    "v": "v",
    "w": "w",
    "j": "y",
    "z": "z",
}

# Some TIMIT phonemes don't exist in the model's 42-token vocabulary.
# This maps them to the closest equivalent so we can do a fair comparison.
# Closures (dcl, gcl, kcl, pcl, tcl, bcl), silence (h#), and
# epenthetic silence (epi) are skipped entirely — the model never predicts them.
TIMIT_NORMALIZE = {
    "hv": "hh",    # voiced h → h
    "axr": "er",   # r-colored schwa → er
    "ao": "aa",    # open-o → open-a (closest in this vocab)
}
TIMIT_SKIP = {"h#", "dcl", "gcl", "kcl", "pcl", "tcl", "bcl", "epi", "q", "pau"}


def load_vocab(path):
    with open(path) as f:
        data = json.load(f)

    id_to_token = {}
    for token, token_id in data.items():
        id_to_token[token_id] = token

    return id_to_token


# Parses a .PHN file into a list of (start_sample, end_sample, phoneme) tuples.
# Each line in the file looks like: "9640 11240 sh"
def load_phn(path):
    entries = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start = int(parts[0])
                end = int(parts[1])
                phoneme = parts[2]
                entries.append((start, end, phoneme))
    return entries


# Returns the ground truth phonemes that overlap with a given sample range.
# A .PHN entry overlaps the chunk if: entry.start < chunk_end AND entry.end > chunk_start
# Skips silence/closure phonemes that the model can't predict,
# and normalizes TIMIT-specific phonemes to their model equivalents.
def get_ground_truth(phn_entries, chunk_start, chunk_end):
    phonemes = []
    for start, end, phoneme in phn_entries:
        if start < chunk_end and end > chunk_start:
            if phoneme in TIMIT_SKIP:
                continue
            phoneme = TIMIT_NORMALIZE.get(phoneme, phoneme)
            phonemes.append(phoneme)
    return phonemes


# Compares two phoneme lists using SequenceMatcher.
# Returns a ratio between 0.0 and 1.0 indicating how similar they are.
# SequenceMatcher finds the longest common subsequences, which handles
# insertions and deletions better than a simple position-by-position comparison.
def calculate_match(predicted, ground_truth):
    if len(ground_truth) == 0 and len(predicted) == 0:
        return 1.0
    if len(ground_truth) == 0 or len(predicted) == 0:
        return 0.0

    matcher = difflib.SequenceMatcher(None, predicted, ground_truth)
    return matcher.ratio()


def normalize(samples):
    mean = np.mean(samples)
    stdev = np.std(samples)
    epsilon = 1e-5
    return (samples - mean) / (stdev + epsilon)


def ctc_decode(ids, vocab):
    phonemes = []
    prev = -1

    for idx in ids:
        idx = int(idx)

        if idx == prev:
            continue
        prev = idx

        token = vocab.get(idx, "")

        if token in SPECIAL_TOKENS:
            continue

        if token in SPACE_TOKENS:
            phonemes.append("|")
        else:
            arpabet = IPA_TO_ARPABET.get(token, token)
            phonemes.append(arpabet)

    return phonemes


def run_inference(samples, session, vocab):
    if len(samples) < MIN_CHUNK_SAMPLES:
        return []

    normalized_audio = normalize(samples)
    input_tensor = normalized_audio.reshape(1, -1).astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    try:
        outputs = session.run([output_name], {input_name: input_tensor})
        logits = outputs[0]
        ids = np.argmax(logits[0], axis=-1)
        phonemes = ctc_decode(ids, vocab)
        return phonemes
    except Exception as e:
        print(f" [ONNX Error: {len(samples)} samples] {e}")
        return []


def chunk_audio(audio, chunk_size):
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunks.append(audio[i:i + chunk_size])
    return chunks


# Discovers all (wav_path, phn_path, speaker, utterance) pairs in the DR1 folder.
# Looks for every .WAV.wav file and checks that a matching .PHN file exists.
def discover_utterances(dr1_path):
    utterances = []
    wav_files = sorted(glob.glob(os.path.join(dr1_path, "*", "*.WAV.wav")))

    for wav_path in wav_files:
        # Derive the .PHN path from the .WAV.wav path
        # e.g. DR1/FAKS0/SA1.WAV.wav → DR1/FAKS0/SA1.PHN
        base = wav_path.replace(".WAV.wav", "")
        phn_path = base + ".PHN"

        if not os.path.exists(phn_path):
            print(f"Warning: no .PHN file for {wav_path}, skipping")
            continue

        speaker = os.path.basename(os.path.dirname(wav_path))
        utterance = os.path.basename(base)
        utterances.append((wav_path, phn_path, speaker, utterance))

    return utterances


def main():
    vocab = load_vocab(VOCAB_PATH)
    session = ort.InferenceSession(MODEL_PATH)

    # Find all utterances in the DR1 folder
    utterances = discover_utterances(DR1_PATH)
    print(f"Found {len(utterances)} utterances in DR1.")

    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)

        # --- Section 1: Experiment results header ---
        writer.writerow([
            "Speaker",
            "Utterance",
            "Chunk (ms)",
            "Start Sample",
            "End Sample",
            "Ground Truth",
            "Predicted",
            "Match %",
        ])

        # Track per-chunk-size averages across ALL utterances
        global_matches = {}
        for ms in CHUNK_LENGTHS_MS:
            global_matches[ms] = []

        for wav_path, phn_path, speaker, utterance in utterances:
            # print(f"\n{'#'*60}")
            # print(f"Speaker: {speaker}  Utterance: {utterance}")
            # print(f"{'#'*60}")

            audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if sr != SAMPLE_RATE:
                print(f"  Warning: sample rate is {sr}, expected {SAMPLE_RATE}")

            phn_entries = load_phn(phn_path)

            # --- Write ground truth for this utterance ---
            writer.writerow([])
            writer.writerow([f"=== GROUND TRUTH: {speaker}/{utterance} ==="])
            writer.writerow(["Start Sample", "End Sample", "Phoneme"])
            for start, end, phoneme in phn_entries:
                writer.writerow([start, end, phoneme])
            writer.writerow([])

            # --- Run experiment for each chunk size ---
            for ms in CHUNK_LENGTHS_MS:
                samples_per_chunk = SAMPLE_RATE * ms // 1000
                chunks = chunk_audio(audio, samples_per_chunk)

                chunk_matches = []

                for i, chunk in enumerate(chunks):
                    chunk_start = i * samples_per_chunk
                    chunk_end = chunk_start + len(chunk)

                    truth = get_ground_truth(phn_entries, chunk_start, chunk_end)

                    predicted_raw = run_inference(chunk, session, vocab)
                    predicted = [p for p in predicted_raw if p != "|"]

                    match = calculate_match(predicted, truth)
                    chunk_matches.append(match)

                    truth_str = " ".join(truth)
                    predicted_str = " ".join(predicted)
                    match_pct = f"{match * 100:.1f}%"

                    writer.writerow([
                        speaker,
                        utterance,
                        ms,
                        chunk_start,
                        chunk_end,
                        truth_str,
                        predicted_str,
                        match_pct,
                    ])

                # Per-utterance average for this chunk size
                if chunk_matches:
                    avg = sum(chunk_matches) / len(chunk_matches)
                    global_matches[ms].append(avg)
                    # print(f"  {ms}ms: {avg * 100:.1f}%")
                    writer.writerow([
                        speaker, utterance, ms, "", "", "", "AVERAGE",
                        f"{avg * 100:.1f}%",
                    ])

            writer.writerow([])

        # --- Final summary across all utterances ---
        writer.writerow([])
        writer.writerow(["=== GLOBAL SUMMARY (ALL DR1 UTTERANCES) ==="])
        writer.writerow(["Chunk (ms)", "Avg Match %", "Utterances"])

        print(f"\n{'='*60}")
        print("GLOBAL SUMMARY")
        print(f"{'='*60}")

        for ms in CHUNK_LENGTHS_MS:
            scores = global_matches[ms]
            if scores:
                avg = sum(scores) / len(scores)
                writer.writerow([ms, f"{avg * 100:.1f}%", len(scores)])
                # print(f"  {ms}ms: {avg * 100:.1f}% (across {len(scores)} utterances)")

    print(f"\nResults saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
