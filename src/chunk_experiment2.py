import csv
import glob
import json
import os

import numpy as np
import onnxruntime as ort
import soundfile as sf

SAMPLE_RATE = 16000
CHUNK_LENGTHS_MS = [
    500,
    750,
    1000,
    1250,
    1500,
    1750,
    2000,
    2250,
    2500,
    2750,
    3000,
    3250,
    3500,
    3750,
    4000,
    4250,
    4500,
    4750,
    5000,
]

DR1_PATH = "../tests/DR1"
OUTPUT_CSV_PATH = "../output/experiment_results_per.csv"
VOCAB_PATH = "../vocab/vocab.json"
MODEL_PATH = "../onnx_output/model.onnx"

SPECIAL_TOKENS = {"[PAD]", "<pad>", "<s>", "</s>", "<unk>", "[UNK]"}
SPACE_TOKENS = {"|", "<|space|>", " "}

IPA_TO_ARPABET = {
    "\u0251": "aa",  # ɑ
    "\u00e6": "ae",  # æ
    "\u0259": "ax",  # ə
    "a\u028a": "aw",  # aʊ
    "a\u026a": "ay",  # aɪ
    "b": "b",
    "\u02a7": "ch",  # ʧ
    "d": "d",
    "\u00f0": "dh",  # ð
    "\u027e": "dx",  # ɾ
    "\u025b": "eh",  # ɛ
    "\u025d": "er",  # ɝ
    "e\u026a": "ey",  # eɪ
    "f": "f",
    "g": "g",
    "h": "hh",
    "\u026a": "ih",  # ɪ
    "i": "iy",
    "\u02a4": "jh",  # ʤ
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "\u014b": "ng",  # ŋ
    "o\u028a": "ow",  # oʊ
    "\u0254\u026a": "oy",  # ɔɪ
    "p": "p",
    "\u0279": "r",  # ɹ
    "s": "s",
    "\u0283": "sh",  # ʃ
    "t": "t",
    "\u03b8": "th",  # θ
    "\u028a": "uh",  # ʊ
    "u": "uw",
    "v": "v",
    "w": "w",
    "j": "y",
    "z": "z",
}

TIMIT_NORMALIZE = {
    "hv": "hh",
    "axr": "er",
    "ao": "aa",
}
TIMIT_SKIP = {"h#", "dcl", "gcl", "kcl", "pcl", "tcl", "bcl", "epi", "q", "pau"}


# ---------------------------------------------------------------------------
# Edit distance with backtrace — returns (S, I, D, alignment)
# ---------------------------------------------------------------------------
def edit_distance(hypothesis, reference):
    """
    Compute the Levenshtein edit distance between two phoneme lists.
    Levenshtein edit distance algorithm

    Returns:
        substitutions (S): hypothesis phoneme != reference phoneme
        insertions   (I): extra phonemes in hypothesis not in reference
        deletions    (D): reference phonemes missing from hypothesis
        alignment: list of (op, hyp_token, ref_token) tuples
            op is one of 'C' (correct), 'S' (substitution),
            'I' (insertion), 'D' (deletion)
    """
    n = len(reference)
    m = len(hypothesis)

    # DP table: dp[i][j] = min edits to align reference[:i] with hypothesis[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Base cases
    for i in range(1, n + 1):
        dp[i][0] = i  # all deletions
    for j in range(1, m + 1):
        dp[0][j] = j  # all insertions

    # Fill
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost_sub = 0
            else:
                cost_sub = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion  (ref phoneme skipped)
                dp[i][j - 1] + 1,  # insertion (extra hyp phoneme)
                dp[i - 1][j - 1] + cost_sub,  # match or substitution
            )

    # Backtrace to recover S, I, D counts and alignment
    i, j = n, m
    S, I, D = 0, 0, 0
    alignment = []

    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            alignment.append(("C", hypothesis[j - 1], reference[i - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append(("S", hypothesis[j - 1], reference[i - 1]))
            S += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            alignment.append(("I", hypothesis[j - 1], "***"))
            I += 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append(("D", "***", reference[i - 1]))
            D += 1
            i -= 1
        else:
            break  # should not happen

    alignment.reverse()
    return S, I, D, alignment


def compute_per(hypothesis, reference):
    """
    Compute Phoneme Error Rate and Phoneme Accuracy.

    Returns dict with keys: S, I, D, N, PER, accuracy, alignment
    """
    N = len(reference)
    if N == 0:
        if len(hypothesis) == 0:
            return {
                "S": 0,
                "I": 0,
                "D": 0,
                "N": 0,
                "PER": 0.0,
                "accuracy": 1.0,
                "alignment": [],
            }
        else:
            return {
                "S": 0,
                "I": len(hypothesis),
                "D": 0,
                "N": 0,
                "PER": float("inf"),
                "accuracy": 0.0,
                "alignment": [],
            }

    S, I, D, alignment = edit_distance(hypothesis, reference)

    per = (S + I + D) / N
    accuracy = max(0.0, (N - S - I - D) / N)

    return {
        "S": S,
        "I": I,
        "D": D,
        "N": N,
        "PER": per,
        "accuracy": accuracy,
        "alignment": alignment,
    }


# ---------------------------------------------------------------------------
# Helpers (same as chunk_experiment.py)
# ---------------------------------------------------------------------------
def load_vocab(path):
    with open(path) as f:
        data = json.load(f)
    return {token_id: token for token, token_id in data.items()}


def load_phn(path):
    entries = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                entries.append((int(parts[0]), int(parts[1]), parts[2]))
    return entries


def get_ground_truth(phn_entries, chunk_start, chunk_end):
    phonemes = []
    for start, end, phoneme in phn_entries:
        if start < chunk_end and end > chunk_start:
            if phoneme in TIMIT_SKIP:
                continue
            phoneme = TIMIT_NORMALIZE.get(phoneme, phoneme)
            phonemes.append(phoneme)
    return phonemes


def normalize(samples):
    mean = np.mean(samples)
    stdev = np.std(samples)
    return (samples - mean) / (stdev + 1e-5)


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
        return ctc_decode(ids, vocab)
    except Exception as e:
        print(f" [ONNX Error: {len(samples)} samples] {e}")
        return []


def chunk_audio(audio, chunk_size):
    return [audio[i : i + chunk_size] for i in range(0, len(audio), chunk_size)]


def discover_utterances(dr1_path):
    utterances = []
    wav_files = sorted(glob.glob(os.path.join(dr1_path, "*", "*.WAV.wav")))
    for wav_path in wav_files:
        base = wav_path.replace(".WAV.wav", "")
        phn_path = base + ".PHN"
        if not os.path.exists(phn_path):
            print(f"Warning: no .PHN file for {wav_path}, skipping")
            continue
        speaker = os.path.basename(os.path.dirname(wav_path))
        utterance = os.path.basename(base)
        utterances.append((wav_path, phn_path, speaker, utterance))
    return utterances


# ---------------------------------------------------------------------------
# Main experiment loop — same structure, but using PER instead of SequenceMatcher
# ---------------------------------------------------------------------------
def main():
    vocab = load_vocab(VOCAB_PATH)
    session = ort.InferenceSession(MODEL_PATH)

    utterances = discover_utterances(DR1_PATH)
    print(f"Found {len(utterances)} utterances in DR1.")

    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "Speaker",
                "Utterance",
                "Chunk (ms)",
                "Start Sample",
                "End Sample",
                "N (ref)",
                "S (sub)",
                "I (ins)",
                "D (del)",
                "PER %",
                "Accuracy %",
                "Ground Truth",
                "Predicted",
                "Alignment",
            ]
        )

        # Track global stats per chunk size
        global_stats = {ms: {"S": 0, "I": 0, "D": 0, "N": 0} for ms in CHUNK_LENGTHS_MS}

        for wav_path, phn_path, speaker, utterance in utterances:
            audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if sr != SAMPLE_RATE:
                print(f"  Warning: sample rate is {sr}, expected {SAMPLE_RATE}")

            phn_entries = load_phn(phn_path)

            # Write ground truth header for this utterance
            writer.writerow([])
            writer.writerow([f"=== GROUND TRUTH: {speaker}/{utterance} ==="])
            writer.writerow(["Start Sample", "End Sample", "Phoneme"])
            for start, end, phoneme in phn_entries:
                writer.writerow([start, end, phoneme])
            writer.writerow([])

            for ms in CHUNK_LENGTHS_MS:
                samples_per_chunk = SAMPLE_RATE * ms // 1000
                chunks = chunk_audio(audio, samples_per_chunk)

                utt_S, utt_I, utt_D, utt_N = 0, 0, 0, 0

                for i, chunk in enumerate(chunks):
                    chunk_start = i * samples_per_chunk
                    chunk_end = chunk_start + len(chunk)

                    truth = get_ground_truth(phn_entries, chunk_start, chunk_end)
                    predicted_raw = run_inference(chunk, session, vocab)
                    predicted = [p for p in predicted_raw if p != "|"]

                    result = compute_per(predicted, truth)

                    utt_S += result["S"]
                    utt_I += result["I"]
                    utt_D += result["D"]
                    utt_N += result["N"]

                    # Format alignment for CSV: C:eh S:ih→eh I:z D:*→t
                    align_str = " ".join(
                        f"{op}:{hyp}→{ref}" if op != "C" else f"C:{ref}"
                        for op, hyp, ref in result["alignment"]
                    )

                    per_pct = f"{result['PER'] * 100:.1f}%"
                    acc_pct = f"{result['accuracy'] * 100:.1f}%"

                    writer.writerow(
                        [
                            speaker,
                            utterance,
                            ms,
                            chunk_start,
                            chunk_end,
                            result["N"],
                            result["S"],
                            result["I"],
                            result["D"],
                            per_pct,
                            acc_pct,
                            " ".join(truth),
                            " ".join(predicted),
                            align_str,
                        ]
                    )

                # Accumulate into global totals
                global_stats[ms]["S"] += utt_S
                global_stats[ms]["I"] += utt_I
                global_stats[ms]["D"] += utt_D
                global_stats[ms]["N"] += utt_N

                # Per-utterance summary row
                if utt_N > 0:
                    utt_per = (utt_S + utt_I + utt_D) / utt_N
                    utt_acc = max(0.0, (utt_N - utt_S - utt_I - utt_D) / utt_N)
                else:
                    utt_per = 0.0
                    utt_acc = 1.0

                writer.writerow(
                    [
                        speaker,
                        utterance,
                        ms,
                        "",
                        "",
                        utt_N,
                        utt_S,
                        utt_I,
                        utt_D,
                        f"{utt_per * 100:.1f}%",
                        f"{utt_acc * 100:.1f}%",
                        "",
                        "UTT AVERAGE",
                        "",
                    ]
                )

            writer.writerow([])

        # --- Global summary ---
        writer.writerow([])
        writer.writerow(["=== GLOBAL SUMMARY (ALL DR1 UTTERANCES) ==="])
        writer.writerow(
            [
                "Chunk (ms)",
                "Total N",
                "Total S",
                "Total I",
                "Total D",
                "PER %",
                "Accuracy %",
            ]
        )

        print(f"\n{'=' * 60}")
        print(
            f"{'Chunk ms':>10} {'N':>6} {'S':>5} {'I':>5} {'D':>5} {'PER':>8} {'Acc':>8}"
        )
        print(f"{'=' * 60}")

        for ms in CHUNK_LENGTHS_MS:
            st = global_stats[ms]
            N = st["N"]
            S = st["S"]
            I = st["I"]
            D = st["D"]

            if N > 0:
                per = (S + I + D) / N
                acc = max(0.0, (N - S - I - D) / N)
            else:
                per = 0.0
                acc = 1.0

            writer.writerow(
                [
                    ms,
                    N,
                    S,
                    I,
                    D,
                    f"{per * 100:.1f}%",
                    f"{acc * 100:.1f}%",
                ]
            )
            print(
                f"{ms:>10} {N:>6} {S:>5} {I:>5} {D:>5} {per * 100:>7.1f}% {acc * 100:>7.1f}%"
            )
            print(f"{ms:>10} {N:>6} {S:>5} {I:>5} {D:>5} {per*100:>7.1f}% {acc*100:>7.1f}%")

    print(f"\nResults saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
