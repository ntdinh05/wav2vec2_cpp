#!/usr/bin/env python3
"""
Sort UltraSuite core-uxtd text files from longest to shortest utterance.

For each .txt file the utterance is the first line (the prompt text).
Length is measured in number of characters (whitespace included).

Usage:
    python3 sort_by_utterance_length.py [--output OUTPUT_CSV]
"""

import argparse
import csv
import re
from pathlib import Path

CORE_DIR = Path("/home/ultraspeech-dev/ultrasuite/core-uxtd/core")

# ---------------------------------------------------------------------------
# English word set — NLTK preferred, falls back to /usr/share/dict/words
# ---------------------------------------------------------------------------
def _build_word_set() -> set[str]:
    try:
        from nltk.corpus import words as _nltk_words
        return set(w.lower() for w in _nltk_words.words())
    except Exception:
        pass
    _dict = Path("/usr/share/dict/words")
    if _dict.exists():
        return set(_dict.read_text(encoding="utf-8", errors="ignore").lower().split())
    return set()

_ENGLISH_WORDS: set[str] = _build_word_set()


def is_meaningful_english(utterance: str, min_words: int = 2, min_ratio: float = 0.75) -> bool:
    """Return True only if the utterance is composed mainly of real English words.

    Filters out phonetic labels (e.g. "close-back-unrounded-vowel post-test no-context")
    and nonsense syllables (e.g. "th atha eethee otho").  Hyphenated compound tokens
    (e.g. "post-test") are treated as a single entry and will typically not appear in
    the dictionary, so they fail the check.
    """
    if not utterance:
        return False
    if not _ENGLISH_WORDS:          # no word list available — keep everything
        return True
    tokens = utterance.split()
    if len(tokens) < min_words:
        return False
    real = sum(
        1 for t in tokens
        if re.sub(r"[^a-z']", "", t.lower()) in _ENGLISH_WORDS
    )
    return real / len(tokens) >= min_ratio


DEFAULT_OUTPUT = Path(
    "/home/ultraspeech-dev/Documents/ultraspeech-dev/"
    "wav2vec2-standalone-testing/wave2vec2_cpp/tests/utterances_by_length.csv"
)


def read_utterance(filepath: Path) -> str:
    """Return the first line (utterance/prompt) of a text file, stripped."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.readline().strip()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Sort UltraSuite text files by utterance length (longest first)."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Path for the output CSV file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    output_csv = Path(args.output)

    txt_files = sorted(CORE_DIR.rglob("*.txt"))
    print(f"Found {len(txt_files)} text files in: {CORE_DIR}")

    rows = []
    skipped = 0
    for filepath in txt_files:
        utterance = read_utterance(filepath)
        if not is_meaningful_english(utterance):
            skipped += 1
            continue
        rows.append(
            {
                "speaker":          filepath.parent.name,
                "filename":         filepath.stem,
                "utterance":        utterance,
                "utterance_length": len(utterance),
                "word_count":       len(utterance.split()) if utterance else 0,
                "filepath":         str(filepath),
            }
        )

    # Sort descending by utterance character length, then by utterance text for ties
    rows.sort(key=lambda r: (r["utterance_length"], r["utterance"]), reverse=True)

    print(f"Skipped {skipped} non-English / nonsense utterances  →  {len(rows)} kept")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["rank", "utterance_length", "word_count", "utterance", "speaker", "filename", "filepath"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({
                "rank":             rank,
                "utterance_length": row["utterance_length"],
                "word_count":       row["word_count"],
                "utterance":        row["utterance"],
                "speaker":          row["speaker"],
                "filename":         row["filename"],
                "filepath":         row["filepath"],
            })

    print(f"Sorted {len(rows)} files — longest utterance: {rows[0]['utterance_length']} chars")
    print(f"Output : {output_csv}")

    print("\nTop 5 longest utterances:")
    for row in rows[:5]:
        print(f"  [{row['utterance_length']} chars] {row['utterance']!r}  ({row['speaker']}/{row['filename']})")

    print("\nTop 5 shortest utterances:")
    for row in rows[-5:]:
        print(f"  [{row['utterance_length']} chars] {row['utterance']!r}  ({row['speaker']}/{row['filename']})")


if __name__ == "__main__":
    main()
