"""
--------------------------------------------------------------------------------
         CHORDONOMICON DATASET BUILDER FOR TRANSFORMER TRAINING              
--------------------------------------------------------------------------------

Source:  ailsntua/Chordonomicon (Hugging Face)
Output:  X_dataset.npy  --  shape (N, 100, 12)  -- CQT chroma sequences
         y_dataset.npy  --  shape (N,)           -- integer chord class labels
         chord_classes.npy                       -- 1-D string array of class names

HOW IT WORKS
------------
Each Chordonomicon sample has:
  • 'id'     — YouTube video ID
  • 'chords' — space-separated chord string e.g. "G D Em C"

Pipeline per song:
  1. Download audio from YouTube using yt-dlp → WAV
  2. Run HPSS to isolate harmonic content (drops drums/clicks)
  3. Extract chroma_cqt (12 pitch-class feature per frame)
  4. Slide a window of L=100 frames across the chromagram
  5. Label each window with the MOST COMMON chord in that window
     (using the chord annotation string's tokens, spread uniformly)
  6. Accumulate windows and labels; save as .npy

USAGE
-----
  python build_dataset.py              # processes MAX_SONGS=300 songs
  python build_dataset.py --songs 100  # process only 100 songs (quick test)
  python build_dataset.py --songs 1000 # larger dataset for better accuracy
"""

import os
import re
import sys
import glob
import argparse
import numpy as np
import librosa
import yt_dlp
import imageio_ffmpeg
from collections import Counter
from datasets import load_dataset

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_X       = "X_dataset.npy"
OUT_Y       = "y_dataset.npy"
OUT_CLASSES = "chord_classes.npy"
TMP_DIR     = "tmp_audio"

# ── Feature hyper-parameters (must match transformer_model.py) ────────────────
SR              = 22050
HOP_LENGTH      = 512
SEQUENCE_LENGTH = 100   # frames per window ≈ 2.3 s


# =============================================================================
# CHORD VOCABULARY
# Maps every chord alias we might see in Chordonomicon → one of 24 canonical
# classes.  Augmented, suspended, dominant-7 etc. are rounded to the nearest
# major/minor triad because the app only displays 24 classes and those
# extra colours are not playable as single guitar chords by a beginner.
# =============================================================================
_NOTE_MAP = {
    # Natural
    'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
    # Sharps
    'C#': 1, 'D#': 3, 'F#': 6, 'G#': 8, 'A#': 10,
    # Flats (enharmonic equivalents)
    'Db': 1, 'Eb': 3, 'Gb': 6, 'Ab': 8, 'Bb': 10,
}

# Ordered list of canonical 24 classes: C Cm C# C#m ... B Bm
_ROOTS  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
CHORD_CLASSES = [f"{r}{q}" for r in _ROOTS for q in ('','m')]
# e.g. ['C','Cm','C#','C#m','D','Dm', ...]

_CLASS_INDEX = {c: i for i, c in enumerate(CHORD_CLASSES)}


def parse_chord_token(token):
    """
    Convert a raw chord token (e.g. "Bmin", "G7", "Abmaj7", "F#m")
    to an integer class index in 0..23, or None if unrecognisable.

    PARSING STEPS:
    1. Strip trailing modifiers (7, maj7, sus4, add9, dim, aug ...) to get root+quality.
    2. Identify the root note (1 or 2 characters).
    3. Decide major vs minor:  'm', 'min', ':min' → minor; else → major.
    4. Return index of the canonical 'Root' or 'Rootm' class.
    """
    token = token.strip()
    if not token or token in ('N', 'N.C.', 'X', 'NC', '|'):
        return None

    # Remove common suffixes (keep m/min for quality detection)
    token = re.sub(r'(maj|add|sus|aug|dim|hdim|\d)+', '', token, flags=re.IGNORECASE)
    token = token.rstrip('/')   # remove bass note suffix like /E

    # Extract root: try 2-char first (e.g. C#, Bb), then 1-char
    root = None
    for length in (2, 1):
        candidate = token[:length].capitalize() if length == 1 else token[:2]
        # Fix capitalisation: first char upper, second char lower
        if length == 2:
            candidate = candidate[0].upper() + candidate[1].lower()
        if candidate in _NOTE_MAP:
            root = candidate
            quality_str = token[length:]
            break

    if root is None:
        return None

    # Determine major/minor
    is_minor = bool(re.search(r'^(m|min|:min)', quality_str, re.IGNORECASE))
    quality  = 'm' if is_minor else ''
    canonical = f"{root}{quality}"

    # Remap to sharp equivalent if needed (Db→C#, Bb→A# …)
    FLAT_TO_SHARP = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}
    root_sharp = FLAT_TO_SHARP.get(root, root)
    canonical  = f"{root_sharp}{quality}"

    return _CLASS_INDEX.get(canonical, None)


# =============================================================================
# AUDIO DOWNLOAD
# =============================================================================
def download_audio(video_id):
    """
    Download the YouTube audio for `video_id` and transcode to WAV.
    Returns the local WAV path, or None on failure.
    """
    os.makedirs(TMP_DIR, exist_ok=True)
    out_path = os.path.join(TMP_DIR, f"{video_id}.wav")

    if os.path.exists(out_path):
        return out_path   # already cached

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(TMP_DIR, f"{video_id}.%(ext)s"),
        'noplaylist': True,
        'quiet': True,
        'ffmpeg_location': ffmpeg_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '128',
        }]
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        # yt-dlp may produce .wav directly or via ffmpeg rename
        candidates = glob.glob(os.path.join(TMP_DIR, f"{video_id}.*"))
        wav_files  = [f for f in candidates if f.endswith('.wav')]
        return wav_files[0] if wav_files else None
    except Exception as e:
        print(f"    [skip] download failed for {video_id}: {e}")
        return None


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
def extract_chroma(wav_path):
    """
    Load WAV → HPSS → chroma_cqt.
    Returns np.ndarray shape (T, 12), or raises on error.

    MATH (see transformer_model.py for full explanation):
    • HPSS separates harmonic (sustained tones) from percussive (clicks/drums)
      using 2D median filtering on the magnitude spectrogram.
    • chroma_cqt maps every CQT frame to 12 pitch-class magnitudes.
    """
    y, _sr = librosa.load(wav_path, sr=SR, mono=True)
    y_harmonic = librosa.effects.harmonic(y, margin=8)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=SR, hop_length=HOP_LENGTH)
    return chroma.T   # (T, 12)


def chroma_to_windows_and_labels(chroma, chord_tokens):
    """
    Slice the chromagram into non-overlapping windows of SEQUENCE_LENGTH frames
    and assign a ground-truth label to each window.

    LABELLING STRATEGY:
    The chord annotation is a sequence of chord tokens (evenly distributed across
    the song's timeline).  For each window we pick the chord token that covers the
    midpoint of that window — simple but effective for most song structures.

    Returns:
      windows : np.ndarray  (N, 100, 12)
      labels  : list[int]   length N  —  class indices 0..23
    """
    T = chroma.shape[0]
    num_windows = T // SEQUENCE_LENGTH

    # Map chord tokens uniformly onto frames
    # token_at_frame[t] = index into chord_tokens list
    labels_by_frame = np.array(
        [min(int(t / T * len(chord_tokens)), len(chord_tokens) - 1)
         for t in range(T)]
    )

    windows = []
    labels  = []

    for i in range(num_windows):
        s = i * SEQUENCE_LENGTH
        e = s + SEQUENCE_LENGTH
        window_tokens = labels_by_frame[s:e]

        # Parse every chord token in this window
        class_counts = Counter()
        for tok_idx in window_tokens:
            cls = parse_chord_token(chord_tokens[tok_idx])
            if cls is not None:
                class_counts[cls] += 1

        if not class_counts:
            continue   # window has no parseable chords, skip

        # Most common class in this window = the label
        most_common_class = class_counts.most_common(1)[0][0]
        windows.append(chroma[s:e])
        labels.append(most_common_class)

    return np.array(windows, dtype=np.float32), labels


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def build_dataset(max_songs=300):
    print(f"\n{'='*60}")
    print(f" Chordonomicon Dataset Builder")
    print(f" Target: {max_songs} songs → {OUT_X}, {OUT_Y}")
    print(f"{'='*60}\n")

    # Stream from HuggingFace (no full download required)
    ds = load_dataset('ailsntua/Chordonomicon', split='train', streaming=True)

    all_X = []
    all_y = []

    processed   = 0
    skipped     = 0
    total_windows = 0

    for sample in ds:
        if processed >= max_songs:
            break

        video_id     = sample.get('id', '')
        chord_string = sample.get('chords', '')

        if not video_id or not chord_string:
            skipped += 1
            continue

        chord_tokens = chord_string.split()
        if len(chord_tokens) < 4:
            skipped += 1
            continue

        print(f"[{processed+1}/{max_songs}] {video_id}  ({len(chord_tokens)} chords: {chord_string[:50]}...)")

        # Download
        wav_path = download_audio(video_id)
        if not wav_path:
            skipped += 1
            continue

        # Extract features
        try:
            chroma = extract_chroma(wav_path)
        except Exception as e:
            print(f"    [skip] feature extraction failed: {e}")
            skipped += 1
            continue

        # Slice into labelled windows
        X_song, y_song = chroma_to_windows_and_labels(chroma, chord_tokens)
        if len(X_song) == 0:
            skipped += 1
            continue

        all_X.append(X_song)
        all_y.extend(y_song)
        total_windows += len(X_song)
        processed += 1
        print(f"    ✓ {len(X_song)} windows extracted. Total so far: {total_windows}")

    if total_windows == 0:
        print("\n[ERROR] No windows were extracted. Check your internet connection or yt-dlp.")
        return

    X = np.concatenate(all_X, axis=0)   # (N, 100, 12)
    y = np.array(all_y, dtype=np.int32) # (N,)

    np.save(OUT_X, X)
    np.save(OUT_Y, y)
    np.save(OUT_CLASSES, np.array(CHORD_CLASSES))

    # Class distribution
    counts = Counter(y.tolist())
    print(f"\n{'═'*60}")
    print(f" DONE — {processed} songs, {total_windows} training windows, {skipped} skipped")
    print(f" Saved: {OUT_X}  shape={X.shape}")
    print(f"        {OUT_Y}  shape={y.shape}")
    print(f"        {OUT_CLASSES}")
    print(f"\n Class distribution (top 10):")
    for cls_idx, cnt in counts.most_common(10):
        print(f"   {CHORD_CLASSES[cls_idx]:6s}  {cnt:5d} windows")
    print(f"{'='*60}\n")
    print("Next step: open transformer_chord_recognition.ipynb and replace")
    print("  the np.random.rand() arrays with np.load('X_dataset.npy') etc.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Chordonomicon training dataset.")
    parser.add_argument('--songs', type=int, default=300,
                        help='Number of songs to process (default: 300)')
    args = parser.parse_args()
    build_dataset(max_songs=args.songs)
