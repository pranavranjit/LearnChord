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

OUT_X       = "X_dataset.npy"
OUT_Y       = "y_dataset.npy"
OUT_CLASSES = "chord_classes.npy"
TMP_DIR     = "tmp_audio"

SR              = 22050
HOP_LENGTH      = 512
SEQUENCE_LENGTH = 100


_NOTE_MAP = {
    'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
    'C#': 1, 'D#': 3, 'F#': 6, 'G#': 8, 'A#': 10,
    'Db': 1, 'Eb': 3, 'Gb': 6, 'Ab': 8, 'Bb': 10,
}

_ROOTS  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
CHORD_CLASSES = [f"{r}{q}" for r in _ROOTS for q in ('','m')]

_CLASS_INDEX = {c: i for i, c in enumerate(CHORD_CLASSES)}


def parse_chord_token(token):
    token = token.strip()
    if not token or token in ('N', 'N.C.', 'X', 'NC', '|'):
        return None

    token = re.sub(r'(maj|add|sus|aug|dim|hdim|\d)+', '', token, flags=re.IGNORECASE)
    token = token.rstrip('/')

    root = None
    for length in (2, 1):
        candidate = token[:length].capitalize() if length == 1 else token[:2]
        if length == 2:
            candidate = candidate[0].upper() + candidate[1].lower()
        if candidate in _NOTE_MAP:
            root = candidate
            quality_str = token[length:]
            break

    if root is None:
        return None

    is_minor = bool(re.search(r'^(m|min|:min)', quality_str, re.IGNORECASE))
    quality  = 'm' if is_minor else ''
    canonical = f"{root}{quality}"

    FLAT_TO_SHARP = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}
    root_sharp = FLAT_TO_SHARP.get(root, root)
    canonical  = f"{root_sharp}{quality}"

    return _CLASS_INDEX.get(canonical, None)


def download_audio(video_id):
    os.makedirs(TMP_DIR, exist_ok=True)
    out_path = os.path.join(TMP_DIR, f"{video_id}.wav")

    if os.path.exists(out_path):
        return out_path

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

        candidates = glob.glob(os.path.join(TMP_DIR, f"{video_id}.*"))
        wav_files  = [f for f in candidates if f.endswith('.wav')]
        return wav_files[0] if wav_files else None
    except Exception as e:
        print(f"    [skip] download failed for {video_id}: {e}")
        return None


def extract_chroma(wav_path):
    y, _sr = librosa.load(wav_path, sr=SR, mono=True)
    y_harmonic = librosa.effects.harmonic(y, margin=8)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=SR, hop_length=HOP_LENGTH)
    return chroma.T


def chroma_to_windows_and_labels(chroma, chord_tokens):
    T = chroma.shape[0]
    num_windows = T // SEQUENCE_LENGTH

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

        class_counts = Counter()
        for tok_idx in window_tokens:
            cls = parse_chord_token(chord_tokens[tok_idx])
            if cls is not None:
                class_counts[cls] += 1

        if not class_counts:
            continue

        most_common_class = class_counts.most_common(1)[0][0]
        windows.append(chroma[s:e])
        labels.append(most_common_class)

    return np.array(windows, dtype=np.float32), labels


def build_dataset(max_songs=300):
    print(f"\n{'='*60}")
    print(f" Chordonomicon Dataset Builder")
    print(f" Target: {max_songs} songs -> {OUT_X}, {OUT_Y}")
    print(f"{'='*60}\n")

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

        wav_path = download_audio(video_id)
        if not wav_path:
            skipped += 1
            continue

        try:
            chroma = extract_chroma(wav_path)
        except Exception as e:
            print(f"    [skip] feature extraction failed: {e}")
            skipped += 1
            continue

        X_song, y_song = chroma_to_windows_and_labels(chroma, chord_tokens)
        if len(X_song) == 0:
            skipped += 1
            continue

        all_X.append(X_song)
        all_y.extend(y_song)
        total_windows += len(X_song)
        processed += 1
        print(f"    {len(X_song)} windows extracted. Total so far: {total_windows}")

    if total_windows == 0:
        print("\n[ERROR] No windows were extracted. Check your internet connection or yt-dlp.")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.array(all_y, dtype=np.int32)

    np.save(OUT_X, X)
    np.save(OUT_Y, y)
    np.save(OUT_CLASSES, np.array(CHORD_CLASSES))

    counts = Counter(y.tolist())
    print(f"\n{'='*60}")
    print(f" DONE - {processed} songs, {total_windows} training windows, {skipped} skipped")
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
