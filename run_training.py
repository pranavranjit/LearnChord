import os
import time
import argparse
import glob as _glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformer_model import (
    build_transformer,
    SEQUENCE_LENGTH,
    FEATURE_DIM,
    NUM_CHORD_CLASSES,
    CHORD_NAMES,
    extract_chroma,
)

X_PATH     = "X_dataset.npy"
Y_PATH     = "y_dataset.npy"
MODEL_PATH = "transformer_chord_model.keras"

BATCH_SIZE_TRAIN = 64
MAX_EPOCHS       = 60
PATIENCE         = 8
LR               = 1e-3

SONG_CACHE_DIR   = "training_audio"
CONF_THRESHOLD   = 0.78
OVERLAP_STRIDE   = 50
MIN_PER_CLASS    = 80

TRAINING_SONGS = [
    "Ed Sheeran Perfect official",
    "The Beatles Let It Be official",
    "Adele Someone Like You official",
    "Oasis Wonderwall official",
    "John Legend All of Me official",
    "Coldplay Yellow official",
    "Eric Clapton Wonderful Tonight official",
    "Tracy Chapman Fast Car official",
    "Passenger Let Her Go official",
    "Bob Marley No Woman No Cry official",
    "The Cranberries Zombie official",
    "Radiohead Creep official",
    "Red Hot Chili Peppers Under The Bridge official",
    "U2 With or Without You official",
    "Pink Floyd Wish You Were Here official",
    "Fleetwood Mac Dreams official",
    "Imagine Dragons Radioactive official",
    "Green Day Boulevard of Broken Dreams official",
    "The Weeknd Blinding Lights official",
    "Bruno Mars Just The Way You Are official",
    "Eagles Hotel California official",
    "Nirvana Come As You Are official",
    "Metallica Nothing Else Matters official",
    "Amy Winehouse Back to Black official",
    "Lana Del Rey Video Games official",
    "Toto Africa official",
    "Queen Love of My Life official",
    "Beyonce Halo official",
    "Guns N Roses Sweet Child O Mine official",
    "Coldplay The Scientist official",
]


def _chord_template(class_idx):
    if class_idx < 12:
        root = class_idx
        intervals = [0, 4, 7]
    else:
        root = class_idx - 12
        intervals = [0, 3, 7]
    vec = np.zeros(12, dtype=np.float32)
    for iv in intervals:
        vec[(root + iv) % 12] = 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _apply_rhythm_density(window, factor, rng):
    L = window.shape[0]
    if abs(factor - 1.0) < 1e-6:
        return window.copy()
    if factor < 1.0:
        active = max(1, int(round(factor * L)))
        out = np.zeros_like(window)
        out[:active] = window[:active]
        noise = rng.uniform(0.0, 0.12, size=(L - active, window.shape[1])).astype(np.float32)
        norms = np.linalg.norm(noise, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        out[active:] = noise / norms
        return out
    else:
        src_len = max(1, int(round(L / factor)))
        src = window[:src_len]
        indices = np.linspace(0, src_len - 1, L).astype(int)
        return src[indices]


def generate_synthetic_dataset(samples_per_class=3000):
    rng = np.random.default_rng(42)
    X_list, y_list = [], []

    DENSITY_FACTORS = [0.75, 1.0, 1.33]

    for cls in range(NUM_CHORD_CLASSES):
        root = cls if cls < 12 else cls - 12
        intervals = [0, 3, 7] if cls >= 12 else [0, 4, 7]
        chord_tones = [(root + iv) % 12 for iv in intervals]

        for _ in range(samples_per_class):
            tone_amp = rng.uniform(0.55, 1.0, size=3)

            window = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
            for t in range(SEQUENCE_LENGTH):
                frame = np.zeros(12, dtype=np.float32)

                for j, tone in enumerate(chord_tones):
                    frame[tone] = tone_amp[j] * rng.uniform(0.88, 1.0)

                frame += rng.uniform(0.0, 0.18, size=12)

                if rng.random() < 0.15:
                    frame[(root + 10) % 12] += rng.uniform(0.1, 0.3)
                if rng.random() < 0.10:
                    frame[(root + 2) % 12] += rng.uniform(0.05, 0.2)

                norm = np.linalg.norm(frame)
                window[t] = frame / norm if norm > 0 else frame

            for factor in DENSITY_FACTORS:
                X_list.append(_apply_rhythm_density(window, factor, rng))
                y_list.append(cls)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def _create_chord_templates():
    pitch = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    tpl = {}
    for i, root in enumerate(pitch):
        maj = np.zeros(12, dtype=np.float32)
        maj[i] = maj[(i + 4) % 12] = maj[(i + 7) % 12] = 1.0
        tpl[root] = maj / np.linalg.norm(maj)
        minor = np.zeros(12, dtype=np.float32)
        minor[i] = minor[(i + 3) % 12] = minor[(i + 7) % 12] = 1.0
        tpl[f"{root}m"] = minor / np.linalg.norm(minor)
    return tpl


def _classify_window(window, templates):
    profile = np.median(window, axis=0)
    norm = np.linalg.norm(profile)
    if norm < 1e-8:
        return None, 0.0
    profile /= norm
    scores = sorted(
        [(float(np.dot(profile, tpl)), name) for name, tpl in templates.items()],
        reverse=True
    )
    best_score, best_name = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0.0
    # Reject transitional windows where two chords overlap
    if second_score > 0.62:
        return None, 0.0
    return best_name, best_score


def _download_song(query, cache_dir):
    import yt_dlp
    import imageio_ffmpeg

    os.makedirs(cache_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in " -_" else "" for c in query).strip()
    safe = safe.replace(" ", "_")[:80]
    wav_path = os.path.join(cache_dir, f"{safe}.wav")
    if os.path.exists(wav_path):
        return wav_path

    out_tpl = os.path.join(cache_dir, f"{safe}.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_tpl,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe(),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"ytsearch1:{query}"])
    except Exception as e:
        print(f"[SKIP] download failed: {e}")
        return None

    if os.path.exists(wav_path):
        return wav_path
    matches = _glob.glob(os.path.join(cache_dir, f"{safe}.*"))
    return matches[0] if matches else None


def build_real_dataset(song_list=None):
    if song_list is None:
        song_list = TRAINING_SONGS

    templates  = _create_chord_templates()
    X_raw, y_raw = [], []

    print(f"\n  Downloading & processing {len(song_list)} songs "
          f"(cached in '{SONG_CACHE_DIR}/')...\n")

    for i, query in enumerate(song_list, 1):
        print(f"  [{i:2d}/{len(song_list)}] {query}...", end=" ", flush=True)
        wav = _download_song(query, SONG_CACHE_DIR)
        if not wav or not os.path.exists(wav):
            print("SKIP")
            continue
        try:
            chroma = extract_chroma(wav)
            T = chroma.shape[0]
            count = 0
            for start in range(0, T - SEQUENCE_LENGTH, OVERLAP_STRIDE):
                win = chroma[start: start + SEQUENCE_LENGTH]
                name, conf = _classify_window(win, templates)
                if name and conf >= CONF_THRESHOLD and name in CHORD_NAMES:
                    X_raw.append(win)
                    y_raw.append(CHORD_NAMES.index(name))
                    count += 1
            print(f"{count} windows")
        except Exception as e:
            print(f"ERROR: {e}")

    if not X_raw:
        print("  [ERROR] No windows extracted from real audio.")
        return None, None

    print(f"\n  Raw windows: {len(X_raw)}  ->  pitch-augmenting x12...")

    # Roll chroma bins by k semitones = transpose key without distortion
    X_aug, y_aug = [], []
    for win, cls in zip(X_raw, y_raw):
        for shift in range(12):
            X_aug.append(np.roll(win, shift, axis=1))
            if cls < 12:
                y_aug.append((cls + shift) % 12)
            else:
                y_aug.append(12 + (cls - 12 + shift) % 12)

    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.int32)

    counts = np.bincount(y_aug, minlength=NUM_CHORD_CLASSES)
    rare   = [c for c in range(NUM_CHORD_CLASSES) if counts[c] < MIN_PER_CLASS]
    if rare:
        print(f"  Supplementing {len(rare)} rare classes with synthetic data...")
        X_syn, y_syn = [], []
        rng = np.random.default_rng(42)
        for cls in rare:
            needed = MIN_PER_CLASS - counts[cls]
            root = cls if cls < 12 else cls - 12
            intervals = [0, 4, 7] if cls < 12 else [0, 3, 7]
            chord_tones = [(root + iv) % 12 for iv in intervals]
            for _ in range(needed):
                tone_amp = rng.uniform(0.55, 1.0, size=3)
                win = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
                for t in range(SEQUENCE_LENGTH):
                    frame = np.zeros(12, dtype=np.float32)
                    for j, tone in enumerate(chord_tones):
                        frame[tone] = tone_amp[j] * rng.uniform(0.88, 1.0)
                    frame += rng.uniform(0.0, 0.18, size=12)
                    n = np.linalg.norm(frame)
                    win[t] = frame / n if n > 0 else frame
                X_syn.append(win)
                y_syn.append(cls)
        X_aug = np.concatenate([X_aug, np.array(X_syn, dtype=np.float32)])
        y_aug = np.concatenate([y_aug, np.array(y_syn, dtype=np.int32)])

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X_aug))
    X_aug, y_aug = X_aug[idx], y_aug[idx]

    counts_final = np.bincount(y_aug, minlength=NUM_CHORD_CLASSES)
    print(f"\n  Total samples : {len(X_aug)}")
    print(f"  Per-class min : {counts_final.min()}  max : {counts_final.max()}")

    return X_aug, y_aug


def train(X, y):
    print("\n" + "=" * 60)
    print("  TRAINING PHASE")
    print("=" * 60)
    print(f"  Dataset: {len(X)} windows, {NUM_CHORD_CLASSES} chord classes")

    try:
        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
    except ValueError:
        X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.15, random_state=42)

    model = build_transformer()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, save_best_only=True,
            monitor='val_sparse_categorical_accuracy', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=PATIENCE, restore_best_weights=True,
            monitor='val_sparse_categorical_accuracy', verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
    ]

    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE_TRAIN,
        callbacks=callbacks,
        verbose=1
    )

    elapsed = (time.time() - t0) / 60
    best_acc = max(history.history.get('val_sparse_categorical_accuracy', [0]))

    print(f"\n{'=' * 60}")
    print(f"  Training done in {elapsed:.1f} minutes")
    print(f"  Best val accuracy : {best_acc * 100:.1f}%")
    print(f"  Model saved to    : {MODEL_PATH}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=3000,
                        help='Synthetic samples per chord class (default: 3000)')
    parser.add_argument('--real', action='store_true',
                        help='Train on real music downloaded from YouTube (~30 songs, '
                             '15-25 min first run; subsequent runs use cached audio)')
    parser.add_argument('--songs', type=int, default=0,
                        help='Limit number of songs in --real mode (0 = all)')
    args = parser.parse_args()

    print(f"\n{'=' * 60}")

    if args.real:
        song_list = TRAINING_SONGS
        if args.songs > 0:
            song_list = song_list[:args.songs]
        print(f"  Real-Music Training Pipeline")
        print(f"  Songs to process  : {len(song_list)}")
        print(f"  Cache directory   : {SONG_CACHE_DIR}/")
        print(f"  Conf. threshold   : {CONF_THRESHOLD}")
        print(f"  Pitch augmentation: x12 (covers all keys)")
        print(f"{'=' * 60}")

        X, y = build_real_dataset(song_list)
        if X is None:
            print("  Falling back to synthetic data...")
            X, y = generate_synthetic_dataset(samples_per_class=args.samples)
    else:
        total = args.samples * NUM_CHORD_CLASSES * 3
        print(f"  Synthetic Training Pipeline")
        print(f"  Samples per class : {args.samples} x 3 variants  (total: {total})")
        print(f"  No downloads required -- data generated instantly!")
        print(f"{'=' * 60}\n")

        t_gen = time.time()
        X, y = generate_synthetic_dataset(samples_per_class=args.samples)
        print(f"  Generated {len(X)} samples in {time.time() - t_gen:.1f}s")

    np.save(X_PATH, X)
    np.save(Y_PATH, y)
    print(f"  Dataset saved -> {X_PATH}, {Y_PATH}")

    train(X, y)


if __name__ == "__main__":
    main()
