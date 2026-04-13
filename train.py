"""
train.py -- Train the Bidirectional Transformer on Chordonomicon data.

RUN AFTER build_dataset.py:
  python train.py

Loads X_dataset.npy and y_dataset.npy produced by build_dataset.py and
trains the model defined in transformer_model.py.

Estimated training time (CPU-only, 300 songs ~ 10,000 windows):
  ~2-5 minutes per epoch  x  50 epochs  =  ~2-4 hours total
  (EarlyStopping typically halts at 20-35 epochs = ~1-2 hours)
  
GPU (if available) is 5-10x faster.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformer_model import (
    build_transformer,
    CHORD_CLASSES,
    SEQUENCE_LENGTH,
    FEATURE_DIM,
    NUM_CHORD_CLASSES,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
X_PATH       = "X_dataset.npy"
Y_PATH       = "y_dataset.npy"
MODEL_OUT    = "transformer_chord_model.keras"   # overwrites existing dummy model
CLASSES_PATH = "chord_classes.npy"               # created by build_dataset.py

# ── Training Hyper-parameters ──────────────────────────────────────────────────
BATCH_SIZE   = 64
MAX_EPOCHS   = 100
PATIENCE     = 10    # early stopping patience
LR           = 1e-4


def estimate_time(n_windows, n_epochs=50):
    """
    Print a rough wall-clock estimate before training starts.
    Benchmark: ~300 windows/sec on a typical CPU (i5/i7, no GPU).
    """
    secs_per_epoch = n_windows / 300.0
    total_secs     = secs_per_epoch * n_epochs
    print(f"\n  Estimated training time:")
    print(f"    Per epoch  : ~{secs_per_epoch/60:.1f} min  ({n_windows} windows at ~300 win/s)")
    print(f"    {n_epochs} epochs max: ~{total_secs/3600:.1f} hours")
    print(f"    EarlyStopping (patience={PATIENCE}) will likely stop at 20-35 epochs.")
    print(f"    Likely total: ~{total_secs*0.4/3600:.1f}–{total_secs*0.7/3600:.1f} hours\n")


def main():
    # ── 1. Load data ────────────────────────────────────────────────────────────
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print(f"[ERROR] {X_PATH} or {Y_PATH} not found.")
        print("  Run  python build_dataset.py --songs 300  first.")
        sys.exit(1)

    print("Loading dataset...")
    X = np.load(X_PATH)   # (N, 100, 12)
    y = np.load(Y_PATH)   # (N,)

    print(f"  X shape : {X.shape}  (N={len(X)} windows)")
    print(f"  y shape : {y.shape}")
    print(f"  Classes present: {len(set(y.tolist()))}/{NUM_CHORD_CLASSES}")

    # Guard: need all 24 classes for a meaningful model
    if len(set(y.tolist())) < 8:
        print("[WARN] Very few chord classes found. Consider running build_dataset.py with more songs.")

    estimate_time(len(X), MAX_EPOCHS)

    # ── 2. Train / validation split (80/20) ─────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y.tolist())) > 1 else None
    )
    print(f"  Train : {len(X_train)} windows")
    print(f"  Val   : {len(X_val)} windows\n")

    # ── 3. Build & compile model ─────────────────────────────────────────────────
    print("Building model...")
    model = build_transformer()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    # ── 4. Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        # Save the best weights (lowest val_loss)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_OUT,
            save_best_only=True,
            monitor='val_sparse_categorical_accuracy',
            verbose=1
        ),
        # Stop early if val accuracy plateaus
        tf.keras.callbacks.EarlyStopping(
            patience=PATIENCE,
            restore_best_weights=True,
            monitor='val_sparse_categorical_accuracy',
            verbose=1
        ),
        # Halve learning rate when val_loss plateaus for 5 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Print epoch timing so the user can extrapolate
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"  [Epoch {epoch+1}] "
                f"acc={logs.get('sparse_categorical_accuracy',0):.3f}  "
                f"val_acc={logs.get('val_sparse_categorical_accuracy',0):.3f}  "
                f"lr={model.optimizer.learning_rate.numpy():.2e}"
            )
        )
    ]

    # ── 5. Train ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("Starting training...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0    # suppress keras default output (our LambdaCallback handles it)
    )

    elapsed = time.time() - t0
    best_acc = max(history.history.get('val_sparse_categorical_accuracy', [0]))

    print(f"\n{'='*55}")
    print(f"  Training complete in {elapsed/60:.1f} minutes")
    print(f"  Best validation accuracy: {best_acc*100:.1f}%")
    print(f"  Model saved to: {MODEL_OUT}")
    print(f"{'='*55}")

    if best_acc < 0.5:
        print("\n[TIP] Accuracy is low. Try running build_dataset.py with --songs 1000.")
        print("      More labelled data = better model.\n")
    else:
        print("\n[DONE] Model is ready. The app will automatically use it on next restart.\n")


if __name__ == "__main__":
    main()
