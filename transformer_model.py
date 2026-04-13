"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           BIDIRECTIONAL TRANSFORMER FOR GUITAR CHORD RECOGNITION            ║
║                  Full Mathematical Annotations                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
-------
Given a short audio clip (guitar + other instruments), predict which guitar
chord is being played. The 24 output classes are the 12 major and 12 minor
root chords: C, C#, D, ... B and Cm, C#m, Dm, ... Bm.


PIPELINE OVERVIEW
-----------------
Audio (WAV)
    │
    ▼──────────────────────────────────────────────────────────
    1. HARMONIC-PERCUSSIVE SOURCE SEPARATION (HPSS)
       Median‑filter the spectrogram along time (percussive) and
       frequency (harmonic) axes:
         H[f,t] = signal_component with smooth frequency structure
         P[f,t] = signal_component with sharp onset structure
       We keep only H, discarding drums and transient clicks.
    │
    ▼──────────────────────────────────────────────────────────
    2. CONSTANT-Q CHROMAGRAM (chroma_cqt)
       The Constant-Q Transform resolves frequency f_k = f_0 · 2^(k/B)
       where B = bins per octave. Unlike STFT (linear bins), CQT bins grow
       logarithmically — matching how musical pitch is perceived.
       
       The CHROMAGRAM folds all octaves onto 12 pitch classes (C through B)
       by summing magnitudes across octaves:
         C[p, t] = Σ_k  |X_CQT[k, t]|  where  k mod 12 == p
       
       Result shape: (12, T)  — 12 pitch classes × T time frames.
    │
    ▼──────────────────────────────────────────────────────────
    3. SLIDING WINDOW SEQUENCING
       We slice the (12, T) chromagram into overlapping windows of length L=200
       so the Transformer sees a 2.32-second context (L × hop/sr = 200 × 256/22050).

       Each window has shape (L, 12) = (200, 12) — a "sequence of 200 pitch
       vectors". Transformer input is SEQUENCE-shaped, not image-shaped, which
       is why we use it instead of a CNN.
    │
    ▼──────────────────────────────────────────────────────────
    4. BIDIRECTIONAL TRANSFORMER ENCODER  (this file)
    │
    ▼──────────────────────────────────────────────────────────
    5. PREDICTION → 24 softmax probabilities
"""

import os
import numpy as np
import tensorflow as tf
import librosa

# =============================================================================
# 0. HYPER‑PARAMETERS
# =============================================================================
SR             = 22050   # Audio sample rate (Hz)
HOP_LENGTH     = 256     # CQT hop size in samples.  Time resolution = HOP/SR ≈ 11.6 ms
                         # Matches FINE_HOP used in app.py inference — eliminates mismatch
SEQUENCE_LENGTH = 200    # Context window: 200 frames × 11.6 ms ≈ 2.32 seconds of audio
                         # Same wall-clock duration as before (was 100 × 23 ms), 2× finer resolution
FEATURE_DIM    = 12      # One value per chromatic pitch class (C, C#, ... B)
NUM_CHORD_CLASSES = 24   # 12 major + 12 minor chords
D_MODEL        = 64      # Latent embedding dimension throughout the Transformer
NUM_HEADS      = 4       # Parallel attention heads (must divide D_MODEL evenly)
FFN_DIM        = 128     # Feed-Forward Network hidden width (usually 2-4× D_MODEL)
NUM_LAYERS     = 4       # How many stacked Transformer encoder blocks
DROPOUT_RATE   = 0.1     # Fraction of neurons dropped during training


# =============================================================================
# 1. FEATURE EXTRACTION
# =============================================================================
def extract_chroma(audio_path):
    """
    Load audio and extract a Constant-Q Chromagram.

    MATH:
    -----
    CQT frequency bins:  f_k = f_0 · 2^(k/B),  k = 0 ... K-1
      where f_0 = lowest freq (typically C1 ≈ 32.7 Hz), B = 12 bins/octave.

    Each chroma bin p ∈ {0,...,11} aggregates all CQT bins at the same
    pitch class across octaves:
      chroma[p, t] = Σ_{k : k mod 12 == p}  |X_cqt[k, t]|

    The result is L2-normalised per frame so that loudness doesn't affect
    the chord classification — only relative pitch energy matters.

    Returns
    -------
    chroma : np.ndarray, shape (T, 12)
        T time frames, each with 12 normalised pitch-class magnitudes.
    """
    y, sr = librosa.load(audio_path, sr=SR)

    # Harmonic–Percussive Source Separation (HPSS)
    # Uses a 2D median filter on the spectrogram:
    #   harmonic  → smooth along frequency axis  (stationary tones)
    #   percussive → smooth along time axis       (sharp onsets)
    # margin=8 means harmonic must be ≥8× stronger than percussive to be kept.
    y_harmonic = librosa.effects.harmonic(y, margin=8)

    # Chroma CQT — shape (12, T)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=SR, hop_length=HOP_LENGTH)

    # Transpose to (T, 12) so each row is one time frame (matches Transformer input)
    return chroma.T


def slice_sequences(chroma):
    """
    Produce non-overlapping windows of shape (SEQUENCE_LENGTH, FEATURE_DIM).

    The Transformer needs a fixed-length sequence as input. We slide a window
    of L=200 frames across the full chromagram with stride=L (non-overlapping).
    Overlapping stride=1 would give more training data but is slower.

    Returns
    -------
    sequences : np.ndarray, shape (N, 200, 12)
    """
    T = chroma.shape[0]
    num_sequences = T // SEQUENCE_LENGTH
    seqs = []
    for i in range(num_sequences):
        s = i * SEQUENCE_LENGTH
        seqs.append(chroma[s : s + SEQUENCE_LENGTH])
    return np.array(seqs)  # (N, 100, 12)


# =============================================================================
# 2. POSITIONAL ENCODING
# =============================================================================
def positional_encoding(max_len, d_model):
    """
    Sinusoidal positional encoding (Vaswani et al., "Attention is All You Need").

    MATH:
    -----
    The Transformer has no built-in notion of order; self-attention is a
    set operation. Positional encoding injects position information by adding
    a deterministic vector PE[pos] to each frame embedding:

      PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
      PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    Properties:
    • Each position gets a unique pattern.
    • For any fixed offset k, PE[pos+k] can be expressed as a linear
      function of PE[pos] — the model can extrapolate to unseen lengths.
    • The wavelengths form a geometric progression from 2π (short-range)
      to 10000·2π (long-range), covering all relevant time scales.

    Returns
    -------
    pe : np.ndarray, shape (1, max_len, d_model)
    """
    positions = np.arange(max_len)[:, np.newaxis]        # (L, 1)
    dims      = np.arange(d_model)[np.newaxis, :]        # (1, D)
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)

    # Even indices → sine, odd indices → cosine
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return angles[np.newaxis, ...]  # (1, L, D)


# =============================================================================
# 3. MULTI-HEAD SELF-ATTENTION BLOCK
# =============================================================================
def scaled_dot_product_attention(Q, K, V):
    """
    Core attention operation.

    MATH:
    -----
    Given queries Q ∈ R^(L×d_k), keys K ∈ R^(L×d_k), values V ∈ R^(L×d_v):

      Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

    Intuition:
    • Q·Kᵀ  — dot product measures similarity between every pair of frames.
              Frame i "attends to" frame j proportional to their similarity.
    • / √d_k — scale factor prevents dot products from growing too large
              in high-dimensional spaces (which would push softmax into
              near-zero gradient regions).
    • softmax( · ) — converts raw scores to a probability distribution over
              source frames for each query frame.
    • · V   — weighted sum of value vectors = attended context.

    This is BIDIRECTIONAL because no causal mask is applied — every frame
    can attend to all other frames, both past AND future.
    """
    d_k = tf.cast(tf.shape(Q)[-1], tf.float32)
    scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(d_k)  # (L, L)
    weights = tf.nn.softmax(scores, axis=-1)                          # (L, L)
    return tf.matmul(weights, V)                                       # (L, d_v)


def build_transformer_encoder_block(d_model, num_heads, ffn_dim, dropout_rate, name="encoder_block"):
    """
    One Transformer Encoder Block:
      LayerNorm → MultiHeadAttention → Dropout → Residual
      LayerNorm → FFN(ReLU) → Dropout → Residual

    MULTI-HEAD ATTENTION MATH:
    --------------------------
    Split d_model into num_heads heads, each with dimension d_k = d_model/num_heads.
    For head h:
      Q_h = X · W_Q_h,   K_h = X · W_K_h,   V_h = X · W_V_h
      head_h = Attention(Q_h, K_h, V_h)

    Concatenate all heads then project:
      MultiHead(Q,K,V) = Concat(head_1,...,head_H) · W_O

    Multiple heads allow the model to jointly attend to information from
    different representational subspaces — e.g. one head might track the
    root note, another the third (major/minor quality), another rhythm.

    Pre-LayerNorm (applied BEFORE attention) is used here — it's more
    stable to train than the original post-norm formulation.

    FEED-FORWARD NETWORK (FFN) MATH:
    ---------------------------------
      FFN(x) = max(0, x·W_1 + b_1) · W_2 + b_2
    
    Applied independently at each time step (position-wise).
    Expands to ffn_dim=128, then projects back to d_model=64.
    This is where most of the model's "memorisation" capacity lives.

    RESIDUAL CONNECTIONS:
    ---------------------
      output = sublayer(LayerNorm(x)) + x

    Guarantees a clean gradient path back to the input (no vanishing
    gradient), enabling stable training of deep stacks.

    LAYER NORMALISATION:
    --------------------
      LayerNorm(x) = (x - μ) / (σ + ε)  ×  γ + β
    where μ, σ are computed per-sample across the feature dimension.
    Stabilises activations without depending on batch statistics, making
    it ideal for variable-length sequences.
    """
    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, d_model))

    # --- Self-Attention sub-layer ---
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads   # d_k = d_model / H for each head
    )(x, x)                            # query=key=value=x (self-attention)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Add()([inputs, x])   # Residual connection

    # --- Feed-Forward sub-layer ---
    skip = x
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dense(ffn_dim, activation='relu')(x)   # W_1  (expand)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(d_model)(x)                      # W_2  (project back)
    x = tf.keras.layers.Add()([skip, x])                       # Residual connection

    return tf.keras.Model(inputs, x, name=name)


# =============================================================================
# 4. FULL TRANSFORMER MODEL
# =============================================================================
def build_transformer(
    sequence_length=SEQUENCE_LENGTH,
    feature_dim=FEATURE_DIM,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    ffn_dim=FFN_DIM,
    num_layers=NUM_LAYERS,
    dropout_rate=DROPOUT_RATE,
    num_classes=NUM_CHORD_CLASSES
):
    """
    Input shape:  (batch, 200, 12)
    Output shape: (batch, 24)   — softmax probability over 24 chord classes

    FLOW:
    -----
    1. Linear projection: 12 → d_model=64
       Elevates the 12‑dimensional chroma vector into a richer d_model-dimensional
       embedding space where the Transformer can learn complex interactions.

         z[t] = chroma[t] · W_embed + b_embed,   W_embed ∈ R^(12 × 64)

    2. Add positional encoding PE ∈ R^(200 × 64) (precomputed, not learned).

    3. Pass through NUM_LAYERS=4 encoder blocks (each = LayerNorm+MHA+FFN+Residual).
       Stacking blocks lets the model build hierarchical representations:
         Layer 1 — local pitch relationships (adjacent frames)
         Layer 2 — short-term harmonic patterns (~4-8 frames)
         Layer 3 — chord transitions within the window
         Layer 4 — global chord context across 200 frames

    4. Global Average Pooling: collapse the sequence dimension
         z_global = (1/L) Σ_t z[t]   →  shape (batch, d_model)
       Averages all frame-level representations into one song-level vector.

    5. Classification head: Dense(128, ReLU) → Dropout → Dense(24, softmax)
       The softmax forces outputs to sum to 1:
         P(class k | x) = exp(logit_k) / Σ_j exp(logit_j)
       The predicted chord is argmax over these 24 probabilities.
    """
    inputs = tf.keras.Input(shape=(sequence_length, feature_dim), name="chroma_input")

    # --- Input Projection: 12 → d_model ---
    x = tf.keras.layers.Dense(d_model, name="input_projection")(inputs)

    # --- Add Positional Encoding ---
    # PE is a constant tensor added to x, not learned
    pe = positional_encoding(sequence_length, d_model).astype("float32")
    x = x + pe   # broadcast addition: (batch, L, d_model) + (1, L, d_model)

    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # --- Stack Encoder Blocks ---
    for layer_idx in range(num_layers):
        block = build_transformer_encoder_block(
            d_model, num_heads, ffn_dim, dropout_rate,
            name=f"encoder_block_{layer_idx}"
        )
        x = block(x)

    # --- Global Average Pooling ---
    # Compresses (batch, L=200, d_model=64) → (batch, 64)
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_pool")(x)

    # --- Classification Head ---
    x = tf.keras.layers.Dense(128, activation='relu', name="classifier_hidden")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="chord_output")(x)

    model = tf.keras.Model(inputs, outputs, name="BidirectionalTransformerChordRecognizer")
    return model


# =============================================================================
# 5. TRAINING SETUP
# =============================================================================
def compile_and_summary():
    """
    LOSS: Sparse Categorical Cross-Entropy
    ----------------------------------------
      L = - Σ_i  y_true_i · log(p_i)
    where y_true_i is a one-hot label (1 for correct chord, 0 elsewhere).
    
    Minimising this pushes the predicted probability of the correct class
    toward 1.0, which corresponds to high confidence.

    OPTIMIZER: Adam
    ---------------
    Adam adapts the learning rate per-parameter using first and second
    moment estimates of the gradient:
      m_t = β1·m_{t-1} + (1-β1)·g_t          (momentum)
      v_t = β2·v_{t-1} + (1-β2)·g_t²         (RMS)
      θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)

    Default β1=0.9, β2=0.999, ε=1e-7.  Learning rate α is set to 1e-4
    which is conservative — avoids overshooting in early training.
    """
    model = build_transformer()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    model.summary()
    return model


# =============================================================================
# 6. INFERENCE ON A REAL AUDIO FILE
# =============================================================================
CHORD_NAMES = [
    'C','C#','D','D#','E','F','F#','G','G#','A','A#','B',
    'Cm','C#m','Dm','D#m','Em','Fm','F#m','Gm','G#m','Am','A#m','Bm'
]
CHORD_CLASSES = CHORD_NAMES   # alias used by app.py and run_training.py

def predict_chords(audio_path, model):
    """
    Run end-to-end chord recognition on a full song.

    TEMPORAL SMOOTHING:
    -------------------
    Raw per-window predictions can flicker (e.g. G → G# → G → G).
    We merge consecutive identical predictions into segments:
      [(G, 0.0s-4.6s), (Em, 4.6s-9.2s), ...]
    This matches how a guitarist reads a chord chart.
    """
    chroma = extract_chroma(audio_path)
    sequences = slice_sequences(chroma)

    if len(sequences) == 0:
        print("Audio too short for analysis.")
        return

    # model.predict → shape (N, 24) probability arrays
    probs = model.predict(sequences, verbose=0)

    # Per-window: take argmax as the predicted chord class
    pred_classes = np.argmax(probs, axis=1)
    pred_confs   = np.max(probs, axis=1)

    # Convert frame index to approximate time
    seconds_per_window = SEQUENCE_LENGTH * HOP_LENGTH / SR  # 200 × 256/22050 ≈ 2.32 s

    # Temporal smoothing: merge consecutive identical chords
    segments = []
    if len(pred_classes) == 0:
        return segments

    current_chord = pred_classes[0]
    start_time    = 0.0
    for i, (chord, conf) in enumerate(zip(pred_classes, pred_confs)):
        if chord != current_chord:
            segments.append({
                "chord": CHORD_NAMES[current_chord],
                "start": start_time,
                "end"  : i * seconds_per_window
            })
            current_chord = chord
            start_time    = i * seconds_per_window

    # Append final segment
    segments.append({
        "chord": CHORD_NAMES[current_chord],
        "start": start_time,
        "end"  : len(pred_classes) * seconds_per_window
    })

    print("\n═══ CHORD TIMELINE ═══")
    for seg in segments:
        print(f"  {seg['start']:5.1f}s – {seg['end']:5.1f}s  →  {seg['chord']}")

    return segments


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import sys

    print("\n[ Building Transformer model... ]")
    model = compile_and_summary()

    if len(sys.argv) >= 2:
        audio_path = sys.argv[1]
        print(f"\n[ Loading saved weights from transformer_chord_model.keras ]")
        try:
            model.load_weights("transformer_chord_model.keras")
        except Exception as e:
            print(f"Could not load weights: {e}\nRunning with random weights (train first!)")

        predict_chords(audio_path, model)
    else:
        print("\nUsage: python transformer_model.py <path_to_audio.wav>")
        print("Model architecture printed above — train it with real data to get high accuracy.")
