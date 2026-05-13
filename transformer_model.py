import os
import numpy as np
import tensorflow as tf
import librosa

SR             = 22050
HOP_LENGTH     = 256   # matches FINE_HOP in app.py
SEQUENCE_LENGTH = 200
FEATURE_DIM    = 12
NUM_CHORD_CLASSES = 24
D_MODEL        = 64
NUM_HEADS      = 4
FFN_DIM        = 128
NUM_LAYERS     = 4
DROPOUT_RATE   = 0.1


def extract_chroma(audio_path):
    y, sr = librosa.load(audio_path, sr=SR)
    y_harmonic = librosa.effects.harmonic(y, margin=8)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=SR, hop_length=HOP_LENGTH)
    return chroma.T


def slice_sequences(chroma):
    T = chroma.shape[0]
    num_sequences = T // SEQUENCE_LENGTH
    seqs = []
    for i in range(num_sequences):
        s = i * SEQUENCE_LENGTH
        seqs.append(chroma[s : s + SEQUENCE_LENGTH])
    return np.array(seqs)


def positional_encoding(max_len, d_model):
    positions = np.arange(max_len)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return angles[np.newaxis, ...]


def scaled_dot_product_attention(Q, K, V):
    d_k = tf.cast(tf.shape(Q)[-1], tf.float32)
    scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(d_k)
    weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(weights, V)


def build_transformer_encoder_block(d_model, num_heads, ffn_dim, dropout_rate, name="encoder_block"):
    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, d_model))

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Add()([inputs, x])

    skip = x
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dense(ffn_dim, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(d_model)(x)
    x = tf.keras.layers.Add()([skip, x])

    return tf.keras.Model(inputs, x, name=name)


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
    inputs = tf.keras.Input(shape=(sequence_length, feature_dim), name="chroma_input")
    x = tf.keras.layers.Dense(d_model, name="input_projection")(inputs)

    pe = positional_encoding(sequence_length, d_model).astype("float32")
    x = x + pe

    x = tf.keras.layers.Dropout(dropout_rate)(x)

    for layer_idx in range(num_layers):
        block = build_transformer_encoder_block(
            d_model, num_heads, ffn_dim, dropout_rate,
            name=f"encoder_block_{layer_idx}"
        )
        x = block(x)

    x = tf.keras.layers.GlobalAveragePooling1D(name="global_pool")(x)
    x = tf.keras.layers.Dense(128, activation='relu', name="classifier_hidden")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="chord_output")(x)

    model = tf.keras.Model(inputs, outputs, name="BidirectionalTransformerChordRecognizer")
    return model


def compile_and_summary():
    model = build_transformer()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    model.summary()
    return model


CHORD_NAMES = [
    'C','C#','D','D#','E','F','F#','G','G#','A','A#','B',
    'Cm','C#m','Dm','D#m','Em','Fm','F#m','Gm','G#m','Am','A#m','Bm'
]
CHORD_CLASSES = CHORD_NAMES

def predict_chords(audio_path, model):
    chroma = extract_chroma(audio_path)
    sequences = slice_sequences(chroma)

    if len(sequences) == 0:
        print("Audio too short for analysis.")
        return

    probs = model.predict(sequences, verbose=0)
    pred_classes = np.argmax(probs, axis=1)
    pred_confs   = np.max(probs, axis=1)

    seconds_per_window = SEQUENCE_LENGTH * HOP_LENGTH / SR

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

    segments.append({
        "chord": CHORD_NAMES[current_chord],
        "start": start_time,
        "end"  : len(pred_classes) * seconds_per_window
    })

    print("\n=== CHORD TIMELINE ===")
    for seg in segments:
        print(f"  {seg['start']:5.1f}s - {seg['end']:5.1f}s  ->  {seg['chord']}")

    return segments


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
        print("Model architecture printed above - train it with real data to get high accuracy.")
