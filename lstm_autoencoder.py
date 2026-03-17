import os
import numpy as np
import tensorflow as tf

# Use tf.keras directly works with TensorFlow 2.20.0
from tf_keras.models import Model
from tf_keras.layers import (
    Input, LSTM, Dense,
    RepeatVector, TimeDistributed
)
from tf_keras.callbacks import EarlyStopping

# ─────────────────────────────────────────────
# STEP 1: Load normal windows
# ─────────────────────────────────────────────

print("=" * 50)
print("STEP 1: Loading normal_windows.npy ...")
print("=" * 50)

normal_windows = np.load("normal_windows.npy")

print(f"  normal_windows shape : {normal_windows.shape}")
# Expected: (9951, 50, 8)
# Meaning : 9951 windows, each 50 time steps, 8 features


# ─────────────────────────────────────────────
# STEP 2: Define input shape
# ─────────────────────────────────────────────

WINDOW_SIZE = normal_windows.shape[1]   # 50
N_FEATURES  = normal_windows.shape[2]   # 8

print(f"\n  Window size : {WINDOW_SIZE}")
print(f"  Features    : {N_FEATURES}")


# ─────────────────────────────────────────────
# STEP 3: Build LSTM Autoencoder
# ─────────────────────────────────────────────

print("STEP 2: Building LSTM Autoencoder ...")

# --- Input layer ---
inputs = Input(shape=(WINDOW_SIZE, N_FEATURES), name="input_layer")

# --- Encoder ---
# First LSTM layer — keeps full sequence for richer encoding
x = LSTM(64, return_sequences=True, name="encoder_lstm1")(inputs)

# Second LSTM layer — compresses to single context vector
x = LSTM(32, return_sequences=False, name="encoder_lstm2")(x)

# --- Bottleneck → Decoder bridge ---
# RepeatVector repeats the context vector 50 times
# so the decoder has one input per time step
x = RepeatVector(WINDOW_SIZE, name="repeat_vector")(x)

# --- Decoder ---
# First decoder LSTM — starts reconstructing the sequence
x = LSTM(32, return_sequences=True, name="decoder_lstm1")(x)

# Second decoder LSTM — expands back to full feature space
x = LSTM(64, return_sequences=True, name="decoder_lstm2")(x)

# Output layer — reconstruct 8 features at each of 50 time steps
outputs = TimeDistributed(Dense(N_FEATURES), name="output_layer")(x)

# --- Assemble model ---
model = Model(inputs=inputs, outputs=outputs, name="LSTM_Autoencoder")


# ─────────────────────────────────────────────
# STEP 4: Compile the model
# ─────────────────────────────────────────────

model.compile(
    optimizer="adam",
    loss="mse"          # Mean Squared Error — measures reconstruction quality
)

print("\n  Model compiled with Adam optimizer and MSE loss.")


# ─────────────────────────────────────────────
# STEP 5: Print model summary
# ─────────────────────────────────────────────

print("STEP 3: Model Summary")
model.summary()


# ─────────────────────────────────────────────
# STEP 6: Train ONLY on normal data
# ─────────────────────────────────────────────

print("STEP 4: Training on normal data only ...")

# Early stopping — stops training if validation loss
# stops improving for 5 consecutive epochs
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Note: Input = Output (autoencoder learns to reconstruct its own input)
history = model.fit(
    normal_windows,         # input
    normal_windows,         # target (same as input)
    epochs=30,
    batch_size=32,
    validation_split=0.1,   # 10% of normal data used for validation
    verbose=1,
    callbacks=[early_stop]
)

print("\n  Training complete!")
print(f"  Final training loss   : {history.history['loss'][-1]:.6f}")
print(f"  Final validation loss : {history.history['val_loss'][-1]:.6f}")


# ─────────────────────────────────────────────
# STEP 7: Save the trained model
# ─────────────────────────────────────────────

print("STEP 5: Saving model ...")

# Create models/ folder if it does not exist
os.makedirs("models", exist_ok=True)

model.save("models/lstm_autoencoder.h5")

print("  Model saved -> models/lstm_autoencoder.h5")

print("Done! LSTM Autoencoder trained and saved successfully.")