import numpy as np
import tensorflow as tf
import tf_keras


# ─────────────────────────────────────────────
# STEP 1: Load the trained LSTM Autoencoder
# ─────────────────────────────────────────────

print("STEP 1: Loading trained model ...")

model = tf_keras.models.load_model("models/lstm_model_v2.h5")

print("  Model loaded successfully!")


# ─────────────────────────────────────────────
# STEP 2: Load both window datasets
# ─────────────────────────────────────────────

print("STEP 2: Loading window datasets ")

normal_windows = np.load("normal_windows.npy")
mixed_windows  = np.load("mixed_windows.npy")

print(f"  normal_windows shape : {normal_windows.shape}")
print(f"  mixed_windows  shape : {mixed_windows.shape}")


# ─────────────────────────────────────────────
# STEP 3: Reconstruct windows using the model
# ─────────────────────────────────────────────

print("STEP 3: Reconstructing windows ")

# model.predict() passes each window through the autoencoder
# Input  shape: (num_windows, 50, 8)
# Output shape: (num_windows, 50, 8)  — reconstructed version

print("  Reconstructing normal windows ...")
normal_reconstructed = model.predict(normal_windows, batch_size=32, verbose=1)

print("  Reconstructing mixed windows ...")
mixed_reconstructed  = model.predict(mixed_windows,  batch_size=32, verbose=1)


# ─────────────────────────────────────────────
# STEP 4: Compute reconstruction error (MSE)
# ─────────────────────────────────────────────

print("STEP 4: Computing reconstruction errors ")

# For each window compute the Mean Squared Error
# between the original and reconstructed window.
#
# Formula per window:
#   MSE = mean( (original - reconstructed)^2 )
#   averaged over all 50 time steps and 8 features
#
# axis=(1, 2) means we average over time steps (axis 1)
# and features (axis 2), giving one error value per window.
#
# Result shape: (num_windows,)  — one error score per window

normal_errors = np.mean(
    np.square(normal_windows - normal_reconstructed),
    axis=(1, 2)
)

mixed_errors = np.mean(
    np.square(mixed_windows - mixed_reconstructed),
    axis=(1, 2)
)

print(f"  normal_errors shape : {normal_errors.shape}")
print(f"  mixed_errors  shape : {mixed_errors.shape}")


# ─────────────────────────────────────────────
# STEP 5: Print mean reconstruction errors
# ─────────────────────────────────────────────

print("STEP 5: Mean Reconstruction Errors")
print(f"  Mean error — normal windows : {np.mean(normal_errors):.6f}")
print(f"  Mean error — mixed  windows : {np.mean(mixed_errors):.6f}")

# The mixed mean should be higher than normal mean
# because anomalous windows are harder to reconstruct
if np.mean(mixed_errors) > np.mean(normal_errors):
    print("\n  mixed errors > normal errors — as expected!")
    print("  Anomalies are producing higher reconstruction errors.")
else:
    print("\n  Unexpected: mixed errors are not higher than normal.")


# ─────────────────────────────────────────────
# STEP 6: Save reconstruction error arrays
# ─────────────────────────────────────────────

print("STEP 6: Saving reconstruction error arrays ...")
np.save("normal_reconstruction_errors.npy", normal_errors)
np.save("mixed_reconstruction_errors.npy",  mixed_errors)

print("  Saved -> normal_reconstruction_errors.npy")
print("  Saved -> mixed_reconstruction_errors.npy")
# ─────────────────────────────────────────────
# Plot Reconstruction Error vs Sample Index
# ─────────────────────────────────────────────
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))

# Plot normal errors in blue
plt.plot(normal_errors, color='steelblue', linewidth=0.8,
         label='Normal Errors', alpha=0.7)

# Plot mixed errors in red
plt.plot(mixed_errors, color='tomato', linewidth=0.8,
         label='Mixed Errors', alpha=0.7)

plt.title('Reconstruction Error Plot')
plt.xlabel('Samples')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reconstruction_error.png', dpi=150)
plt.close()
print("\n  Plot saved -> reconstruction_error.png")

print("Done! Reconstruction errors computed and saved.")