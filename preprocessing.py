import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────
# SAFETY CHECK: Confirm working directory
# ─────────────────────────────────────────────

print("=" * 50)
print("Working directory:", os.getcwd())
print("Files here:", os.listdir("."))
print("=" * 50)


# ─────────────────────────────────────────────
# STEP 1: Load both CSV files
# ─────────────────────────────────────────────

print("\nSTEP 1: Loading CSV files ...")

normal_df = pd.read_csv("normal_telemetry.csv")
mixed_df  = pd.read_csv("mixed_telemetry.csv")

print(f"  Normal dataset loaded  -> {normal_df.shape[0]} rows, {normal_df.shape[1]} columns")
print(f"  Mixed  dataset loaded  -> {mixed_df.shape[0]} rows, {mixed_df.shape[1]} columns")


# ─────────────────────────────────────────────
# STEP 2: Separate features and labels
# ─────────────────────────────────────────────

print("\nSTEP 2: Separating features and labels ...")

# Columns that are NOT telemetry features
non_feature_cols = ["timestamp", "label", "anomaly_type"]

# Keep only telemetry sensor columns
feature_cols = [col for col in normal_df.columns if col not in non_feature_cols]

print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

# Extract features as NumPy arrays
normal_features = normal_df[feature_cols].values
mixed_features  = mixed_df[feature_cols].values

# Extract labels from mixed dataset
mixed_labels = mixed_df["label"].values

print(f"  Unique labels   : {list(np.unique(mixed_labels))}")
print(f"  Normal  count   : {int(np.sum(mixed_labels == 'normal'))}")
print(f"  Anomaly count   : {int(np.sum(mixed_labels == 'anomaly'))}")


# ─────────────────────────────────────────────
# STEP 3 & 4: Scale using MinMaxScaler
#   Fit ONLY on normal data
#   Transform both datasets with same scaler
# ─────────────────────────────────────────────

print("\nSTEP 3 & 4: Scaling data ...")

scaler = MinMaxScaler(feature_range=(0, 1))

# Fit ONLY on normal (clean) data
scaler.fit(normal_features)
print("  Scaler fitted on normal dataset only.")

# Transform both datasets
normal_scaled = scaler.transform(normal_features)
mixed_scaled  = scaler.transform(mixed_features)

print("  Both datasets transformed successfully.")


# ─────────────────────────────────────────────
# STEP 5: Print shapes
# ─────────────────────────────────────────────

print("\nSTEP 5: Shapes of processed arrays ...")
print(f"  normal_scaled shape : {normal_scaled.shape}")
print(f"  mixed_scaled  shape : {mixed_scaled.shape}")
print(f"  mixed_labels  shape : {mixed_labels.shape}")


# ─────────────────────────────────────────────
# STEP 6: Save to disk
# ─────────────────────────────────────────────

print("\nSTEP 6: Saving .npy files ...")

np.save("normal_scaled.npy", normal_scaled)
np.save("mixed_scaled.npy",  mixed_scaled)
np.save("mixed_labels.npy",  mixed_labels)

print("  Saved -> normal_scaled.npy")
print("  Saved -> mixed_scaled.npy")
print("  Saved -> mixed_labels.npy")

print("\n" + "=" * 50)
print("Preprocessing complete. Ready for model training.")
print("=" * 50)