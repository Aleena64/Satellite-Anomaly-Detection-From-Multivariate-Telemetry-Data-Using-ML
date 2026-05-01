import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
 
 
# ══════════════════════════════════════════════════════════
# STEP 1: Load Files
# ══════════════════════════════════════════════════════════
 
print("=" * 60)
print("  STEP 1: Loading Files")
print("=" * 60)
 
normal_errors = np.load("normal_reconstruction_errors.npy")
mixed_errors  = np.load("mixed_reconstruction_errors.npy")
mixed_labels  = np.load("mixed_window_labels.npy", allow_pickle=True)
 
# Convert string labels to integers if needed
# "normal" → 0,  "anomaly" → 1
if mixed_labels.dtype.kind in ('U', 'O'):
    mixed_labels_int = (mixed_labels == "anomaly").astype(int)
else:
    mixed_labels_int = mixed_labels.astype(int)
 
print(f"  normal_errors shape : {normal_errors.shape}")
print(f"  mixed_errors  shape : {mixed_errors.shape}")
print(f"  mixed_labels  shape : {mixed_labels_int.shape}")
print(f"  Normal  windows     : {np.sum(mixed_labels_int == 0)}")
print(f"  Anomaly windows     : {np.sum(mixed_labels_int == 1)}")
 
 
# ══════════════════════════════════════════════════════════
# STEP 2: Compute Threshold
#   threshold = mean + 3 × std
#   This is a standard statistical method — values beyond
#   3 standard deviations from the mean are considered
#   statistically unusual (anomalous).
# ══════════════════════════════════════════════════════════
 
print("\n" + "=" * 60)
print("  STEP 2: Computing Threshold")
print("=" * 60)
 
mean_normal = np.mean(normal_errors)
std_normal  = np.std(normal_errors)
threshold   = mean_normal + 3 * std_normal
 
print(f"  Mean of normal errors : {mean_normal:.6f}")
print(f"  Std  of normal errors : {std_normal:.6f}")
print(f"  Threshold (mean+3std) : {threshold:.6f}")
 
 
# ══════════════════════════════════════════════════════════
# STEP 3: Classify Anomalies
#   error > threshold → anomaly (1)
#   error ≤ threshold → normal  (0)
# ══════════════════════════════════════════════════════════
 
print("\n" + "=" * 60)
print("  STEP 3: Classifying Anomalies")
print("=" * 60)
 
predictions = (mixed_errors > threshold).astype(int)
 
print(f"  Predicted normal  : {np.sum(predictions == 0)}")
print(f"  Predicted anomaly : {np.sum(predictions == 1)}")
 
 
# ══════════════════════════════════════════════════════════
# STEP 4: Evaluation Metrics
# ══════════════════════════════════════════════════════════
 
print("\n" + "=" * 60)
print("  STEP 4: Evaluation Metrics")
print("=" * 60)
 
accuracy  = accuracy_score(mixed_labels_int, predictions)
precision = precision_score(mixed_labels_int, predictions, zero_division=0)
recall    = recall_score(mixed_labels_int, predictions, zero_division=0)
f1        = f1_score(mixed_labels_int, predictions, zero_division=0)
 
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
 
 
# ══════════════════════════════════════════════════════════
# STEP 5: Confusion Matrix
# ══════════════════════════════════════════════════════════
 
print("\n" + "=" * 60)
print("  STEP 5: Confusion Matrix")
print("=" * 60)
 
cm = confusion_matrix(mixed_labels_int, predictions)
tn, fp, fn, tp = cm.ravel()
 
print(f"\n  TN (True  Normal  ) : {tn}")
print(f"  FP (False Positive) : {fp}  ← normal flagged as anomaly")
print(f"  FN (False Negative) : {fn}  ← anomaly missed")
print(f"  TP (True  Anomaly ) : {tp}")
 
# Plot and save confusion matrix
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"]
)
disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
ax_cm.set_title(
    f"Confusion Matrix — Threshold Detection\n"
    f"Threshold = {threshold:.5f}  |  "
    f"Precision = {precision:.3f}  |  "
    f"Recall = {recall:.3f}",
    fontsize=10
)
plt.tight_layout()
plt.savefig("plot_threshold_confusion_matrix.png", dpi=150)
plt.close()
print("\n  Saved → plot_threshold_confusion_matrix.png")
 
 
# ══════════════════════════════════════════════════════════
# STEP 6a: Histogram Plot
#   normal errors vs mixed errors + threshold line
# ══════════════════════════════════════════════════════════
 
print("\n" + "=" * 60)
print("  STEP 6: Generating Visualizations")
print("=" * 60)
 
fig1, ax1 = plt.subplots(figsize=(11, 5))
 
ax1.hist(
    normal_errors,
    bins=120,
    alpha=0.65,
    color="steelblue",
    label="Normal Errors",
    edgecolor="none"
)
ax1.hist(
    mixed_errors,
    bins=120,
    alpha=0.55,
    color="tomato",
    label="Mixed Errors (Normal + Anomaly)",
    edgecolor="none"
)
 
# Threshold line
ax1.axvline(
    x=threshold,
    color="darkred",
    linewidth=2.0,
    linestyle="--",
    label=f"Threshold = {threshold:.5f}"
)
 
# Shade anomaly region
ax1.axvspan(
    threshold,
    mixed_errors.max() * 1.05,
    alpha=0.08,
    color="red",
    label="Anomaly Region"
)
 
ax1.set_title("Reconstruction Error Distribution — Normal vs Mixed",
              fontsize=13, fontweight="bold")
ax1.set_xlabel("Reconstruction Error (MSE)", fontsize=11)
ax1.set_ylabel("Number of Windows",          fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_threshold_histogram.png", dpi=150)
plt.close()
print("  Saved → plot_threshold_histogram.png")
 
 
# ══════════════════════════════════════════════════════════
# STEP 6b: Error vs Sample Index Plot
#   Mixed errors over time + threshold line + anomaly marks
# ══════════════════════════════════════════════════════════
 
sample_idx = np.arange(len(mixed_errors))
 
# Separate indices by classification outcome
true_normal_idx    = np.where((predictions == 0) & (mixed_labels_int == 0))[0]
true_anomaly_idx   = np.where((predictions == 1) & (mixed_labels_int == 1))[0]
false_positive_idx = np.where((predictions == 1) & (mixed_labels_int == 0))[0]
false_negative_idx = np.where((predictions == 0) & (mixed_labels_int == 1))[0]
 
fig2, ax2 = plt.subplots(figsize=(15, 5))
 
# Background error line
ax2.plot(
    sample_idx,
    mixed_errors,
    color="lightsteelblue",
    linewidth=0.6,
    alpha=0.8,
    zorder=1,
    label="Reconstruction Error"
)
 
# True normal (blue dots — small)
ax2.scatter(
    true_normal_idx,
    mixed_errors[true_normal_idx],
    s=3, color="steelblue",
    alpha=0.4, zorder=2,
    label="True Normal"
)
 
# True anomaly (green dots)
ax2.scatter(
    true_anomaly_idx,
    mixed_errors[true_anomaly_idx],
    s=15, color="green",
    alpha=0.9, zorder=4,
    label=f"True Anomaly Detected ({len(true_anomaly_idx)})"
)
 
# False positives (orange dots)
ax2.scatter(
    false_positive_idx,
    mixed_errors[false_positive_idx],
    s=15, color="orange",
    alpha=0.9, zorder=4,
    label=f"False Positive ({len(false_positive_idx)})"
)
 
# False negatives (red dots)
ax2.scatter(
    false_negative_idx,
    mixed_errors[false_negative_idx],
    s=15, color="red",
    alpha=0.9, zorder=4,
    label=f"Missed Anomaly ({len(false_negative_idx)})"
)
 
# Threshold line
ax2.axhline(
    y=threshold,
    color="darkred",
    linewidth=1.8,
    linestyle="--",
    zorder=5,
    label=f"Threshold = {threshold:.5f}"
)
 
# Shade above threshold
ax2.axhspan(
    threshold,
    mixed_errors.max() * 1.1,
    alpha=0.06,
    color="red"
)
 
ax2.set_title("Reconstruction Error vs Sample Index — Anomaly Detection",
              fontsize=13, fontweight="bold")
ax2.set_xlabel("Sample Index (Window Number)", fontsize=11)
ax2.set_ylabel("Reconstruction Error (MSE)",   fontsize=11)
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("plot_threshold_detection.png", dpi=150)
plt.close()
print("  Saved → plot_threshold_detection.png")