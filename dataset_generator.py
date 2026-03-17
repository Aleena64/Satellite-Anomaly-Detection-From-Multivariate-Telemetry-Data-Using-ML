import numpy as np
import pandas as pd

# Reproducibility 
RNG = np.random.default_rng(42)
N = 10_000          # time steps
DT = 10             # seconds between samples (10 s → ~28 h of data)

# Helpers 

def timestamp_series(n, dt=DT):
    """Return ISO-8601 UTC timestamps starting at a fixed epoch."""
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    return pd.date_range(start, periods=n, freq=f"{dt}s")


def smooth_signal(n, mean, amplitude, period_steps, noise_std, rng):
    """Sinusoidal baseline + small Gaussian noise — mimics orbital periodicity."""
    t = np.arange(n)
    base = mean + amplitude * np.sin(2 * np.pi * t / period_steps)
    return base + rng.normal(0, noise_std, n)


def clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)


#  Orbital period 
# Typical LEO ~90 min orbit → 90×60/DT = 540 steps per orbit
ORBIT_STEPS = 540

# ==============================================================================
# DATASET 1 — NORMAL TELEMETRY
# ==============================================================================

def generate_normal(n=N, rng=RNG):
    ts = timestamp_series(n)

    # Attitude (degrees) — small sinusoidal excursions ± noise
    roll  = smooth_signal(n, 0, 3.0, ORBIT_STEPS,     0.20, rng)
    pitch = smooth_signal(n, 0, 2.5, ORBIT_STEPS*1.1, 0.15, rng)
    yaw   = smooth_signal(n, 0, 1.8, ORBIT_STEPS*0.9, 0.25, rng)

    # Battery voltage (V) — nominal 28–32 V, slight orbital variation
    batt_v = smooth_signal(n, 30.0, 1.5, ORBIT_STEPS, 0.10, rng)
    batt_v = clamp(batt_v, 28.0, 32.0)

    # Battery current (A) — peaks during eclipse (payload draw)
    batt_i = smooth_signal(n, 2.5, 1.2, ORBIT_STEPS, 0.15, rng)
    batt_i = clamp(batt_i, 0.0, 5.0)

    # Battery temperature (°C) — slow thermal cycling
    batt_t = smooth_signal(n, 25.0, 8.0, ORBIT_STEPS, 0.30, rng)
    batt_t = clamp(batt_t, 10.0, 40.0)

    # Payload temperature (°C) — larger excursion, slower drift
    payload_t = smooth_signal(n, 25.0, 12.0, ORBIT_STEPS*2, 0.40, rng)
    payload_t = clamp(payload_t, 0.0, 50.0)

    # Reaction wheel speed (rpm) — slow variation around working point
    rw_speed = smooth_signal(n, 3000.0, 800.0, ORBIT_STEPS*3, 20.0, rng)
    rw_speed = clamp(rw_speed, 1000.0, 5000.0)

    df = pd.DataFrame({
        "timestamp":           ts,
        "roll_deg":            np.round(roll,  4),
        "pitch_deg":           np.round(pitch, 4),
        "yaw_deg":             np.round(yaw,   4),
        "battery_voltage_V":   np.round(batt_v,  4),
        "battery_current_A":   np.round(batt_i,  4),
        "battery_temp_C":      np.round(batt_t,  4),
        "payload_temp_C":      np.round(payload_t, 4),
        "reaction_wheel_rpm":  np.round(rw_speed,  2),
    })
    return df


# ==============================================================================
# DATASET 2 — MIXED TELEMETRY (with injected anomalies)
# ==============================================================================

def inject_anomalies(df_in, rng):
    df = df_in.copy()
    n = len(df)
    label = np.array(["normal"] * n, dtype=object)
    anomaly_type = np.array(["none"] * n, dtype=object)

    #  1. Spike in battery voltage 
    # Three random short spikes (3–8 steps each)
    spike_centers = [1200, 4500, 7800]
    for center in spike_centers:
        width = rng.integers(3, 9)
        idx = slice(center, center + width)
        df.loc[idx, "battery_voltage_V"] += rng.uniform(3.5, 5.5)
        df["battery_voltage_V"] = clamp(df["battery_voltage_V"].values, 0, 45)
        label[center: center + width] = "anomaly"
        anomaly_type[center: center + width] = "voltage_spike"

    #  2. Gradual drift in reaction wheel speed 
    # Linear ramp from step 2000 to 2500 (500 steps ≈ 1.4 h)
    drift_start, drift_end = 2000, 2500
    drift_len = drift_end - drift_start
    drift_ramp = np.linspace(0, 1800, drift_len)   # drifts +1800 rpm
    df.loc[drift_start:drift_end - 1, "reaction_wheel_rpm"] += drift_ramp
    df["reaction_wheel_rpm"] = clamp(df["reaction_wheel_rpm"].values, 0, 9000)
    label[drift_start:drift_end] = "anomaly"
    anomaly_type[drift_start:drift_end] = "rw_drift"

    # 3. Bias shift in temperature sensors 
    # Step offset on both temp channels from step 5500 to 6200
    bias_start, bias_end = 5500, 6200
    df.loc[bias_start:bias_end - 1, "battery_temp_C"] += 12.0
    df.loc[bias_start:bias_end - 1, "payload_temp_C"] += 15.0
    df["battery_temp_C"] = clamp(df["battery_temp_C"].values, -20, 80)
    df["payload_temp_C"] = clamp(df["payload_temp_C"].values, -20, 90)
    label[bias_start:bias_end] = "anomaly"
    anomaly_type[bias_start:bias_end] = "temp_bias_shift"

    # 4. Dropouts (zero values) 
    # Three dropout bursts of 10–20 steps affecting multiple channels
    dropout_centers = [3100, 6700, 8900]
    dropout_cols = ["battery_voltage_V", "battery_current_A",
                    "reaction_wheel_rpm"]
    for center in dropout_centers:
        width = rng.integers(10, 21)
        idx = slice(center, center + width)
        df.loc[idx, dropout_cols] = 0.0
        label[center: center + width] = "anomaly"
        anomaly_type[center: center + width] = "dropout"

    # 5. Noise burst 
    # High-frequency noise on attitude channels, two bursts
    noise_centers = [3800, 7300]
    attitude_cols = ["roll_deg", "pitch_deg", "yaw_deg"]
    for center in noise_centers:
        width = rng.integers(30, 61)
        idx = np.arange(center, min(center + width, n))
        for col in attitude_cols:
            df.loc[idx, col] += rng.normal(0, 4.0, len(idx))
        label[center: center + len(idx)] = "anomaly"
        anomaly_type[center: center + len(idx)] = "noise_burst"

    # 6. Stuck sensor intervals 
    # Sensor reads a frozen constant value for 50–100 steps
    stuck_configs = [
        (4200, "payload_temp_C"),
        (8200, "battery_current_A"),
        (9100, "roll_deg"),
    ]
    for center, col in stuck_configs:
        width = rng.integers(50, 101)
        stuck_val = float(df.loc[center - 1, col])   # freeze at last good value
        idx = slice(center, min(center + width, n))
        df.loc[idx, col] = stuck_val
        label[center: min(center + width, n)] = "anomaly"
        anomaly_type[center: min(center + width, n)] = "stuck_sensor"

    df["label"] = label
    df["anomaly_type"] = anomaly_type
    return df


def generate_mixed(n=N, rng=RNG):
    # Use a separate RNG seed so the base signal matches normal dataset
    base_rng = np.random.default_rng(42)
    df_base = generate_normal(n, base_rng)

    # Introduce slightly more base noise to make the dataset harder
    noise_rng = np.random.default_rng(7)
    for col in ["roll_deg", "pitch_deg", "yaw_deg",
                "battery_voltage_V", "battery_current_A",
                "battery_temp_C", "payload_temp_C", "reaction_wheel_rpm"]:
        df_base[col] += noise_rng.normal(0, 0.05, n)

    df_mixed = inject_anomalies(df_base, rng=np.random.default_rng(99))
    return df_mixed


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("Generating normal telemetry dataset")
    df_normal = generate_normal()
    df_normal.to_csv("normal_telemetry.csv", index=False)
    print(f"  Saved normal_telemetry.csv  ({len(df_normal):,} rows)")

    print("Generating mixed telemetry dataset")
    df_mixed = generate_mixed()
    df_mixed.to_csv("mixed_telemetry.csv", index=False)
    n_anom = (df_mixed["label"] == "anomaly").sum()
    print(f"  Saved mixed_telemetry.csv   ({len(df_mixed):,} rows, "
          f"{n_anom:,} anomaly steps → "
          f"{n_anom / len(df_mixed) * 100:.1f}%)")

    print("\nAnomaly type breakdown:")
    print(df_mixed[df_mixed["label"] == "anomaly"]["anomaly_type"]
          .value_counts().to_string())
    print("\nDone")