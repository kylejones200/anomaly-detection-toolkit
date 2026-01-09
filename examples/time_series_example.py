"""Time series anomaly detection example."""

import numpy as np
import pandas as pd

from anomaly_detection_toolkit import SeasonalBaselineDetector, WaveletDenoiser, WaveletDetector


def main():
    """Run time series anomaly detection example."""
    print("=" * 70)
    print("TIME SERIES ANOMALY DETECTION EXAMPLE")
    print("=" * 70)

    # Generate synthetic time series with anomalies
    np.random.seed(42)
    n_days = 365
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    # Base signal with trend and seasonality
    trend = np.linspace(50, 60, n_days)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)  # Annual
    weekly = 5 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # Weekly
    noise = np.random.randn(n_days) * 2

    values = trend + seasonality + weekly + noise

    # Inject anomalies
    anomaly_indices = [50, 100, 150, 200, 250]
    for idx in anomaly_indices:
        values[idx : idx + 3] += 15  # 3-day anomaly periods

    ts_df = pd.DataFrame({"date": dates, "value": values})

    print(f"\nGenerated {n_days} days of time series data")
    print(f"Injected {len(anomaly_indices)} anomaly periods at indices: {anomaly_indices}\n")

    # Method 1: Wavelet-based detection
    print("1. Wavelet-Based Anomaly Detection:")
    print("-" * 70)
    wavelet_detector = WaveletDetector(
        wavelet="db4", threshold_factor=2.5, level=5, random_state=42
    )
    wavelet_detector.fit(values)
    wavelet_predictions, wavelet_scores = wavelet_detector.fit_predict(values)
    wavelet_anomalies = np.where(wavelet_predictions == -1)[0]
    print(f"   Detected {len(wavelet_anomalies)} anomaly periods")
    print(f"   Anomaly indices: {wavelet_anomalies[:15].tolist()}...")

    # Method 2: Seasonal baseline detection
    print("\n2. Seasonal Baseline Detection (Weekly Seasonality):")
    print("-" * 70)
    seasonal_detector = SeasonalBaselineDetector(seasonality="week", threshold_sigma=2.5)
    seasonal_detector.fit(ts_df, date_col="date", value_col="value")
    seasonal_predictions = seasonal_detector.predict(ts_df, date_col="date", value_col="value")
    seasonal_anomalies = np.where(seasonal_predictions == -1)[0]
    print(f"   Detected {len(seasonal_anomalies)} anomaly days")
    print(f"   Anomaly indices: {seasonal_anomalies[:15].tolist()}...")

    # Method 3: Wavelet denoising
    print("\n3. Wavelet Denoising:")
    print("-" * 70)
    denoiser = WaveletDenoiser(wavelet="db4", threshold_mode="soft", level=5)
    denoised = denoiser.denoise(values)
    noise_removed = values - denoised
    print(f"   Original signal std: {np.std(values):.2f}")
    print(f"   Denoised signal std: {np.std(denoised):.2f}")
    print(f"   Removed noise std: {np.std(noise_removed):.2f}")
    print(f"   SNR improvement: {np.std(values) / np.std(noise_removed):.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"True anomalies:        {len(anomaly_indices)} periods at indices {anomaly_indices}")
    print(f"Wavelet detected:      {len(wavelet_anomalies)} anomaly periods")
    print(f"Seasonal detected:     {len(seasonal_anomalies)} anomaly days")

    # Calculate overlap with true anomalies (within ±2 days)
    true_anomaly_days = set()
    for idx in anomaly_indices:
        true_anomaly_days.update(range(idx - 2, idx + 5))  # Include surrounding days

    wavelet_set = set(wavelet_anomalies)
    seasonal_set = set(seasonal_anomalies)

    if wavelet_set:
        wavelet_overlap = len(true_anomaly_days & wavelet_set)
        wavelet_precision = wavelet_overlap / len(wavelet_set) * 100
        print(
            f"\nWavelet precision (overlap with true anomalies): "
            f"{wavelet_overlap}/{len(wavelet_set)} ({wavelet_precision:.1f}%)"
        )

    if seasonal_set:
        seasonal_overlap = len(true_anomaly_days & seasonal_set)
        seasonal_precision = seasonal_overlap / len(seasonal_set) * 100
        print(
            f"Seasonal precision (overlap with true anomalies): "
            f"{seasonal_overlap}/{len(seasonal_set)} ({seasonal_precision:.1f}%)"
        )


if __name__ == "__main__":
    main()
