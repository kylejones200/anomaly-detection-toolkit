"""Complete Predictive Maintenance Workflow Example.

This example demonstrates the full predictive maintenance workflow including:
- Loading PM data with asset_id and cycle columns
- Feature engineering (RUL calculation, rolling statistics, labels)
- Training RUL forecasting models
- Making predictions and evaluating performance
- Integration with anomaly detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from anomaly_detection_toolkit.predictive_maintenance import RULEstimator, prepare_pm_features
from anomaly_detection_toolkit.statistical import ZScoreDetector


def generate_synthetic_pm_data(
    n_assets: int = 10, min_cycles: int = 100, max_cycles: int = 300, n_sensors: int = 5
) -> pd.DataFrame:
    """Generate synthetic predictive maintenance data.

    Parameters
    ----------
    n_assets : int, default=10
        Number of assets/equipment.
    min_cycles : int, default=100
        Minimum cycles until failure.
    max_cycles : int, default=300
        Maximum cycles until failure.
    n_sensors : int, default=5
        Number of sensor/feature columns.

    Returns
    -------
    df : DataFrame
        Synthetic PM data with asset_id, cycle, and sensor columns.
    """
    np.random.seed(42)
    records = []

    for asset_id in range(1, n_assets + 1):
        # Random failure cycle for this asset
        failure_cycle = np.random.randint(min_cycles, max_cycles + 1)

        for cycle in range(1, failure_cycle + 1):
            # Simulate degradation: sensors increase over time
            degradation_factor = cycle / failure_cycle

            record = {"asset_id": f"ASSET_{asset_id:03d}", "cycle": cycle}

            # Generate sensor readings with degradation trend
            for sensor_id in range(1, n_sensors + 1):
                base_value = np.random.uniform(20, 50)
                trend = degradation_factor * np.random.uniform(10, 30)
                noise = np.random.randn() * 2
                record[f"sensor_{sensor_id}"] = base_value + trend + noise

            records.append(record)

    return pd.DataFrame(records)


def forecast_rul_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "RUL",
    model_type: str = "linear",
) -> tuple:
    """
    Forecast RUL using regression models.

    Parameters
    ----------
    train_df : DataFrame
        Training data with features and RUL.
    test_df : DataFrame
        Test data with features.
    feature_cols : list of str
        Feature column names.
    target_col : str, default='RUL'
        Target column name.
    model_type : str, default='linear'
        Model type: 'linear' or 'random_forest'.

    Returns
    -------
    tuple
        (model, predictions, metrics_dict)
    """
    # Prepare features
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col] if target_col in test_df.columns else None

    # Fit model
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics if test RUL available
    metrics = {}
    if y_test is not None:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2,
            "MSE": mse,
        }

    return model, y_pred, metrics


def main():
    """Run complete predictive maintenance workflow example."""
    print("=" * 70)
    print("PREDICTIVE MAINTENANCE WORKFLOW EXAMPLE")
    print("=" * 70)

    # Step 1: Generate or load data
    print("\n1. Generating synthetic predictive maintenance data...")
    df = generate_synthetic_pm_data(n_assets=10, min_cycles=100, max_cycles=300, n_sensors=5)
    print(f"   Generated {len(df)} records for {df['asset_id'].nunique()} assets")
    print(f"   Columns: {list(df.columns)}")

    # Step 2: Feature engineering
    print("\n2. Engineering predictive maintenance features...")
    print("   - Calculating RUL...")
    print("   - Adding health status labels...")
    print("   - Adding rolling statistics...")

    df = prepare_pm_features(
        df=df,
        asset_id_col="asset_id",
        cycle_col="cycle",
        feature_cols=None,  # Auto-detect
        calculate_rul_flag=True,
        add_labels=True,
        add_rolling_stats=True,
        add_degradation_rates=False,
        rolling_window=5,
        warning_threshold=30,
        critical_threshold=15,
    )

    print(f"   Features after engineering: {len(df.columns)} columns")
    print(f"   Sample columns: {list(df.columns[:10])}...")

    # Step 3: Split data by asset
    print("\n3. Splitting data into train/test sets...")
    asset_ids = df["asset_id"].unique()
    n_train_assets = int(len(asset_ids) * 0.8)
    train_assets = asset_ids[:n_train_assets]
    test_assets = asset_ids[n_train_assets:]

    train_df = df[df["asset_id"].isin(train_assets)].copy()
    test_df = df[df["asset_id"].isin(test_assets)].copy()

    print(f"   Train: {len(train_df)} records from {len(train_assets)} assets")
    print(f"   Test:  {len(test_df)} records from {len(test_assets)} assets")

    # Step 4: Prepare feature columns
    exclude_cols = [
        "asset_id",
        "cycle",
        "RUL",
        "health_status",
        "binary_label",
        "multi_class_label",
        "max_cycle",
    ]
    all_feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\n4. Using {len(all_feature_cols)} features for modeling")

    # Step 5: Train RUL forecasting model
    print("\n5. Training RUL forecasting model...")
    model, predictions, metrics = forecast_rul_regression(
        train_df=train_df,
        test_df=test_df,
        feature_cols=all_feature_cols,
        target_col="RUL",
        model_type="random_forest",
    )

    # Add predictions to test dataframe
    test_df = test_df.copy()
    test_df["RUL_predicted"] = predictions

    # Print metrics
    if metrics:
        print("\n   Test Set Performance:")
        for metric, value in metrics.items():
            print(f"     {metric}: {value:.4f}")

    # Step 6: Health status distribution
    print("\n6. Health Status Analysis:")
    if "health_status" in train_df.columns:
        health_counts = train_df["health_status"].value_counts()
        print("   Training set health status:")
        for status, count in health_counts.items():
            print(f"     {status}: {count}")

    # Step 7: Integration with anomaly detection
    print("\n7. Integrating with anomaly detection...")
    # Use sensor_1 as example for anomaly detection
    sensor_col = "sensor_1"
    if sensor_col in train_df.columns:
        detector = ZScoreDetector(n_std=2.5)
        detector.fit(train_df[sensor_col].values.reshape(-1, 1))

        # Detect anomalies in test set
        test_anomalies = detector.predict(test_df[sensor_col].values.reshape(-1, 1))
        n_anomalies = (test_anomalies == -1).sum()
        print(f"   Detected {n_anomalies} anomalies in test set sensor readings")

    # Step 8: Sample predictions
    print("\n8. Sample Predictions:")
    sample_assets = test_assets[:3]
    for asset_id in sample_assets:
        asset_data = test_df[test_df["asset_id"] == asset_id].sort_values("cycle")
        if len(asset_data) > 0:
            last_row = asset_data.iloc[-1]
            print(f"\n   {asset_id}:")
            print(f"     Cycle: {last_row['cycle']}")
            if "RUL" in last_row:
                print(f"     Actual RUL: {last_row['RUL']:.1f}")
            print(f"     Predicted RUL: {last_row['RUL_predicted']:.1f}")
            if "health_status" in last_row:
                print(f"     Health Status: {last_row['health_status']}")

    # Step 9: Using RULEstimator class
    print("\n9. Using RULEstimator class for RUL prediction...")
    rul_estimator = RULEstimator(method="regression", n_estimators=50, random_state=42)

    # Prepare features for estimator
    X_train = train_df[all_feature_cols].fillna(0)
    y_train = train_df["RUL"]
    X_test = test_df[all_feature_cols].fillna(0)

    # Fit and predict
    rul_estimator.fit(X_train, y_train)
    rul_predictions = rul_estimator.predict(X_test)

    # Compare with actual RUL if available
    if "RUL" in test_df.columns:
        mae_estimator = mean_absolute_error(test_df["RUL"], rul_predictions)
        print(f"   RULEstimator MAE: {mae_estimator:.4f}")

    print("\n" + "=" * 70)
    print("Workflow completed successfully!")
    print("=" * 70)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "model": model,
        "metrics": metrics,
        "rul_estimator": rul_estimator,
    }


if __name__ == "__main__":
    results = main()
