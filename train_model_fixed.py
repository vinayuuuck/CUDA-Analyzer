#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pickle


def train_model():
    df = pd.read_csv("klaraptor_enriched_data.csv")
    print(f"Loaded {len(df)} samples\n")

    print("Cleaning data...")
    print(
        f"  Original range: {df['exec_time'].min():.6f}s - {df['exec_time'].max():.6f}s"
    )

    df_clean = df[df["exec_time"] <= 10.0].copy()
    df_clean = df_clean[df_clean["exec_time"] >= 0.000001].copy()

    print(f"  Removed {len(df) - len(df_clean)} outliers")
    print(
        f"  Cleaned range: {df_clean['exec_time'].min():.6f}s - {df_clean['exec_time'].max():.6f}s"
    )
    print(f"  Final dataset: {len(df_clean)} samples\n")

    feature_columns = [
        "N",
        "bx",
        "by",
        "bz",
        "total_threads",
        "dimensionality",
        "compute_intensity",
        "has_shared_memory",
        "global_reads",
        "global_writes",
        "arithmetic_ops",
        "memory_ops",
    ]

    X = df_clean[feature_columns]
    # Log transform the target for better scale handling
    y = np.log1p(df_clean["exec_time"])  # log(1 + exec_time)

    print(f"Features ({len(feature_columns)}):")
    for i, col in enumerate(feature_columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\nTarget: log(exec_time)")
    print(f"  Min:    {y.min():.4f}")
    print(f"  Max:    {y.max():.4f}")
    print(f"  Mean:   {y.mean():.4f}")
    print(f"  Median: {y.median():.4f}\n")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}\n")

    # Train Random Forest
    print("=" * 70)
    print("Training Random Forest (log-transformed target)...")
    print("=" * 70)

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    model.fit(X_train, y_train)
    print("\n✓ Training complete!\n")

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Also calculate R² on original scale
    y_train_orig = np.expm1(y_train)  # Convert back from log
    y_test_orig = np.expm1(y_test)
    y_pred_train_orig = np.expm1(y_pred_train)
    y_pred_test_orig = np.expm1(y_pred_test)

    test_r2_orig = r2_score(y_test_orig, y_pred_test_orig)
    test_mae_orig = mean_absolute_error(y_test_orig, y_pred_test_orig)

    print("=" * 70)
    print("MODEL PERFORMANCE (Log Scale)")
    print("=" * 70)
    print(f"Training MAE:    {train_mae:.4f}")
    print(f"Test MAE:        {test_mae:.4f}")
    print(f"Training R²:     {train_r2:.4f}")
    print(f"Test R²:         {test_r2:.4f}")
    print("=" * 70)
    print("MODEL PERFORMANCE (Original Scale - Seconds)")
    print("=" * 70)
    print(f"Test MAE:        {test_mae_orig:.6f}s")
    print(f"Test R²:         {test_r2_orig:.4f}")
    print("=" * 70)

    # Feature importance
    print("\nTop 10 Most Important Features:")
    print("-" * 70)
    importances = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    for idx, row in importances.head(10).iterrows():
        bar_length = int(row["importance"] * 50)
        bar = "█" * bar_length
        print(f"  {row['feature']:20s} {row['importance']:6.4f}  {bar}")

    # Save model
    model_file = "grid_block_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump({"model": model, "log_transform": True}, f)

    print(f"\n✓ Model saved to: {model_file}")

    with open("feature_names.pkl", "wb") as f:
        pickle.dump(feature_columns, f)

    print(f"✓ Feature names saved to: feature_names.pkl\n")

    # Performance assessment
    print("=" * 70)
    if test_r2 > 0.85:
        print("🎉 EXCELLENT! R² > 0.85 - Model is highly accurate!")
    elif test_r2 > 0.75:
        print("👍 GOOD! R² > 0.75 - Model performance is acceptable")
    elif test_r2 > 0.65:
        print("⚠️  FAIR. R² > 0.65 - Model could be improved")
    else:
        print("❌ POOR. R² < 0.65 - Model needs significant improvement")
    print("=" * 70)

    return model, test_r2


if __name__ == "__main__":
    model, test_r2 = train_model()
