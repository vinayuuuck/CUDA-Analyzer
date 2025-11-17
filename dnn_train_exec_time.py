#!/usr/bin/env python3
"""
Neural Network for CUDA Execution Time Prediction
Predicts exec_time given (kernel, N, block_x, block_y) → enables optimization

This replaces KLARAPTOR's mathematical model with a learned model.
At runtime: Try all block configs, predict exec_time for each, pick minimum.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class KernelDataset(Dataset):
    """PyTorch dataset for kernel execution time prediction"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ExecTimePredictor(nn.Module):
    """
    Neural network that predicts execution time

    Input: kernel features + block configuration
    Output: log(exec_time) in ms
    """

    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout=0.3):
        super(ExecTimePredictor, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output: single value (log exec_time)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ExecTimePredictorTrainer:
    """Handles training, evaluation, and prediction"""

    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.history = {"train_loss": [], "val_loss": [], "epoch": []}

    def train_epoch(self, train_loader, criterion, optimizer, use_weighted_loss=True):
        """Train for one epoch with optional weighted loss"""
        self.model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_X)

            if use_weighted_loss:
                # Weighted MSE: proportional weighting for better real-space accuracy
                # Convert log predictions to real space for better weighting
                # Penalize relative errors equally across all time scales
                weights = 1.0 + torch.abs(batch_y)  # Higher weight for larger log(time)
                loss = ((outputs - batch_y) ** 2 * weights).mean()
            else:
                loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader, criterion):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader,
        val_loader,
        epochs=200,
        lr=0.001,
        patience=25,
        use_weighted_loss=True,
        verbose=True,
    ):
        """Full training loop with early stopping and weighted loss"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=verbose
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        if verbose:
            print("Training Execution Time Predictor...")
            print("  Weighted loss:", "enabled" if use_weighted_loss else "disabled")
            print("=" * 70)
            print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'LR':<15}")
            print("-" * 70)

        for epoch in range(epochs):
            train_loss = self.train_epoch(
                train_loader, criterion, optimizer, use_weighted_loss
            )
            val_loss = self.validate(val_loader, criterion)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"{epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {current_lr:<15.6e}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if verbose:
            print("=" * 70)
            print(f"Best validation loss: {best_val_loss:.6f}")

        return self.history

    def predict(self, X):
        """Predict execution time for given inputs"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()

        return predictions

    def save(self, model_path, scaler_path, metadata):
        """Save model and metadata"""
        torch.save(self.model.state_dict(), model_path)

        with open(scaler_path, "wb") as f:
            pickle.dump(
                {
                    "scaler": metadata["scaler"],
                    "features": metadata["features"],
                    "kernel_encoder": metadata.get("kernel_encoder"),
                    "history": self.history,
                    "use_log_target": metadata.get("use_log_target", True),
                    "training_bounds": metadata.get("training_bounds"),  # NEW
                    "N_range": metadata.get("N_range"),  # NEW
                },
                f,
            )

        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Metadata saved to: {scaler_path}")


def prepare_data(csv_file, use_kernel_names=True, use_log_target=True):
    """
    Prepare data for execution time prediction

    Key difference from block prediction:
    - Use ALL samples (not just optimal)
    - Target is exec_time (not block dims)
    - Features include block_x, block_y as inputs
    """
    df = pd.read_csv(csv_file)

    # Handle column naming
    column_mapping = {
        "kernel": "kernel_name",
        "bx": "block_x",
        "by": "block_y",
        "bz": "block_z",
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    print(f"Loaded data: {len(df):,} rows")
    print(f"Kernels: {df['kernel_name'].nunique()}")
    print(f"Data sizes: {sorted(df['N'].unique())}")

    if "gpu" in df.columns:
        print(f"GPUs: {df['gpu'].nunique()}")

    # Define feature columns (now including block config!)
    numeric_features = [
        "N",
        "log_N",
        "N_squared",
        "block_x",
        "block_y",
        "block_z",
        "dimensionality",
        "compute_intensity",
        "has_shared_memory",
        "global_reads",
        "global_writes",
        "arithmetic_ops",
        "memory_ops",
        "control_flow_ops",
        "loop_ops",
        "uses_syncthreads",
        "estimated_flops",
        "estimated_memory_bytes",
        "uses_threadIdx_x",
        "uses_threadIdx_y",
        "uses_threadIdx_z",
        "uses_blockIdx_x",
        "uses_blockIdx_y",
        "uses_blockIdx_z",
        "uses_blockDim_x",
        "uses_blockDim_y",
        "uses_blockDim_z",
    ]

    # Add N-based scaling features
    df["log_N"] = np.log(df["N"])
    df["N_squared"] = df["N"] ** 2

    # Add kernel name encoding
    kernel_encoder = None
    if use_kernel_names:
        kernel_encoder = LabelEncoder()
        df["kernel_encoded"] = kernel_encoder.fit_transform(df["kernel_name"])
        numeric_features.insert(0, "kernel_encoded")
        print(f"✓ Encoded {len(kernel_encoder.classes_)} unique kernels")

    # Extract features
    X = df[numeric_features].values

    # Target: execution time (use log for better distribution)
    if use_log_target:
        y = np.log(df["exec_time"].values)
        print(f"✓ Using log(exec_time) as target")
    else:
        y = df["exec_time"].values

    print(f"\nFeature matrix: {X.shape}")
    print(f"Target vector: {y.shape}")
    print(f"Features: {numeric_features}")
    print(
        f"Exec time range: {df['exec_time'].min():.6f} - {df['exec_time'].max():.6f} ms"
    )

    return X, y, numeric_features, kernel_encoder, df


def evaluate_model(trainer, test_loader, scaler, features, use_log_target=True):
    """Comprehensive evaluation"""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Get predictions
    all_predictions = []
    all_actuals = []

    trainer.model.eval()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(trainer.device)
            outputs = trainer.model(batch_X).cpu().numpy().flatten()
            all_predictions.append(outputs)
            all_actuals.append(batch_y.numpy().flatten())

    predictions = np.concatenate(all_predictions)
    actuals = np.concatenate(all_actuals)

    # Convert back from log if needed
    if use_log_target:
        predictions_real = np.exp(predictions)
        actuals_real = np.exp(actuals)
    else:
        predictions_real = predictions
        actuals_real = actuals

    # Calculate metrics
    metrics = {}

    # In log space
    metrics["mse_log"] = mean_squared_error(actuals, predictions)
    metrics["mae_log"] = mean_absolute_error(actuals, predictions)
    metrics["r2_log"] = r2_score(actuals, predictions)

    # In real space
    metrics["mse_real"] = mean_squared_error(actuals_real, predictions_real)
    metrics["mae_real"] = mean_absolute_error(actuals_real, predictions_real)
    metrics["r2_real"] = r2_score(actuals_real, predictions_real)

    # Percentage error
    mape = np.mean(np.abs((actuals_real - predictions_real) / actuals_real)) * 100
    metrics["mape"] = mape

    # Within X% accuracy
    pct_errors = np.abs((actuals_real - predictions_real) / actuals_real) * 100
    metrics["within_10pct"] = (pct_errors < 10).mean() * 100
    metrics["within_20pct"] = (pct_errors < 20).mean() * 100
    metrics["within_30pct"] = (pct_errors < 30).mean() * 100

    # Print results
    print(f"\nMetrics (log space):")
    print(f"  R² Score: {metrics['r2_log']:.4f}")
    print(f"  MAE: {metrics['mae_log']:.4f}")

    print(f"\nMetrics (real time):")
    print(f"  R² Score: {metrics['r2_real']:.4f}")
    print(f"  MAE: {metrics['mae_real']:.6f} ms")
    print(f"  MAPE: {metrics['mape']:.2f}%")

    print(f"\nAccuracy:")
    print(f"  Within 10%: {metrics['within_10pct']:.2f}%")
    print(f"  Within 20%: {metrics['within_20pct']:.2f}%")
    print(f"  Within 30%: {metrics['within_30pct']:.2f}%")

    # Show examples
    print(f"\nSample Predictions (first 15):")
    print("-" * 70)
    print(f"{'Predicted (ms)':<20} {'Actual (ms)':<20} {'Error %':<15}")
    print("-" * 70)

    for i in range(min(15, len(predictions_real))):
        pred = predictions_real[i]
        actual = actuals_real[i]
        error_pct = abs(pred - actual) / actual * 100

        print(f"{pred:<20.6f} {actual:<20.6f} {error_pct:<15.2f}")

    return metrics


def find_optimal_config(model_path, scaler_path, kernel_info, candidate_configs=None):
    """
    Find optimal block config by trying all candidates and predicting exec_time

    This is the KEY function that replaces KLARAPTOR's rational program!

    Args:
        model_path: Path to trained model
        scaler_path: Path to scaler/metadata
        kernel_info: Dict with kernel characteristics (no block config)
        candidate_configs: List of (block_x, block_y) tuples to try

    Returns:
        (best_block_x, best_block_y, predicted_time)
    """
    # Load metadata
    with open(scaler_path, "rb") as f:
        metadata = pickle.load(f)

    scaler = metadata["scaler"]
    features = metadata["features"]
    kernel_encoder = metadata.get("kernel_encoder")
    use_log_target = metadata.get("use_log_target", True)
    training_bounds = metadata.get("training_bounds")
    N_range = metadata.get("N_range")

    # Warn if extrapolating
    if N_range and kernel_info.get("N"):
        N = kernel_info["N"]
        if N < N_range[0] or N > N_range[1]:
            print(
                f"⚠ WARNING: N={N} is outside training range [{N_range[0]}, {N_range[1]}]"
            )
            print(f"  Predictions may be unreliable (extrapolation)")

    # Load model
    input_dim = len(features)
    model = ExecTimePredictor(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate candidate configs if not provided
    if candidate_configs is None:
        dim = kernel_info.get("dimensionality", 1)

        if dim == 1:
            # 1D configs
            candidate_configs = [
                (32, 1),
                (64, 1),
                (128, 1),
                (256, 1),
                (512, 1),
                (1024, 1),
            ]
        elif dim == 2:
            # 2D configs
            candidate_configs = [
                (8, 4),
                (8, 8),
                (8, 16),
                (8, 32),
                (16, 4),
                (16, 8),
                (16, 16),
                (16, 32),
                (32, 4),
                (32, 8),
                (32, 16),
                (32, 32),
                (64, 4),
                (64, 8),
                (64, 16),
                (128, 4),
                (128, 8),
                (256, 4),
            ]
        else:  # 3D
            candidate_configs = [(8, 8, 4), (8, 8, 8), (16, 8, 4), (16, 8, 8)]
            # Convert to 2D for compatibility
            candidate_configs = [(x * z, y) for x, y, z in candidate_configs]

    # Predict exec_time for each config
    predictions = []

    for block_x, block_y in candidate_configs:
        # Prepare features
        feature_values = []

        for feat in features:
            if feat == "kernel_encoded" and kernel_encoder:
                try:
                    encoded = kernel_encoder.transform([kernel_info["kernel_name"]])[0]
                except:
                    encoded = 0
                feature_values.append(encoded)
            elif feat == "block_x":
                feature_values.append(block_x)
            elif feat == "block_y":
                feature_values.append(block_y)
            elif feat == "log_N":
                feature_values.append(np.log(kernel_info.get("N", 1024)))
            elif feat == "N_squared":
                N = kernel_info.get("N", 1024)
                feature_values.append(N**2)
            else:
                feature_values.append(kernel_info.get(feat, 0))

        X = np.array([feature_values])
        X_scaled = scaler.transform(X)

        # Predict
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            pred_log = model(X_tensor).item()

        # Convert from log
        pred_time = np.exp(pred_log) if use_log_target else pred_log

        # Clip to reasonable bounds (3x training range)
        if training_bounds:
            min_bound = training_bounds[0] / 3.0
            max_bound = training_bounds[1] * 3.0
            pred_time = np.clip(pred_time, min_bound, max_bound)

        predictions.append(
            {"block_x": block_x, "block_y": block_y, "predicted_time": pred_time}
        )

    # Find best config
    best = min(predictions, key=lambda x: x["predicted_time"])

    return best["block_x"], best["block_y"], best["predicted_time"], predictions


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train NN to predict CUDA execution time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This trains a model to predict exec_time given (kernel, N, block_x, block_y).
At runtime, you can try all block configs and pick the one with minimum predicted time.

This REPLACES KLARAPTOR's mathematical rational program with a learned model!

Examples:
  # Train on all data
  python3 train_nn_exec_time_predictor.py klaraptor_enriched_data.csv
  
  # Find optimal config for a kernel
  python3 train_nn_exec_time_predictor.py --optimize \\
      --kernel Convolution2D_kernel --N 4096 --dim 2 --compute 8.3
        """,
    )

    parser.add_argument("csv_file", nargs="?", help="CSV with ALL configs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64, 32])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--no-kernel-names", action="store_true")
    parser.add_argument(
        "--no-log", action="store_true", help="Do not use log transform on exec_time"
    )
    parser.add_argument(
        "--size-filter",
        choices=["small", "medium", "large", "all"],
        default="all",
        help="Train on specific problem sizes (small: N<=256, medium: N<=2048, large: N>2048)",
    )

    # Optimization mode
    parser.add_argument(
        "--optimize", action="store_true", help="Find optimal block config"
    )
    parser.add_argument("--model", default="output/exec_time_predictor_nn.pth")
    parser.add_argument("--scaler", default="output/exec_time_predictor_scaler.pkl")
    parser.add_argument("--kernel", help="Kernel name")
    parser.add_argument("--N", type=int, help="Data size")
    parser.add_argument("--dim", type=int, help="Dimensionality")
    parser.add_argument("--compute", type=float, help="Compute intensity")

    args = parser.parse_args()

    # Optimization mode
    if args.optimize:
        if not all([args.kernel, args.N, args.dim]):
            parser.error("Optimization requires: --kernel, --N, --dim")

        kernel_info = {
            "kernel_name": args.kernel,
            "N": args.N,
            "dimensionality": args.dim,
            "compute_intensity": args.compute or 10.0,
            "has_shared_memory": 0,
            "global_reads": 10,
            "global_writes": 5,
            "arithmetic_ops": 100,
            "memory_ops": 15,
        }

        print(f"Finding optimal config for {args.kernel} with N={args.N}...")
        print()

        bx, by, pred_time, all_preds = find_optimal_config(
            args.model, args.scaler, kernel_info
        )

        print(f"Optimal configuration:")
        print(f"  block_dims: ({bx}, {by}, 1)")
        print(f"  predicted_time: {pred_time:.6f} ms")

        print(f"\nAll predictions:")
        print("-" * 60)
        for pred in sorted(all_preds, key=lambda x: x["predicted_time"])[:10]:
            print(
                f"  ({pred['block_x']:3d}, {pred['block_y']:3d}): {pred['predicted_time']:.6f} ms"
            )

        return

    # Training mode
    if not args.csv_file:
        parser.error("CSV file required for training")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("EXECUTION TIME PREDICTOR (Neural Network)")
    print("=" * 70)
    print(f"Data: {args.csv_file}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    # Prepare data with size stratification
    df_full = pd.read_csv(args.csv_file)

    # Apply size-based stratification
    if args.size_filter == "small":
        df_filtered = df_full[df_full["N"] <= 256]
        print(f"🔍 Training on SMALL problems only (N ≤ 256)")
        print(f"   Filtered: {len(df_filtered):,} / {len(df_full):,} samples\n")
    elif args.size_filter == "medium":
        df_filtered = df_full[df_full["N"] <= 2048]
        print(f"🔍 Training on SMALL+MEDIUM problems (N ≤ 2048)")
        print(f"   Filtered: {len(df_filtered):,} / {len(df_full):,} samples\n")
    elif args.size_filter == "large":
        df_filtered = df_full[df_full["N"] > 2048]
        print(f"🔍 Training on LARGE problems only (N > 2048)")
        print(f"   Filtered: {len(df_filtered):,} / {len(df_full):,} samples\n")
    else:
        df_filtered = df_full

    # Save filtered data temporarily
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        temp_csv = tmp.name
        df_filtered.to_csv(temp_csv, index=False)

    X, y, features, kernel_encoder, df = prepare_data(
        temp_csv,
        use_kernel_names=not args.no_kernel_names,
        use_log_target=not args.no_log,
    )

    # Clean up temp file
    os.remove(temp_csv)

    # Compute training statistics for bounds
    exec_times = df["exec_time"].values
    training_bounds = (exec_times.min(), exec_times.max())
    N_range = (df["N"].min(), df["N"].max())

    print(f"\nTraining bounds:")
    print(f"  Exec time: {training_bounds[0]:.6f} - {training_bounds[1]:.6f} ms")
    print(f"  N range: {N_range[0]} - {N_range[1]}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Datasets
    train_dataset = KernelDataset(X_train_scaled, y_train)
    test_dataset = KernelDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ExecTimePredictor(
        input_dim=len(features), hidden_dims=args.hidden, dropout=args.dropout
    )

    trainer = ExecTimePredictorTrainer(model, device=device)

    # Train
    history = trainer.train(train_loader, test_loader, epochs=args.epochs, lr=args.lr)

    # Evaluate
    metrics = evaluate_model(
        trainer, test_loader, scaler, features, use_log_target=not args.no_log
    )

    # Save
    trainer.save(
        str(output_dir / "exec_time_predictor_nn.pth"),
        str(output_dir / "exec_time_predictor_scaler.pkl"),
        {
            "scaler": scaler,
            "features": features,
            "kernel_encoder": kernel_encoder,
            "use_log_target": not args.no_log,
            "training_bounds": training_bounds,
            "N_range": N_range,
        },
    )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nTo find optimal config:")
    print(f"  python3 {__file__} --optimize \\")
    print(f"    --kernel myKernel --N 4096 --dim 2")


if __name__ == "__main__":
    main()
