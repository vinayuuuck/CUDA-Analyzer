"""
Ensemble Execution Time Predictor with Multi-Model Architecture

Architecture:
- Fast model: exec_time < 1ms (launch overhead dominated)
- Medium model: 1ms ≤ exec_time < 100ms (computation dominated)
- Slow model: exec_time ≥ 100ms (memory/large problem dominated)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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
    """Neural network that predicts execution time"""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
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

        # Output: log(exec_time)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ModelTrainer:
    """Handles training with best practices"""

    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.history = {"train_loss": [], "val_loss": [], "epoch": []}

    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader, criterion):
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
        self, train_loader, val_loader, epochs=200, lr=0.001, patience=30, verbose=True
    ):
        """Training loop with early stopping"""
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15, verbose=False
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        if verbose:
            print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'LR':<15}")
            print("-" * 60)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % 20 == 0:
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
            print(f"Best validation loss: {best_val_loss:.6f}\n")

        return best_val_loss


def prepare_features(df):
    """Prepare features with engineering"""
    # N-based features
    df["log_N"] = np.log(df["N"] + 1)
    df["sqrt_N"] = np.sqrt(df["N"])
    df["N_squared"] = df["N"] ** 2

    # Block configuration features
    df["total_threads"] = df["block_x"] * df["block_y"]
    df["log_threads"] = np.log(df["total_threads"] + 1)
    df["block_aspect_ratio"] = df["block_x"] / (df["block_y"] + 1)

    # Interaction features
    df["N_per_thread"] = df["N"] / (df["total_threads"] + 1)
    df["compute_per_thread"] = (
        df["compute_intensity"] * df["arithmetic_ops"] / (df["total_threads"] + 1)
    )

    return df


def prepare_data(csv_file, time_min=None, time_max=None, use_kernel_names=True):
    """Prepare data with time filtering"""
    df = pd.read_csv(csv_file)

    # Handle column naming
    column_mapping = {
        "kernel": "kernel_name",
        "bx": "block_x",
        "by": "block_y",
        "bz": "block_z",
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Filter by execution time
    if time_min is not None:
        df = df[df["exec_time"] >= time_min]
    if time_max is not None:
        df = df[df["exec_time"] < time_max]

    print(f"Data: {len(df):,} samples")
    print(
        f"Exec time range: {df['exec_time'].min():.6f} - {df['exec_time'].max():.6f} ms"
    )
    print(
        f"Log exec time range: {np.log(df['exec_time'].min()):.2f} - {np.log(df['exec_time'].max()):.2f}"
    )

    # Feature engineering
    df = prepare_features(df)

    # Define feature columns
    numeric_features = [
        "N",
        "log_N",
        "sqrt_N",
        "N_squared",
        "block_x",
        "block_y",
        "total_threads",
        "log_threads",
        "block_aspect_ratio",
        "N_per_thread",
        "dimensionality",
        "compute_intensity",
        "compute_per_thread",
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

    # Kernel encoding
    kernel_encoder = None
    if use_kernel_names:
        kernel_encoder = LabelEncoder()
        df["kernel_encoded"] = kernel_encoder.fit_transform(df["kernel_name"])
        numeric_features.insert(0, "kernel_encoded")

    X = df[numeric_features].values
    y = np.log(df["exec_time"].values)  # Always use log

    print(f"Features ({len(numeric_features)}): {numeric_features[:5]}...")

    return X, y, numeric_features, kernel_encoder, df


def evaluate_model(model, test_loader, scaler, device="cpu"):
    """Comprehensive evaluation"""
    model.eval()
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).cpu().numpy().flatten()
            all_predictions.append(outputs)
            all_actuals.append(batch_y.numpy().flatten())

    predictions_log = np.concatenate(all_predictions)
    actuals_log = np.concatenate(all_actuals)

    # Convert to real space
    predictions_real = np.exp(predictions_log)
    actuals_real = np.exp(actuals_log)

    # Metrics
    r2_log = r2_score(actuals_log, predictions_log)
    mae_log = mean_absolute_error(actuals_log, predictions_log)

    r2_real = r2_score(actuals_real, predictions_real)
    mae_real = mean_absolute_error(actuals_real, predictions_real)

    mape = np.mean(np.abs((actuals_real - predictions_real) / actuals_real)) * 100

    pct_errors = np.abs((actuals_real - predictions_real) / actuals_real) * 100
    within_10 = (pct_errors < 10).mean() * 100
    within_20 = (pct_errors < 20).mean() * 100

    print(f"  R² (log): {r2_log:.4f}  |  R² (real): {r2_real:.4f}")
    print(f"  MAE (log): {mae_log:.4f}  |  MAE (real): {mae_real:.6f} ms")
    print(
        f"  MAPE: {mape:.2f}%  |  Within 10%: {within_10:.1f}%  |  Within 20%: {within_20:.1f}%"
    )

    return {
        "r2_log": r2_log,
        "r2_real": r2_real,
        "mae_log": mae_log,
        "mae_real": mae_real,
        "mape": mape,
        "within_10pct": within_10,
        "within_20pct": within_20,
    }


def train_specialized_model(
    csv_file, time_min, time_max, model_name, output_dir, epochs=200
):
    """Train a model for a specific time regime"""
    print("\n" + "=" * 70)
    print(f"TRAINING {model_name.upper()} MODEL")
    print(
        f"Time range: {time_min if time_min else 0:.3f} - {time_max if time_max else 'inf'} ms"
    )
    print("=" * 70)

    X, y, features, kernel_encoder, df = prepare_data(csv_file, time_min, time_max)

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Datasets
    train_dataset = KernelDataset(X_train_scaled, y_train)
    test_dataset = KernelDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ExecTimePredictor(
        input_dim=len(features), hidden_dims=[128, 64, 32], dropout=0.2
    )

    trainer = ModelTrainer(model, device=device)
    best_loss = trainer.train(train_loader, test_loader, epochs=epochs, patience=30)

    # Evaluate
    print(f"\n{model_name.upper()} MODEL EVALUATION:")
    metrics = evaluate_model(model, test_loader, scaler, device)

    # Save
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "model.pth")

    with open(output_path / "metadata.pkl", "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "features": features,
                "kernel_encoder": kernel_encoder,
                "time_range": (time_min, time_max),
                "metrics": metrics,
            },
            f,
        )

    print(f"✓ Saved to {output_path}/")

    return metrics


class EnsemblePredictor:
    """Smart ensemble predictor with model routing"""

    def __init__(self, models_dir):
        self.models = {}
        self.metadata = {}

        for model_name in ["fast", "medium", "slow"]:
            model_path = Path(models_dir) / model_name
            if model_path.exists():
                # Load metadata
                with open(model_path / "metadata.pkl", "rb") as f:
                    meta = pickle.load(f)

                # Load model
                model = ExecTimePredictor(input_dim=len(meta["features"]))
                model.load_state_dict(torch.load(model_path / "model.pth"))
                model.eval()

                self.models[model_name] = model
                self.metadata[model_name] = meta

                print(
                    f"✓ Loaded {model_name} model (R²={meta['metrics']['r2_real']:.3f})"
                )

    def estimate_regime(self, kernel_info):
        """Estimate which time regime this kernel falls into"""
        # Simple heuristic based on problem size and complexity
        N = kernel_info.get("N", 1024)
        dim = kernel_info.get("dimensionality", 1)
        compute = kernel_info.get("compute_intensity", 1.0)

        # Rough estimate: time ~ N^dim * compute / 1000
        estimated_time = (N**dim) * compute / 100000.0

        if estimated_time < 1.0:
            return "fast"
        elif estimated_time < 100.0:
            return "medium"
        else:
            return "slow"

    def predict_single(self, model_name, kernel_info, block_x, block_y):
        """Predict using a specific model"""
        meta = self.metadata[model_name]
        model = self.models[model_name]

        # Prepare features (match training)
        feature_values = []

        # Create temp dataframe to use prepare_features
        temp_df = pd.DataFrame(
            [
                {
                    "N": kernel_info.get("N", 1024),
                    "block_x": block_x,
                    "block_y": block_y,
                    "dimensionality": kernel_info.get("dimensionality", 1),
                    "compute_intensity": kernel_info.get("compute_intensity", 1.0),
                    "has_shared_memory": kernel_info.get("has_shared_memory", 0),
                    "global_reads": kernel_info.get("global_reads", 10),
                    "global_writes": kernel_info.get("global_writes", 5),
                    "arithmetic_ops": kernel_info.get("arithmetic_ops", 100),
                    "memory_ops": kernel_info.get("memory_ops", 15),
                    "control_flow_ops": kernel_info.get("control_flow_ops", 0),
                    "loop_ops": kernel_info.get("loop_ops", 0),
                    "uses_syncthreads": kernel_info.get("uses_syncthreads", 0),
                    "estimated_flops": kernel_info.get("estimated_flops", 0),
                    "estimated_memory_bytes": kernel_info.get("estimated_memory_bytes", 0),
                    "uses_threadIdx_x": kernel_info.get("uses_threadIdx_x", 0),
                    "uses_threadIdx_y": kernel_info.get("uses_threadIdx_y", 0),
                    "uses_threadIdx_z": kernel_info.get("uses_threadIdx_z", 0),
                    "uses_blockIdx_x": kernel_info.get("uses_blockIdx_x", 0),
                    "uses_blockIdx_y": kernel_info.get("uses_blockIdx_y", 0),
                    "uses_blockIdx_z": kernel_info.get("uses_blockIdx_z", 0),
                    "uses_blockDim_x": kernel_info.get("uses_blockDim_x", 0),
                    "uses_blockDim_y": kernel_info.get("uses_blockDim_y", 0),
                    "uses_blockDim_z": kernel_info.get("uses_blockDim_z", 0),
                }
            ]
        )

        temp_df = prepare_features(temp_df)

        # Extract feature values in correct order
        for feat in meta["features"]:
            if feat == "kernel_encoded":
                try:
                    encoded = meta["kernel_encoder"].transform(
                        [kernel_info["kernel_name"]]
                    )[0]
                except:
                    encoded = 0
                feature_values.append(encoded)
            elif feat in temp_df.columns:
                feature_values.append(temp_df[feat].iloc[0])
            else:
                feature_values.append(0)

        X = np.array([feature_values])
        X_scaled = meta["scaler"].transform(X)

        # Predict
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            pred_log = model(X_tensor).item()

        return np.exp(pred_log)

    def predict_optimal_config(self, kernel_info, candidate_configs=None):
        """Find optimal config using appropriate model(s)"""
        # Determine regime
        regime = self.estimate_regime(kernel_info)

        if regime not in self.models:
            # Fallback to any available model
            regime = list(self.models.keys())[0]
            print(f"⚠ Using fallback model: {regime}")

        # Generate candidates
        if candidate_configs is None:
            dim = kernel_info.get("dimensionality", 1)
            if dim == 1:
                candidate_configs = [
                    (32, 1),
                    (64, 1),
                    (128, 1),
                    (256, 1),
                    (512, 1),
                    (1024, 1),
                ]
            else:
                candidate_configs = [
                    (8, 8),
                    (16, 8),
                    (16, 16),
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

        # Predict for all configs
        predictions = []
        for block_x, block_y in candidate_configs:
            pred_time = self.predict_single(regime, kernel_info, block_x, block_y)
            predictions.append(
                {"block_x": block_x, "block_y": block_y, "predicted_time": pred_time}
            )

        # Find best
        best = min(predictions, key=lambda x: x["predicted_time"])

        return best["block_x"], best["block_y"], best["predicted_time"], predictions


def analyze_cuda_file_and_predict(cuda_file, N, ensemble, output_dir="ensemble_models"):
    """
    Analyze a CUDA source file and predict optimal configs for all kernels
    
    Args:
        cuda_file: Path to .cu file
        N: Problem size to optimize for
        ensemble: EnsemblePredictor instance
        output_dir: Directory with trained models
    """
    from cuda_anal import CUDAKernelAnalyzer
    import os
    
    if not os.path.exists(cuda_file):
        print(f"Error: File '{cuda_file}' not found")
        return
    
    print("=" * 80)
    print(f"ANALYZING CUDA FILE: {cuda_file}")
    print("=" * 80)
    
    # Analyze the CUDA file
    analyzer = CUDAKernelAnalyzer(cuda_file)
    kernels = analyzer.analyze()
    
    if not kernels:
        print("No CUDA kernels found in file!")
        return
    
    print(f"\nFound {len(kernels)} kernel(s)\n")
    
    # Predict optimal config for each kernel
    results = []
    
    for i, kernel_data in enumerate(kernels, 1):
        kernel_name = kernel_data["name"]
        print(f"\n{'='*80}")
        print(f"KERNEL #{i}: {kernel_name}")
        print(f"{'='*80}")
        
        # Extract features from cuda_anal
        metrics = kernel_data["metrics"]
        memory = kernel_data["memory_access"]
        ops = kernel_data["operations"]
        thread_usage = kernel_data["thread_usage"]
        
        kernel_info = {
            "kernel_name": kernel_name,
            "N": N,
            "dimensionality": metrics.get("dimensionality", thread_usage.get("dimensionality", 1)),
            "compute_intensity": metrics.get("compute_intensity", 1.0),
            "has_shared_memory": int(metrics.get("has_shared_memory", memory.get("shared_memory", False))),
            "global_reads": memory.get("global_reads", 10),
            "global_writes": memory.get("global_writes", 5),
            "arithmetic_ops": ops.get("arithmetic", 100),
            "memory_ops": ops.get("memory", 15),
            "control_flow_ops": ops.get("control_flow", 0),
            "loop_ops": ops.get("loops", 0),
            "uses_syncthreads": int(metrics.get("uses_syncthreads", False)),
            "estimated_flops": metrics.get("estimated_flops", 0),
            "estimated_memory_bytes": metrics.get("estimated_memory_bytes", 0),
            "uses_threadIdx_x": int(thread_usage.get("uses_threadIdx_x", False)),
            "uses_threadIdx_y": int(thread_usage.get("uses_threadIdx_y", False)),
            "uses_threadIdx_z": int(thread_usage.get("uses_threadIdx_z", False)),
            "uses_blockIdx_x": int(thread_usage.get("uses_blockIdx_x", False)),
            "uses_blockIdx_y": int(thread_usage.get("uses_blockIdx_y", False)),
            "uses_blockIdx_z": int(thread_usage.get("uses_blockIdx_z", False)),
            "uses_blockDim_x": int(thread_usage.get("uses_blockDim_x", False)),
            "uses_blockDim_y": int(thread_usage.get("uses_blockDim_y", False)),
            "uses_blockDim_z": int(thread_usage.get("uses_blockDim_z", False)),
        }
        
        # Print kernel characteristics
        print(f"  Location: Line {kernel_data['location']['line']}")
        print(f"  Dimensionality: {kernel_info['dimensionality']}D")
        print(f"  Compute Intensity: {kernel_info['compute_intensity']:.2f} FLOPs/byte")
        print(f"  Memory: {kernel_info['global_reads']} reads, {kernel_info['global_writes']} writes")
        print(f"  Shared Memory: {'Yes' if kernel_info['has_shared_memory'] else 'No'}")
        print(f"  Operations: {kernel_info['arithmetic_ops']} arithmetic, {kernel_info['control_flow_ops']} control flow")
        
        # Predict optimal configuration
        print(f"\n  Optimizing for N={N}...")
        bx, by, pred_time, all_preds = ensemble.predict_optimal_config(kernel_info)
        
        # Compute grid dimensions
        dim = kernel_info['dimensionality']
        if dim == 1:
            grid_x = (N + bx - 1) // bx
            grid_dims = f"({grid_x}, 1, 1)"
        elif dim == 2:
            grid_x = (N + bx - 1) // bx
            grid_y = (N + by - 1) // by
            grid_dims = f"({grid_x}, {grid_y}, 1)"
        else:  # 3D
            grid_x = (N + bx - 1) // bx
            grid_y = (N + by - 1) // by
            grid_z = 1
            grid_dims = f"({grid_x}, {grid_y}, {grid_z})"
        
        print(f"\n  ✓ OPTIMAL LAUNCH CONFIGURATION:")
        print(f"      Block dims:  ({bx}, {by}, 1)")
        print(f"      Grid dims:   {grid_dims}")
        print(f"      Total threads: {bx * by}")
        print(f"      Predicted time: {pred_time:.6f} ms")
        
        print(f"\n  Top 5 configurations:")
        for pred in sorted(all_preds, key=lambda x: x["predicted_time"])[:5]:
            bx_alt = pred['block_x']
            by_alt = pred['block_y']
            if dim == 1:
                gx = (N + bx_alt - 1) // bx_alt
                grid_str = f"grid=({gx},1,1)"
            else:
                gx = (N + bx_alt - 1) // bx_alt
                gy = (N + by_alt - 1) // by_alt
                grid_str = f"grid=({gx},{gy},1)"
            print(f"      block=({bx_alt:4d}, {by_alt:4d}, 1)  {grid_str}  →  {pred['predicted_time']:.6f} ms")
        
        # Store result
        results.append({
            "kernel": kernel_name,
            "block_x": bx,
            "block_y": by,
            "block_z": 1,
            "grid_dims": grid_dims,
            "predicted_time": pred_time,
            "dimensionality": dim
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - RECOMMENDED LAUNCH CONFIGURATIONS")
    print("=" * 80)
    print(f"Problem size: N={N}\n")
    
    for result in results:
        print(f"{result['kernel']:<30}")
        print(f"  Launch: <<<{result['grid_dims']}, ({result['block_x']}, {result['block_y']}, {result['block_z']})>>>")
        print(f"  Predicted time: {result['predicted_time']:.6f} ms\n")
    
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ensemble Execution Time Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the ensemble models
  python3 ensemble_exec_time_predictor.py klaraptor_enriched_data.csv --train --epochs 200
  
  # Analyze a CUDA file and predict optimal configs
  python3 ensemble_exec_time_predictor.py --predict --cuda-file vec_add.cu --N 1048576
  
  # Analyze all kernels in a file for different problem sizes
  python3 ensemble_exec_time_predictor.py --predict --cuda-file 2DConvolution.cu --N 2048
        """
    )
    parser.add_argument("csv_file", nargs="?", help="Path to enriched CSV (for training)")
    parser.add_argument("--train", action="store_true", help="Train all models")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output-dir", default="ensemble_models")

    # Prediction mode - now accepts CUDA file
    parser.add_argument("--predict", action="store_true", help="Predict optimal configs")
    parser.add_argument("--cuda-file", help="Path to CUDA source file (.cu)")
    parser.add_argument("--N", type=int, help="Problem size")

    args = parser.parse_args()

    if args.train:
        if not args.csv_file:
            parser.error("CSV file required for training")

        print("=" * 70)
        print("ENSEMBLE EXECUTION TIME PREDICTOR - TRAINING")
        print("=" * 70)

        # Train three specialized models
        metrics_fast = train_specialized_model(
            args.csv_file,
            time_min=None,
            time_max=1.0,
            model_name="fast",
            output_dir=args.output_dir,
            epochs=args.epochs,
        )

        metrics_medium = train_specialized_model(
            args.csv_file,
            time_min=1.0,
            time_max=100.0,
            model_name="medium",
            output_dir=args.output_dir,
            epochs=args.epochs,
        )

        metrics_slow = train_specialized_model(
            args.csv_file,
            time_min=100.0,
            time_max=None,
            model_name="slow",
            output_dir=args.output_dir,
            epochs=args.epochs,
        )

        # Summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE - SUMMARY")
        print("=" * 70)
        print(
            f"Fast model:   R²={metrics_fast['r2_real']:.3f}, MAPE={metrics_fast['mape']:.1f}%"
        )
        print(
            f"Medium model: R²={metrics_medium['r2_real']:.3f}, MAPE={metrics_medium['mape']:.1f}%"
        )
        print(
            f"Slow model:   R²={metrics_slow['r2_real']:.3f}, MAPE={metrics_slow['mape']:.1f}%"
        )

        print(f"\nModels saved to: {args.output_dir}/")
        print(f"\nTo predict:")
        print(f"  python3 {__file__} --predict --cuda-file mykernel.cu --N 1024")

    elif args.predict:
        if not args.cuda_file:
            parser.error("Prediction requires: --cuda-file")
        if not args.N:
            parser.error("Prediction requires: --N (problem size)")

        # Load trained ensemble
        ensemble = EnsemblePredictor(args.output_dir)
        
        # Analyze CUDA file and predict for all kernels
        analyze_cuda_file_and_predict(args.cuda_file, args.N, ensemble, args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
