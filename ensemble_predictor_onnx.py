"""
Lightweight Ensemble Predictor for ONNX Inference Only
No PyTorch dependency required

This module provides the same interface as EnsemblePredictor from ensemble_dnn.py
but only supports ONNX models and doesn't require torch to be installed.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path


class EnsemblePredictor:
    """Smart ensemble predictor with 5-regime model routing (ONNX-only)"""

    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.metadata = {}
        self.use_onnx = True  # Always True for this module

        print(f"Loading ensemble models from {models_dir}...")

        import onnxruntime as ort

        for model_name in ["fast", "medium_fast", "medium", "medium_slow", "slow"]:
            model_path = self.models_dir / model_name
            if not model_path.exists():
                continue

            meta_path = model_path / "metadata.pkl"
            if not meta_path.exists():
                continue

            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            onnx_path = model_path / "model.onnx"
            if not onnx_path.exists():
                print(f"  ⚠️  {model_name}: ONNX model not found, skipping")
                continue

            session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            self.models[model_name] = session

            time_range = meta.get("time_range", (None, None))
            print(
                f"  ✓ {model_name}: {time_range[0] if time_range[0] else 0:.3f} - {time_range[1] if time_range[1] else 'inf'} ms (ONNX)"
            )

            self.metadata[model_name] = meta

        if not self.models:
            raise ValueError(f"No ONNX models found in {models_dir}")

    def estimate_regime(self, kernel_info):
        """Estimate which time regime this kernel falls into (5 regimes)"""
        N = kernel_info.get("N", 1024)
        dim = kernel_info.get("dimensionality", 1)
        compute = kernel_info.get("compute_intensity", 1.0)

        estimated_time = (N**dim) * compute / 100000.0

        if estimated_time < 0.5:
            return "fast"
        elif estimated_time < 5.0:
            return "medium_fast"
        elif estimated_time < 50.0:
            return "medium"
        elif estimated_time < 500.0:
            return "medium_slow"
        else:
            return "slow"

    def predict_single(self, model_name, kernel_info, block_x, block_y):
        """Predict using a specific model"""
        meta = self.metadata[model_name]
        model = self.models[model_name]

        feature_values = []

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
                    "estimated_memory_bytes": kernel_info.get(
                        "estimated_memory_bytes", 0
                    ),
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

        temp_df["log_N"] = np.log(temp_df["N"] + 1)
        temp_df["sqrt_N"] = np.sqrt(temp_df["N"])
        temp_df["N_squared"] = temp_df["N"] ** 2
        temp_df["N_cubed"] = temp_df["N"] ** 3

        temp_df["total_threads"] = temp_df["block_x"] * temp_df["block_y"]
        temp_df["log_threads"] = np.log(temp_df["total_threads"] + 1)
        temp_df["block_aspect_ratio"] = temp_df["block_x"] / (temp_df["block_y"] + 1)
        temp_df["block_product"] = temp_df["block_x"] * temp_df["block_y"]

        temp_df["N_per_thread"] = temp_df["N"] / (temp_df["total_threads"] + 1)
        temp_df["compute_per_thread"] = (
            temp_df["compute_intensity"]
            * temp_df["arithmetic_ops"]
            / (temp_df["total_threads"] + 1)
        )
        temp_df["memory_per_thread"] = temp_df["estimated_memory_bytes"] / (
            temp_df["total_threads"] + 1
        )
        temp_df["flops_per_thread"] = temp_df["estimated_flops"] / (
            temp_df["total_threads"] + 1
        )

        temp_df["compute_to_memory_ratio"] = temp_df["arithmetic_ops"] / (
            temp_df["memory_ops"] + 1
        )
        temp_df["read_write_ratio"] = temp_df["global_reads"] / (
            temp_df["global_writes"] + 1
        )

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

        pred_log = model.run(["output"], {"input": X_scaled.astype(np.float32)})[0][
            0, 0
        ]

        return float(np.exp(pred_log))

    def predict_optimal_config(self, kernel_info, candidate_configs=None):
        """Find optimal config using appropriate model(s)"""
        regime = self.estimate_regime(kernel_info)

        if regime not in self.models:
            regime = list(self.models.keys())[0]
            print(f"⚠ Using fallback model: {regime}")

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

        predictions = []
        for block_x, block_y in candidate_configs:
            pred_time = self.predict_single(regime, kernel_info, block_x, block_y)
            predictions.append(
                {"block_x": block_x, "block_y": block_y, "predicted_time": pred_time}
            )

        predictions.sort(key=lambda x: x["predicted_time"])
        best = predictions[0]

        return best["block_x"], best["block_y"], best["predicted_time"], predictions
