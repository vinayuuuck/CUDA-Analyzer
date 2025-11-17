#!/usr/bin/env python3
"""
Grid/Block Configuration Suggester
Predicts optimal CUDA grid and block dimensions for a given kernel
"""
import pickle
import numpy as np
import sys
import os

sys.path.insert(0, "CUDA-Analyzer")
from cuda_anal import CUDAKernelAnalyzer


class GridBlockSuggester:
    def __init__(
        self, model_path="grid_block_model.pkl", features_path="feature_names.pkl"
    ):
        # Load model
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            if isinstance(model_data, dict):
                self.model = model_data["model"]
                self.log_transform = model_data.get("log_transform", False)
            else:
                self.model = model_data
                self.log_transform = False

        # Load feature names
        with open(features_path, "rb") as f:
            self.feature_names = pickle.load(f)

        print("✓ Model loaded successfully")

    def suggest(self, kernel_file, problem_size):
        """
        Suggest optimal grid and block configuration

        Args:
            kernel_file: Path to .cu file
            problem_size: int or tuple (W, H) or (W, H, D)

        Returns:
            dict with 'block', 'grid', 'predicted_time'
        """
        # Analyze kernel to extract static features
        print(f"\n{'='*70}")
        print(f"Analyzing kernel: {kernel_file}")
        print(f"{'='*70}")

        if not os.path.exists(kernel_file):
            raise FileNotFoundError(f"Kernel file not found: {kernel_file}")

        analyzer = CUDAKernelAnalyzer(kernel_file)
        analyzer.analyze()

        if not analyzer.kernels:
            raise ValueError("No CUDA kernels found in file!")

        kernel_info = analyzer.kernels[0]

        # Extract static features
        dimensionality = kernel_info["thread_usage"]["dimensionality"]
        compute_intensity = kernel_info["metrics"]["compute_intensity"]
        has_shared_memory = int(kernel_info["memory_access"]["shared_memory"])
        global_reads = kernel_info["memory_access"]["global_reads"]
        global_writes = kernel_info["memory_access"]["global_writes"]
        arithmetic_ops = kernel_info["operations"]["arithmetic"]
        memory_ops = kernel_info["operations"]["memory"]

        print(f"  Kernel: {kernel_info['name']}")
        print(f"  Dimensionality: {dimensionality}D")
        print(f"  Compute intensity: {compute_intensity:.2f}")
        print(f"  Arithmetic ops: {arithmetic_ops}")
        print(f"  Memory ops: {memory_ops}")

        # Generate candidate configurations based on dimensionality
        if dimensionality == 1 or not (
            kernel_info["thread_usage"]["uses_threadIdx_y"]
            or kernel_info["thread_usage"]["uses_threadIdx_z"]
        ):
            # 1D kernel
            candidates = [
                (32, 1, 1),
                (64, 1, 1),
                (128, 1, 1),
                (256, 1, 1),
                (512, 1, 1),
                (1024, 1, 1),
            ]
            print(f"\nTesting {len(candidates)} 1D configurations...")
        else:
            # 2D/3D kernel (most PolyBench kernels are 2D)
            candidates = [
                (8, 8, 1),
                (16, 16, 1),
                (32, 32, 1),
                (16, 8, 1),
                (8, 16, 1),
                (32, 16, 1),
                (16, 32, 1),
                (64, 4, 1),
                (4, 64, 1),
                (32, 8, 1),
                (8, 32, 1),
            ]
            print(f"\nTesting {len(candidates)} 2D configurations...")

        # Normalize problem size
        if isinstance(problem_size, (list, tuple)):
            N = problem_size[0]
        else:
            N = problem_size

        # Predict time for each candidate
        best_config = None
        best_time = float("inf")
        predictions = []

        for bx, by, bz in candidates:
            total_threads = bx * by * bz

            # Prepare features in the same order as training
            features = np.array(
                [
                    [
                        N,
                        bx,
                        by,
                        bz,
                        total_threads,
                        dimensionality,
                        compute_intensity,
                        has_shared_memory,
                        global_reads,
                        global_writes,
                        arithmetic_ops,
                        memory_ops,
                    ]
                ]
            )

            # Predict
            pred_log_time = self.model.predict(features)[0]

            # Convert back from log scale
            if self.log_transform:
                pred_time = np.expm1(pred_log_time)
            else:
                pred_time = pred_log_time

            predictions.append((bx, by, bz, pred_time))

            if pred_time < best_time:
                best_time = pred_time
                best_config = (bx, by, bz)

        # Calculate grid dimensions
        bx, by, bz = best_config

        if isinstance(problem_size, (list, tuple)):
            if len(problem_size) == 2:
                W, H = problem_size
                grid = ((W + bx - 1) // bx, (H + by - 1) // by, 1)
            elif len(problem_size) == 3:
                W, H, D = problem_size
                grid = ((W + bx - 1) // bx, (H + by - 1) // by, (D + bz - 1) // bz)
            else:
                grid = ((N + bx - 1) // bx, 1, 1)
        else:
            # 1D problem
            grid = ((N + bx - 1) // bx, 1, 1)

        # Show top 3 configurations
        predictions_sorted = sorted(predictions, key=lambda x: x[3])
        print(f"\nTop 3 configurations:")
        for i, (bx, by, bz, time) in enumerate(predictions_sorted[:3], 1):
            print(f"  {i}. Block ({bx:3d}, {by:3d}, {bz}) → {time:.6f}s")

        return {
            "kernel": kernel_info["name"],
            "block": best_config,
            "grid": grid,
            "predicted_time": best_time,
            "all_predictions": predictions_sorted,
        }


def main():
    if len(sys.argv) < 3:
        print("Usage: python grid_block_suggester.py <kernel.cu> <problem_size>")
        print("\nExamples:")
        print("  python grid_block_suggester.py gemm.cu 8192")
        print("  python grid_block_suggester.py conv2d.cu 1024")
        sys.exit(1)

    kernel_file = sys.argv[1]
    problem_size = int(sys.argv[2])

    suggester = GridBlockSuggester()
    result = suggester.suggest(kernel_file, problem_size)

    print(f"\n{'='*70}")
    print(f"RECOMMENDED CONFIGURATION")
    print(f"{'='*70}")
    print(f"Problem size: {problem_size}")
    print(f"Block dims:   {result['block']}")
    print(f"Grid dims:    {result['grid']}")
    print(f"Predicted execution time: {result['predicted_time']:.6f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
