#!/usr/bin/env python3
"""
CUDA Kernel Launch Configuration Optimizer
Suggests optimal grid/block configurations using multiple ML approaches
"""
import sys
import os
from pathlib import Path
import pickle
import numpy as np
import torch
from cuda_anal import CUDAKernelAnalyzer


class MultiModelSuggester:
    """
    Suggests optimal launch configs using 3 approaches:
    1. Random Forest (baseline)
    2. Ensemble DNNs (best accuracy)
    3. Fine-tuned LLM (if available)
    """

    def __init__(self):
        self.models_loaded = {"random_forest": False, "ensemble": False, "llm": False}

        # Try to load Random Forest
        try:
            self._load_random_forest()
        except Exception as e:
            print(f"⚠ Random Forest not available: {e}")

        # Try to load Ensemble DNNs
        try:
            self._load_ensemble()
        except Exception as e:
            print(f"⚠ Ensemble DNNs not available: {e}")

        # Try to load LLM
        try:
            self._load_llm()
        except Exception as e:
            print(f"⚠ LLM not available: {e}")

        if not any(self.models_loaded.values()):
            raise RuntimeError("No models available! Train at least one model first.")

    def _load_random_forest(self):
        """Load Random Forest model"""
        model_path = "grid_block_model.pkl"
        features_path = "feature_names.pkl"

        if not os.path.exists(model_path):
            return

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            if isinstance(model_data, dict):
                self.rf_model = model_data["model"]
                self.rf_log_transform = model_data.get("log_transform", False)
            else:
                self.rf_model = model_data
                self.rf_log_transform = False

        with open(features_path, "rb") as f:
            self.rf_feature_names = pickle.load(f)

        self.models_loaded["random_forest"] = True
        print("✓ Random Forest loaded")

    def _load_ensemble(self):
        """Load Ensemble DNN models"""
        from ensemble_exec_time_predictor import EnsemblePredictor

        models_dir = "ensemble_models"
        if not os.path.exists(models_dir):
            return

        # Check if at least one model exists
        has_models = any(
            (Path(models_dir) / model).exists() for model in ["fast", "medium", "slow"]
        )
        if not has_models:
            return

        self.ensemble = EnsemblePredictor(models_dir)
        self.models_loaded["ensemble"] = True
        print("✓ Ensemble DNNs loaded")

    def _load_llm(self):
        """Load fine-tuned LLM"""
        import re
        from transformers import AutoTokenizer, AutoModelForCausalLM

        llm_dir = "cuda_block_predictor_llm"
        if not os.path.exists(llm_dir):
            return

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_dir)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_dir, device_map="auto" if torch.cuda.is_available() else None
        )
        self.llm_model.eval()

        self.models_loaded["llm"] = True
        print("✓ LLM loaded")

    def _predict_random_forest(self, kernel_info, N, candidates):
        """Predict using Random Forest"""
        predictions = []

        for bx, by, bz in candidates:
            total_threads = bx * by * bz

            features = np.array(
                [
                    [
                        N,
                        bx,
                        by,
                        bz,
                        total_threads,
                        kernel_info["dimensionality"],
                        kernel_info["compute_intensity"],
                        kernel_info["has_shared_memory"],
                        kernel_info["global_reads"],
                        kernel_info["global_writes"],
                        kernel_info["arithmetic_ops"],
                        kernel_info["memory_ops"],
                    ]
                ]
            )

            pred_log_time = self.rf_model.predict(features)[0]
            pred_time = (
                np.expm1(pred_log_time) if self.rf_log_transform else pred_log_time
            )

            predictions.append(
                {
                    "block_x": bx,
                    "block_y": by,
                    "block_z": bz,
                    "predicted_time": pred_time,
                }
            )

        return sorted(predictions, key=lambda x: x["predicted_time"])

    def _predict_ensemble(self, kernel_info, N, candidates):
        """Predict using Ensemble DNNs"""
        # Add N to kernel_info
        kernel_info_with_n = kernel_info.copy()
        kernel_info_with_n["N"] = N

        # Get predictions for all candidates
        _, _, _, all_preds = self.ensemble.predict_optimal_config(
            kernel_info_with_n, candidate_configs=[(bx, by) for bx, by, _ in candidates]
        )

        return all_preds

    def _predict_llm(self, kernel_info, N, candidates):
        """Predict using fine-tuned LLM"""
        import re

        predictions = []

        for bx, by, bz in candidates:
            total_threads = bx * by

            prompt = f"""### Kernel: {kernel_info['kernel_name']}
N: {N}
dimensionality: {kernel_info['dimensionality']}D
compute_intensity: {kernel_info['compute_intensity']:.2f}
has_shared_memory: {kernel_info['has_shared_memory']}
global_reads: {kernel_info['global_reads']}
global_writes: {kernel_info['global_writes']}
arithmetic_ops: {kernel_info['arithmetic_ops']}
memory_ops: {kernel_info['memory_ops']}
Configuration: block_dims=({bx}, {by}, 1), total_threads={total_threads}

### Predicted Execution Time:
"""

            inputs = self.llm_tokenizer(prompt, return_tensors="pt")
            device = next(self.llm_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                )

            generated_text = self.llm_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            # Extract exec_time
            match = re.search(r"exec_time:\s*([\d.]+)\s*ms", generated_text)
            if match:
                exec_time = float(match.group(1))
            else:
                exec_time = 999.0  # Fallback

            predictions.append(
                {
                    "block_x": bx,
                    "block_y": by,
                    "block_z": bz,
                    "predicted_time": exec_time / 1000.0,  # Convert to seconds
                }
            )

        return sorted(predictions, key=lambda x: x["predicted_time"])

    def suggest(self, cuda_file, N):
        """
        Suggest optimal configurations using all available models

        Args:
            cuda_file: Path to .cu file
            N: Problem size

        Returns:
            dict with results from all models
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING: {cuda_file}")
        print(f"Problem size: N={N}")
        print(f"{'='*80}\n")

        # Analyze kernel
        if not os.path.exists(cuda_file):
            raise FileNotFoundError(f"File not found: {cuda_file}")

        analyzer = CUDAKernelAnalyzer(cuda_file)
        kernels = analyzer.analyze()

        if not kernels:
            raise ValueError("No CUDA kernels found!")

        results = []

        for kernel_data in kernels:
            kernel_name = kernel_data["name"]
            print(f"\n{'─'*80}")
            print(f"KERNEL: {kernel_name}")
            print(f"{'─'*80}")

            # Extract features
            metrics = kernel_data["metrics"]
            memory = kernel_data["memory_access"]
            ops = kernel_data["operations"]
            thread_usage = kernel_data["thread_usage"]

            kernel_info = {
                "kernel_name": kernel_name,
                "dimensionality": metrics.get(
                    "dimensionality", thread_usage.get("dimensionality", 1)
                ),
                "compute_intensity": metrics.get("compute_intensity", 1.0),
                "has_shared_memory": int(
                    metrics.get("has_shared_memory", memory.get("shared_memory", False))
                ),
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

            print(f"  Dimensionality: {kernel_info['dimensionality']}D")
            print(
                f"  Compute Intensity: {kernel_info['compute_intensity']:.2f} FLOPs/byte"
            )
            print(
                f"  Shared Memory: {'Yes' if kernel_info['has_shared_memory'] else 'No'}"
            )

            # Generate candidates
            dim = kernel_info["dimensionality"]
            if dim == 1:
                candidates = [
                    (32, 1, 1),
                    (64, 1, 1),
                    (128, 1, 1),
                    (256, 1, 1),
                    (512, 1, 1),
                    (1024, 1, 1),
                ]
            else:
                candidates = [
                    (8, 8, 1),
                    (16, 8, 1),
                    (16, 16, 1),
                    (32, 8, 1),
                    (32, 16, 1),
                    (32, 32, 1),
                    (64, 4, 1),
                    (64, 8, 1),
                    (64, 16, 1),
                    (128, 4, 1),
                    (128, 8, 1),
                    (256, 4, 1),
                ]

            kernel_result = {"kernel": kernel_name, "dimensionality": dim, "models": {}}

            # Predict with each available model
            if self.models_loaded["random_forest"]:
                print(f"\n  Random Forest predictions:")
                rf_preds = self._predict_random_forest(kernel_info, N, candidates)
                best_rf = rf_preds[0]
                print(
                    f"    Best: ({best_rf['block_x']}, {best_rf['block_y']}, {best_rf['block_z']}) → {best_rf['predicted_time']:.6f}s"
                )
                kernel_result["models"]["random_forest"] = {
                    "best": best_rf,
                    "all": rf_preds[:5],
                }

            if self.models_loaded["ensemble"]:
                print(f"\n  Ensemble DNN predictions:")
                ens_preds = self._predict_ensemble(kernel_info, N, candidates)
                best_ens = ens_preds[0]
                print(
                    f"    Best: ({best_ens['block_x']}, {best_ens['block_y']}, 1) → {best_ens['predicted_time']:.6f}s"
                )
                kernel_result["models"]["ensemble"] = {
                    "best": best_ens,
                    "all": ens_preds[:5],
                }

            if self.models_loaded["llm"]:
                print(f"\n  LLM predictions:")
                llm_preds = self._predict_llm(kernel_info, N, candidates)
                best_llm = llm_preds[0]
                print(
                    f"    Best: ({best_llm['block_x']}, {best_llm['block_y']}, {best_llm['block_z']}) → {best_llm['predicted_time']:.6f}s"
                )
                kernel_result["models"]["llm"] = {
                    "best": best_llm,
                    "all": llm_preds[:5],
                }

            results.append(kernel_result)

        return results

    def print_summary(self, results, N):
        """Print final summary with recommendations"""
        print(f"\n{'='*80}")
        print(f"RECOMMENDED LAUNCH CONFIGURATIONS (N={N})")
        print(f"{'='*80}\n")

        for result in results:
            kernel_name = result["kernel"]
            dim = result["dimensionality"]

            print(f"{kernel_name}:")
            print(f"  Dimensionality: {dim}D\n")

            # Show recommendation from each model
            for model_name, model_data in result["models"].items():
                best = model_data["best"]
                bx, by = best["block_x"], best["block_y"]
                bz = best.get("block_z", 1)
                time = best["predicted_time"]

                # Compute grid
                if dim == 1:
                    grid = f"({(N + bx - 1) // bx}, 1, 1)"
                else:
                    grid = f"({(N + bx - 1) // bx}, {(N + by - 1) // by}, 1)"

                model_label = {
                    "random_forest": "Random Forest",
                    "ensemble": "Ensemble DNN",
                    "llm": "LLM",
                }.get(model_name, model_name)

                print(
                    f"  {model_label:15} → <<<{grid}, ({bx}, {by}, {bz})>>> ({time:.6f}s)"
                )

            # Consensus recommendation
            if len(result["models"]) > 1:
                # Use ensemble if available, otherwise RF
                if "ensemble" in result["models"]:
                    best = result["models"]["ensemble"]["best"]
                    model_used = "Ensemble DNN"
                elif "random_forest" in result["models"]:
                    best = result["models"]["random_forest"]["best"]
                    model_used = "Random Forest"
                else:
                    best = result["models"]["llm"]["best"]
                    model_used = "LLM"

                bx, by = best["block_x"], best["block_y"]
                bz = best.get("block_z", 1)

                if dim == 1:
                    grid = f"({(N + bx - 1) // bx}, 1, 1)"
                else:
                    grid = f"({(N + bx - 1) // bx}, {(N + by - 1) // by}, 1)"

                print(f"\n  ⭐ RECOMMENDED (using {model_used}):")
                print(f"     {kernel_name}<<<{grid}, ({bx}, {by}, {bz})>>>();")

            print()


def main():
    if len(sys.argv) < 3:
        print("CUDA Kernel Launch Configuration Optimizer")
        print("=" * 60)
        print("\nUsage: python3 main.py <cuda_file.cu> <N>")
        print("\nExamples:")
        print("  python3 main.py vec_add.cu 1048576")
        print(
            "  python3 main.py PolybenchCUDA/stencils/convolution-2d/2DConvolution.cu 2048"
        )
        print("\nThis tool uses multiple ML models to suggest optimal launch configs:")
        print("  • Random Forest (baseline)")
        print("  • Ensemble DNNs (best accuracy)")
        print("  • Fine-tuned LLM (if available)")
        print()
        sys.exit(1)

    cuda_file = sys.argv[1]
    N = int(sys.argv[2])

    try:
        suggester = MultiModelSuggester()
        results = suggester.suggest(cuda_file, N)
        suggester.print_summary(results, N)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
