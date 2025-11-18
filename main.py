import sys
import os
from pathlib import Path
import pickle
import numpy as np
import torch
from typing import List, Tuple, Optional, Set
from cuda_anal import CUDAKernelAnalyzer
import warnings

# Suppress sklearn feature name warnings and joblib parallel output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
# Suppress joblib parallel backend messages
import logging
logging.getLogger('joblib').setLevel(logging.ERROR)


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

    def suggest(self, cuda_file, N=None):
        """
        Suggest optimal configurations using all available models

        Args:
            cuda_file: Path to .cu file
            N: Problem size (optional, defaults to 1024 for candidate generation)

        Returns:
            dict with results from all models
        """
        # Use default value if N not provided
        if N is None:
            N = 1024
            n_display = f"{N} (default)"
        else:
            n_display = str(N)
        
        print(f"\n{'='*80}")
        print(f"ANALYZING: {cuda_file}")
        print(f"Problem size: N={n_display}")
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

            candidates_raw = self.generate_candidate_block_configs(
                kernel_info, N=N, dimensionality=kernel_info["dimensionality"]
            )
            candidates = [(bx, by, bz) for (bx, by, bz) in candidates_raw]
            kernel_result = {
                "kernel": kernel_name,
                "dimensionality": kernel_info["dimensionality"],
                "models": {},
            }

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

    @staticmethod
    def _powers_of_two_upto(max_val):
        """Generate powers of 2 up to max_val"""
        v = 1
        out = []
        while v <= max_val:
            out.append(v)
            v *= 2
        return out

    @staticmethod
    def _divisors_of(n: int, max_val: int):
        """Return reasonable divisors of n up to max_val (keeps only divisors >=8)."""
        if n <= 0:
            return []
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                if 8 <= i <= max_val:
                    divs.append(i)
                j = n // i
                if 8 <= j <= max_val:
                    divs.append(j)
            i += 1
        return sorted(set(divs))

    @staticmethod
    def generate_candidate_block_configs(
        kernel_info: dict,
        N: Optional[int] = None,
        dimensionality: Optional[int] = None,
        max_threads_per_block: int = 1024,
        warp_size: int = 32,
        prefer_more_threads_if_compute_bound: bool = True,
        refine_topk: int = 6,
    ) -> List[Tuple[int, int, int]]:
        """
        Generate a prioritized list of (block_x, block_y, block_z) candidates.
        - kernel_info: dictionary with keys like compute_intensity, has_shared_memory
        - N: optional problem size (int). If provided, include divisors of N as candidate tile sizes.
        - dimensionality: if not provided, read from kernel_info['dimensionality'] (default 1)
        """
        if dimensionality is None:
            dimensionality = int(kernel_info.get("dimensionality", 1))

        compute_intensity = float(kernel_info.get("compute_intensity", 1.0))
        has_shared = bool(kernel_info.get("has_shared_memory", False))

        candidates: List[Tuple[int, int, int]] = []
        seen: Set[Tuple[int, int, int]] = set()

        # --- 1D kernels: block_x choices ---
        if dimensionality == 1:
            # Baseline warp-friendly sizes
            base = [32, 64, 128, 256, 512, 1024]
            base = [b for b in base if b <= max_threads_per_block]

            # if N known, include divisors of N and neighbors
            if N:
                divs = MultiModelSuggester._divisors_of(N, max_threads_per_block)
                base = sorted(set(base + divs))

            # bias: if compute-heavy prefer larger block sizes (more threads), else smaller
            if prefer_more_threads_if_compute_bound and compute_intensity >= 2.0:
                base = sorted(base, reverse=True)
            else:
                base = sorted(base)

            for b in base:
                t = (b, 1, 1)
                if t not in seen:
                    candidates.append(t)
                    seen.add(t)

            return candidates

        # --- 2D kernels: build tile-like candidates ---
        # Good practice: include square tiles, rectangular tiles, and warp-aligned rows
        small = [8, 16, 32, 64, 128]
        # make sure we keep values <= max_threads_per_block
        small = [x for x in small if x <= max_threads_per_block]
        # generate combos
        basic_tiles = [
            (16, 16),
            (32, 8),
            (8, 32),
            (32, 16),
            (16, 32),
            (32, 32),
            (64, 4),
            (64, 8),
            (64, 16),
            (128, 8),
            (128, 4),
            (256, 1),
            (8, 8),
        ]
        # restrict to those with <= max_threads_per_block
        basic_tiles = [t for t in basic_tiles if t[0] * t[1] <= max_threads_per_block]

        # If N known, prefer divisors and tiles that divide N nicely (square tiling)
        divisors = []
        if N:
            divisors = MultiModelSuggester._divisors_of(N, max_threads_per_block)

        # helper to add candidate with warp friendliness (multiple of warp where possible)
        def add2(bx, by):
            if bx * by > max_threads_per_block or bx <= 0 or by <= 0:
                return
            # nudge to warp multiple: try making bx or by multiple of warp_size
            for bx_try in {bx, ((bx + warp_size - 1) // warp_size) * warp_size}:
                for by_try in {by, ((by + warp_size - 1) // warp_size) * warp_size}:
                    if bx_try * by_try <= max_threads_per_block:
                        t = (int(bx_try), int(by_try), 1)
                        if t not in seen:
                            candidates.append(t)
                            seen.add(t)

        # seed with basic tiles
        for bx, by in basic_tiles:
            add2(bx, by)

        # include tiles from small * small grid
        for bx in small:
            for by in small:
                if bx * by <= max_threads_per_block:
                    add2(bx, by)

        # include divisors-derived tiles: try (div,div), (div, 16), (16, div)
        if divisors:
            for d in divisors[:6]:
                add2(d, d)
                add2(d, 16)
                add2(16, d)
                add2(d, 8)
                add2(8, d)

        # bias selection by compute intensity and shared memory:
        # - compute-bound: prefer larger total threads (sort descending)
        # - memory-bound or shared-memory heavy: prefer smaller tiles with better locality (sort ascending)
        if (
            prefer_more_threads_if_compute_bound and compute_intensity >= 2.0
        ) and not has_shared:
            candidates = sorted(candidates, key=lambda t: -(t[0] * t[1]))
        else:
            candidates = sorted(candidates, key=lambda t: (t[0] * t[1]))

        # --- local hierarchical refine: evaluate top-K neighborhood ---
        # build neighbor multipliers
        def neighbors(tile):
            bx, by, _ = tile
            neigh = []
            for mx in (0.5, 1.0, 2.0):
                for my in (0.5, 1.0, 2.0):
                    nbx = int(max(1, round(bx * mx)))
                    nby = int(max(1, round(by * my)))
                    if nbx * nby <= max_threads_per_block:
                        neigh.append((nbx, nby, 1))
            return neigh

        topk = candidates[:refine_topk]
        for t in topk:
            for n in neighbors(t):
                if n not in seen:
                    candidates.append(n)
                    seen.add(n)

        # final prune: keep unique, limited size, sorted by preference (preserve existing order)
        final = []
        for t in candidates:
            if t not in final:
                final.append(t)
        # limit to ~60 candidates to bound runtime
        return final[:60]

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
    if len(sys.argv) < 2:
        print("CUDA Kernel Launch Configuration Optimizer")
        print("=" * 60)
        print("\nUsage: python3 main.py <cuda_file.cu> [N]")
        print("\nExamples:")
        print("  python3 main.py vec_add.cu")
        print("  python3 main.py vec_add.cu 1048576")
        print(
            "  python3 main.py PolybenchCUDA/stencils/convolution-2d/2DConvolution.cu 2048"
        )
        print("\nArguments:")
        print("  cuda_file.cu  Path to CUDA kernel file")
        print("  N             Problem size (optional, default: 1024)")
        print("\nThis tool uses multiple ML models to suggest optimal launch configs:")
        print("  • Random Forest (baseline)")
        print("  • Ensemble DNNs (best accuracy)")
        print("  • Fine-tuned LLM (if available)")
        print()
        sys.exit(1)

    cuda_file = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) >= 3 else None

    try:
        suggester = MultiModelSuggester()
        results = suggester.suggest(cuda_file, N)
        suggester.print_summary(results, N)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
