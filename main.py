import sys
import os
import math
import warnings
import statistics
from collections import defaultdict
from pathlib import Path
import pickle
import numpy as np
import torch
from typing import List, Tuple, Optional, Set
from cuda_anal import CUDAKernelAnalyzer

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"


class MultiModelSuggester:
    """
    Suggests optimal launch configs using 3 approaches:
    1. Random Forest (baseline)
    2. Ensemble DNNs (best accuracy)
    3. Fine-tuned LLM (if available)
    """

    def __init__(self):
        self.models_loaded = {"random_forest": False, "ensemble": False, "llm": False}

        try:
            self._load_random_forest()
        except Exception as e:
            print(f"Random Forest not available: {e}")

        try:
            self._load_ensemble()
        except Exception as e:
            print(f"Ensemble DNNs not available: {e}")

        try:
            self._load_llm()
        except Exception as e:
            print(f"LLM not available: {e}")

        if not any(self.models_loaded.values()):
            raise RuntimeError("No models available! Train at least one model first.")

    def _load_random_forest(self):
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
        print("Random Forest loaded")

    def _load_ensemble(self):
        from ensemble_exec_time_predictor_large import EnsemblePredictor

        models_dir = "ensemble_models_large"
        if not os.path.exists(models_dir):
            return

        has_models = any(
            (Path(models_dir) / model).exists()
            for model in ["fast", "medium_fast", "medium", "medium_slow", "slow"]
        )
        if not has_models:
            return

        self.ensemble = EnsemblePredictor(models_dir)
        self.models_loaded["ensemble"] = True
        print("Ensemble DNNs loaded")

    def _load_llm(self):
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
        print("LLM loaded")

    def _predict_random_forest(self, kernel_info, N, candidates):
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
                        kernel_info.get("control_flow_ops", 0),
                        kernel_info.get("loop_ops", 0),
                        kernel_info.get("uses_syncthreads", 0),
                        kernel_info.get("estimated_flops", 0),
                        kernel_info.get("estimated_memory_bytes", 0),
                        kernel_info.get("uses_threadIdx_x", 0),
                        kernel_info.get("uses_threadIdx_y", 0),
                        kernel_info.get("uses_threadIdx_z", 0),
                        kernel_info.get("uses_blockIdx_x", 0),
                        kernel_info.get("uses_blockIdx_y", 0),
                        kernel_info.get("uses_blockIdx_z", 0),
                        kernel_info.get("uses_blockDim_x", 0),
                        kernel_info.get("uses_blockDim_y", 0),
                        kernel_info.get("uses_blockDim_z", 0),
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
        kernel_info_with_n = kernel_info.copy()
        kernel_info_with_n["N"] = N

        _, _, _, all_preds = self.ensemble.predict_optimal_config(
            kernel_info_with_n, candidate_configs=[(bx, by) for bx, by, _ in candidates]
        )

        return all_preds

    def _predict_llm(self, kernel_info, N, candidates):
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

    def pick_best_across_Ns(self, per_N_results, strategy: str = "geom_mean"):

        by_config = defaultdict(list)
        for r in per_N_results:
            t = r.get("predicted_time")
            if t is None:
                continue
            key = (int(r["block_x"]), int(r["block_y"]), int(r.get("block_z", 1)))
            by_config[key].append((int(r["N"]), float(t)))

        stats = {}
        for k, samples in by_config.items():
            times = [t for (_, t) in samples]
            if not times:
                continue
            mean_t = statistics.mean(times)
            median_t = statistics.median(times)
            safe_times = [max(1e-12, tt) for tt in times]
            geom_mean_t = math.exp(
                sum(math.log(tt) for tt in safe_times) / len(safe_times)
            )
            max_t = max(times)
            stats[k] = {
                "times": times,
                "mean": mean_t,
                "median": median_t,
                "geom_mean": geom_mean_t,
                "max": max_t,
                "count_best": 0,
                "num_samples": len(times),
            }

        # count per-N winners
        times_by_N = defaultdict(list)
        for (bx, by, bz), samples in by_config.items():
            for N, t in samples:
                times_by_N[N].append(((bx, by, bz), t))
        for N, lst in times_by_N.items():
            best_key = min(lst, key=lambda x: x[1])[0]
            stats[best_key]["count_best"] = stats[best_key].get("count_best", 0) + 1

        # scoring strategies
        scores = {}
        if strategy == "geom_mean":
            for k, s in stats.items():
                scores[k] = s["geom_mean"]
        elif strategy == "median":
            for k, s in stats.items():
                scores[k] = s["median"]
        elif strategy == "min_max":
            for k, s in stats.items():
                scores[k] = s["max"]
        elif strategy == "mean":
            for k, s in stats.items():
                scores[k] = s["mean"]
        elif strategy == "consensus":
            max_count = max(s["count_best"] for s in stats.values())
            candidates = [k for k, s in stats.items() if s["count_best"] == max_count]
            best = min(candidates, key=lambda k: stats[k]["geom_mean"])
            for k in stats:
                scores[k] = 0 if k == best else 1
        else:
            for k, s in stats.items():
                scores[k] = s["geom_mean"]

        if not scores:
            raise ValueError("No candidate stats available to pick from")

        best_key = min(scores.items(), key=lambda x: x[1])[0]
        best_stats = stats[best_key].copy()
        best_stats.update(
            {
                "block_x": best_key[0],
                "block_y": best_key[1],
                "block_z": best_key[2],
                "score": scores[best_key],
            }
        )
        return best_stats, stats

    def evaluate_fixed_Ns_and_pick(
        self,
        cuda_file: str,
        Ns=None,
        strategy: str = "geom_mean",
        topk_per_N: int = 1,
        verbose: bool = True,
    ):
        if Ns is None:
            Ns = [128, 256, 512, 1024, 2048, 4096]

        if not os.path.exists(cuda_file):
            raise FileNotFoundError(f"File not found: {cuda_file}")

        analyzer = CUDAKernelAnalyzer(cuda_file)
        kernels = analyzer.analyze()
        if not kernels:
            raise ValueError("No CUDA kernels found!")

        all_kernel_results = []

        for kernel_data in kernels:
            kernel_name = kernel_data["name"]
            if verbose:
                print(f"\n{'='*70}\nKernel: {kernel_name}\n{'='*70}")

            metrics = kernel_data["metrics"]
            memory = kernel_data["memory_access"]
            ops = kernel_data["operations"]
            thread_usage = kernel_data["thread_usage"]

            kernel_info_base = {
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
                "uses_syncthreads": int(metrics.get("uses_syncthreads", False)),
            }

            per_N_results = []

            for N in Ns:
                kernel_info = dict(kernel_info_base)
                kernel_info["N"] = int(N)

                candidates_raw = self.generate_candidate_block_configs(
                    kernel_info, N=int(N), dimensionality=kernel_info["dimensionality"]
                )
                candidates = [(bx, by, bz) for (bx, by, bz) in candidates_raw]

                all_model_preds = {}

                if self.models_loaded.get("ensemble"):
                    all_model_preds["ensemble"] = self._predict_ensemble(
                        kernel_info, int(N), candidates
                    )

                if self.models_loaded.get("random_forest"):
                    all_model_preds["random_forest"] = self._predict_random_forest(
                        kernel_info, int(N), candidates
                    )

                if self.models_loaded.get("llm"):
                    all_model_preds["llm"] = self._predict_llm(
                        kernel_info, int(N), candidates
                    )

                if not all_model_preds:
                    raise RuntimeError("No models available to predict!")

                if verbose:
                    print(f"\n N={N:6d}")

                    for model_name, preds in all_model_preds.items():
                        if not preds:
                            continue
                        best = preds[0]
                        model_label = {
                            "ensemble": "Ensemble DNN",
                            "random_forest": "Random Forest",
                            "llm": "LLM",
                        }.get(model_name, model_name)

                        print(
                            f"   {model_label:15} best: block=({best['block_x']},{best['block_y']},1)  pred={best['predicted_time']:.6f}"
                        )
                        for alt in preds[1:topk_per_N]:
                            print(
                                f"   {' '*15}  alt: block=({alt['block_x']},{alt['block_y']},1)  pred={alt['predicted_time']:.6f}"
                            )

                primary_model = (
                    "ensemble"
                    if "ensemble" in all_model_preds
                    else list(all_model_preds.keys())[0]
                )
                preds = all_model_preds[primary_model]

                if not preds:
                    continue

                best = preds[0]
                bx = int(best["block_x"])
                by = int(best["block_y"])
                bz = int(best.get("block_z", 1))

                # Store predictions from all available models
                model_best_configs = {}
                for model_name, model_preds in all_model_preds.items():
                    if model_preds:
                        model_best = model_preds[0]
                        model_best_configs[model_name] = {
                            "block_x": int(model_best["block_x"]),
                            "block_y": int(model_best["block_y"]),
                            "block_z": int(model_best.get("block_z", 1)),
                            "predicted_time": float(model_best["predicted_time"])
                        }

                # compute grid dims for this N (use dimensionality info)
                dim = kernel_info["dimensionality"]
                # For 2D/3D we assume square domain if only single N provided (common convention).
                if dim == 1:
                    problem_shape = int(N)
                elif dim == 2:
                    problem_shape = (int(N), int(N))
                else:  # 3D (best-effort)
                    problem_shape = (int(N), int(N), 1)

                grid_tuple = MultiModelSuggester.grid_from_block_and_problem(
                    bx, by, problem_shape, dimensionality=dim
                )
                grid_dims = f"({grid_tuple[0]}, {grid_tuple[1]}, {grid_tuple[2]})"

                per_N_results.append(
                    {
                        "N": int(N),
                        "block_x": bx,
                        "block_y": by,
                        "block_z": bz,
                        "predicted_time": float(best["predicted_time"]),
                        "all_preds": preds,
                        "all_model_predictions": all_model_preds,
                        "model_best_configs": model_best_configs,
                        "primary_model": primary_model,
                        "grid": grid_tuple,
                        "grid_dims": grid_dims,
                    }
                )

            if not per_N_results:
                if verbose:
                    print("  No per-N results for this kernel — skipping")
                continue

            chosen, stats = self.pick_best_across_Ns(per_N_results, strategy=strategy)
            
            # Aggregate recommendations per model across all N values
            model_recommendations = {}
            for model_name in ["ensemble", "random_forest", "llm"]:
                model_configs_per_n = []
                for n_result in per_N_results:
                    if model_name in n_result.get("model_best_configs", {}):
                        config = n_result["model_best_configs"][model_name]
                        model_configs_per_n.append({
                            "N": n_result["N"],
                            "block_x": config["block_x"],
                            "block_y": config["block_y"],
                            "block_z": config["block_z"],
                            "predicted_time": config["predicted_time"],
                        })
                
                if model_configs_per_n:
                    # Pick best config for this model using the same strategy
                    model_chosen, _ = self.pick_best_across_Ns(model_configs_per_n, strategy=strategy)
                    model_recommendations[model_name] = model_chosen

            if verbose:
                print("\n" + "="*70)
                print("MODEL RECOMMENDATIONS")
                print("="*70)
                
                # Determine primary model
                primary = per_N_results[0].get("primary_model", "ensemble") if per_N_results else "ensemble"
                
                for model_name in ["ensemble", "random_forest", "llm"]:
                    if model_name in model_recommendations:
                        rec = model_recommendations[model_name]
                        model_label = {
                            "ensemble": "Ensemble DNN",
                            "random_forest": "Random Forest",
                            "llm": "LLM",
                        }[model_name]
                        
                        is_primary = "✓ PREFERRED" if model_name == primary else "         "
                        print(f"{is_primary} {model_label:15} → block=({rec['block_x']},{rec['block_y']},{rec['block_z']})  score={rec['score']:.6f}")
                
                print("="*70)
                
                print("\n  Aggregated per-config stats (showing top 6 by geom_mean):")
                sorted_stats = sorted(stats.items(), key=lambda kv: kv[1]["geom_mean"])
                for (bx, by, bz), s in sorted_stats[:6]:
                    print(
                        f"    block=({bx},{by},{bz})  geom_mean={s['geom_mean']:.6f}  median={s['median']:.6f}  max={s['max']:.6f}  wins={s['count_best']}"
                    )

                print(
                    f"\n  ✓ CHOSEN by '{strategy}': block=({chosen['block_x']},{chosen['block_y']},{chosen['block_z']})  score={chosen['score']:.6f}"
                )
                print("  Per-N bests:")
                for r in per_N_results:
                    print(
                        f"    N={r['N']:6d}  best=({r['block_x']},{r['block_y']},{r.get('block_z',1)})  grid={r['grid_dims']}  pred={r['predicted_time']:.6f} ms"
                    )

            all_kernel_results.append(
                {
                    "kernel": kernel_name,
                    "per_N_results": per_N_results,
                    "chosen": chosen,
                    "stats": stats,
                    "model_recommendations": model_recommendations,
                }
            )

        return all_kernel_results

    @staticmethod
    def grid_from_block_and_problem(
        block_x: int, block_y: int, problem_shape, dimensionality: int = 1
    ):
        """
        Return (grid_x, grid_y, grid_z) that cover problem_shape with block dims.
        problem_shape may be:
          - int (interpreted as N for 1D or Nx=Ny=N for 2D)
          - tuple/list of ints (Nx,) or (Nx,Ny) or (Nx,Ny,Nz)
        """
        # normalize problem_shape into tuple
        if isinstance(problem_shape, (int, np.integer)):
            if dimensionality == 1:
                Nx = int(problem_shape)
                gx = max(1, math.ceil(Nx / block_x))
                return (gx, 1, 1)
            elif dimensionality == 2:
                Nx = int(problem_shape)
                gx = max(1, math.ceil(Nx / block_x))
                gy = max(1, math.ceil(Nx / block_y))
                return (gx, gy, 1)
            else:
                Nx = int(problem_shape)
                gx = max(1, math.ceil(Nx / block_x))
                gy = max(1, math.ceil(Nx / block_y))
                gz = 1
                return (gx, gy, gz)
        # tuple/list
        try:
            shape = tuple(int(x) for x in problem_shape)
        except Exception:
            # fallback to 1D N=1
            return (1, 1, 1)

        if dimensionality == 1:
            Nx = shape[0]
            gx = max(1, math.ceil(Nx / block_x))
            return (gx, 1, 1)
        elif dimensionality == 2:
            if len(shape) >= 2:
                Nx, Ny = shape[0], shape[1]
            else:
                Nx = Ny = shape[0]
            gx = max(1, math.ceil(Nx / block_x))
            gy = max(1, math.ceil(Ny / block_y))
            return (gx, gy, 1)
        elif dimensionality == 3:
            if len(shape) >= 3:
                Nx, Ny, Nz = shape[0], shape[1], shape[2]
            elif len(shape) == 2:
                Nx, Ny = shape
                Nz = 1
            else:
                Nx = Ny = shape[0]
                Nz = 1
            gx = max(1, math.ceil(Nx / block_x))
            gy = max(1, math.ceil(Ny / block_y))
            gz = max(1, math.ceil(Nz / 1))
            return (gx, gy, gz)
        else:
            return (1, 1, 1)

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
        Generate prioritized list of (block_x, block_y, block_z) candidates.
        Supports 1D, 2D, and 3D kernels.
        Keeps total threads <= max_threads_per_block.
        """
        if dimensionality is None:
            dimensionality = int(kernel_info.get("dimensionality", 1))

        compute_intensity = float(kernel_info.get("compute_intensity", 1.0))
        has_shared = bool(kernel_info.get("has_shared_memory", False))

        candidates: List[Tuple[int, int, int]] = []
        seen: Set[Tuple[int, int, int]] = set()

        # helper to add candidate, nudging to warp-size multiples
        def add_candidate(bx, by, bz=1):
            if bx <= 0 or by <= 0 or bz <= 0:
                return
            if bx * by * bz > max_threads_per_block:
                return
            # try nudging bx/by/bz to warp multiples where sensible (only for bx or by)
            bx_opts = {bx, ((bx + warp_size - 1) // warp_size) * warp_size}
            by_opts = {by, ((by + warp_size - 1) // warp_size) * warp_size}
            bz_opts = {bz}
            for bx_try in bx_opts:
                for by_try in by_opts:
                    for bz_try in bz_opts:
                        if bx_try * by_try * bz_try <= max_threads_per_block:
                            t = (int(bx_try), int(by_try), int(bz_try))
                            if t not in seen:
                                candidates.append(t)
                                seen.add(t)

        # 1D kernels -> vary block_x; keep by=bz=1
        if dimensionality == 1:
            base = [32, 64, 128, 256, 512, 1024]
            base = [b for b in base if b <= max_threads_per_block]
            if N:
                divs = MultiModelSuggester._divisors_of(N, max_threads_per_block)
                base = sorted(set(base + divs))
            if prefer_more_threads_if_compute_bound and compute_intensity >= 2.0:
                base = sorted(base, reverse=True)
            else:
                base = sorted(base)
            for b in base:
                add_candidate(b, 1, 1)
            return candidates

        # For 2D and 3D kernels we build base tiles and then extend with small z values for 3D
        small = [4, 8, 16, 32, 64, 128]
        small = [x for x in small if x <= max_threads_per_block]

        basic_tiles_2d = [
            (8, 8),
            (16, 16),
            (32, 8),
            (8, 32),
            (16, 32),
            (32, 16),
            (32, 32),
            (64, 4),
            (64, 8),
            (64, 16),
            (128, 4),
            (128, 8),
            (256, 1),
        ]
        basic_tiles_2d = [
            t for t in basic_tiles_2d if t[0] * t[1] <= max_threads_per_block
        ]

        # divisors for tiling if N present
        divisors = []
        if N:
            divisors = MultiModelSuggester._divisors_of(N, max_threads_per_block)

        # candidate z-values for 3D kernels (keep small to avoid exceeding per-block threads)
        # For 3D kernels, bias towards higher z values by listing them first
        z_candidates = [8, 4, 2, 1] if dimensionality >= 3 else [1]

        # seed with basic 2D tiles and small*small grid
        for bx, by in basic_tiles_2d:
            for bz in z_candidates:
                add_candidate(bx, by, bz)

        for bx in small:
            for by in small:
                for bz in z_candidates:
                    if bx * by * bz <= max_threads_per_block:
                        add_candidate(bx, by, bz)

        # include divisor-derived tiles
        if divisors:
            for d in divisors[:8]:
                for bz in z_candidates:
                    add_candidate(d, d, bz)
                    add_candidate(d, 16, bz)
                    add_candidate(16, d, bz)
                    add_candidate(d, 8, bz)
                    add_candidate(8, d, bz)

        # bias selection
        if dimensionality >= 3:
            # For 3D kernels, prefer configs with higher block_z values
            candidates = sorted(candidates, key=lambda t: (-t[2], t[0] * t[1] * t[2]))
        elif (
            prefer_more_threads_if_compute_bound and compute_intensity >= 2.0
        ) and not has_shared:
            candidates = sorted(candidates, key=lambda t: -(t[0] * t[1] * t[2]))
        else:
            candidates = sorted(candidates, key=lambda t: (t[0] * t[1] * t[2]))

        # local neighborhood refine for top-K
        def neighbors(tile):
            bx, by, bz = tile
            neigh = []
            for mx in (0.5, 1.0, 2.0):
                for my in (0.5, 1.0, 2.0):
                    for mz in (1.0, 2.0) if dimensionality >= 3 else (1.0,):
                        nbx = int(max(1, round(bx * mx)))
                        nby = int(max(1, round(by * my)))
                        nbz = int(max(1, round(bz * mz)))
                        if nbx * nby * nbz <= max_threads_per_block:
                            neigh.append((nbx, nby, nbz))
            return neigh

        topk = candidates[:refine_topk]
        for t in topk:
            for n in neighbors(t):
                if n not in seen:
                    candidates.append(n)
                    seen.add(n)

        # dedupe and limit
        final = []
        for t in candidates:
            if t not in final:
                final.append(t)
        return final[:120]  # allow a few more candidates for 3D searches


def main():
    if len(sys.argv) < 2:
        print("CUDA Kernel Launch Configuration Optimizer")
        print("=" * 60)
        print("\nUsage: python3 main.py <cuda_file.cu>")
        print("\nExamples:")
        print("  python3 main.py vec_add.cu")
        print(
            "  python3 main.py PolybenchCUDA/stencils/convolution-2d/2DConvolution.cu"
        )
        print("\nThis tool evaluates kernels across multiple problem sizes")
        print("and recommends the best configuration using:")
        print("  • Random Forest (baseline)")
        print("  • Ensemble DNNs (best accuracy)")
        print("  • Fine-tuned LLM (if available)")
        print()
        sys.exit(1)

    cuda_file = sys.argv[1]

    try:
        suggester = MultiModelSuggester()
        results = suggester.evaluate_fixed_Ns_and_pick(
            cuda_file,
            Ns=[128, 256, 512, 1024, 2048, 4096],
            strategy="geom_mean",
            topk_per_N=2,
            verbose=True,
        )

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
