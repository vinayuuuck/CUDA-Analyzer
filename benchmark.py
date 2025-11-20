#!/usr/bin/env python3
import json
import subprocess
import tempfile
import os
import random
import statistics
import re


class CUDABenchmarker:
    def __init__(self, test_suite_json, num_random_trials=10, num_runs_per_config=5):
        with open(test_suite_json, "r") as f:
            self.test_suite = json.load(f)

        self.num_random_trials = num_random_trials
        self.num_runs_per_config = num_runs_per_config
        self.results = {}

    def generate_random_block_config(self, max_threads=1024):
        valid_dims = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        valid_z = [1, 2, 4, 8]

        for _ in range(100):
            block_x = random.choice(valid_dims)
            block_y = random.choice(valid_dims)
            block_z = random.choice(valid_z)
            total = block_x * block_y * block_z

            if 32 <= total <= max_threads and total % 32 == 0:
                return block_x, block_y, block_z

        return 256, 1, 1

    def compile_cuda(self, source_file, output_binary, block_x, block_y, block_z):
        try:
            cmd = [
                "nvcc",
                source_file,
                "-o",
                output_binary,
                f"-DBLOCK_DIM_X={block_x}",
                f"-DBLOCK_DIM_Y={block_y}",
                f"-DBLOCK_DIM_Z={block_z}",
                "-O3",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0

        except Exception as e:
            print(f"  Compilation error: {e}")
            return False

    def run_and_time(self, binary_path):
        import time

        times = []

        for _ in range(self.num_runs_per_config):
            try:
                start = time.time()
                result = subprocess.run(
                    [binary_path], capture_output=True, text=True, timeout=10
                )
                end = time.time()

                if result.returncode != 0:
                    return None

                time_ms = (end - start) * 1000  # Convert to ms
                times.append(time_ms)

            except subprocess.TimeoutExpired:
                return None
            except Exception as e:
                print(f"    Warning: {e}")
                return None

        return statistics.mean(times) if times else None

    def benchmark_kernel(self, kernel_key, kernel_info):
        print(f"\n{'='*70}")
        print(f"{kernel_key}")
        print(f"{'='*70}")

        cuda_file = kernel_info["cuda_file"]

        # Test both ensemble and random forest configs if available
        configs_to_test = []

        if "ensemble_config" in kernel_info and kernel_info["ensemble_config"]:
            ens = kernel_info["ensemble_config"]
            configs_to_test.append(
                ("Ensemble DNN", ens["block_x"], ens["block_y"], ens.get("block_z", 1))
            )

        if (
            "random_forest_config" in kernel_info
            and kernel_info["random_forest_config"]
        ):
            rf = kernel_info["random_forest_config"]
            configs_to_test.append(
                ("Random Forest", rf["block_x"], rf["block_y"], rf.get("block_z", 1))
            )

        # Fallback to optimal_config if no specific model configs
        if not configs_to_test:
            suggested = kernel_info["optimal_config"]
            configs_to_test.append(
                (
                    "Suggested",
                    suggested["block_x"],
                    suggested["block_y"],
                    suggested.get("block_z", 1),
                )
            )

        model_results = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test each model's configuration
            for model_name, bx, by, bz in configs_to_test:
                print(f"\n{model_name}: ({bx}, {by}, {bz})")

                binary = os.path.join(tmpdir, f"{model_name.replace(' ', '_').lower()}")

                if not self.compile_cuda(cuda_file, binary, bx, by, bz):
                    print("  Compilation failed")
                    continue

                exec_time = self.run_and_time(binary)
                if exec_time is None:
                    print("  Execution failed")
                    continue

                print(f"  Time: {exec_time:.4f} ms")
                model_results[model_name] = {"config": (bx, by, bz), "time": exec_time}

            if not model_results:
                print("  All configurations failed")
                return None

            # Test random configurations
            print(f"\nTesting {self.num_random_trials} random configs...")
            random_times = []

            for trial in range(self.num_random_trials):
                rand_bx, rand_by, rand_bz = self.generate_random_block_config()

                random_binary = os.path.join(tmpdir, f"random_{trial}")

                if not self.compile_cuda(
                    cuda_file, random_binary, rand_bx, rand_by, rand_bz
                ):
                    continue

                rand_time = self.run_and_time(random_binary)
                if rand_time is not None:
                    random_times.append(rand_time)
                    print(
                        f"  ({rand_bx:3d}, {rand_by:3d}, {rand_bz}) → {rand_time:.4f} ms"
                    )

        if not random_times:
            print("No successful random trials")
            return None

        best_random = min(random_times)
        avg_random = statistics.mean(random_times)

        print(f"\nRandom - Best: {best_random:.4f} ms  Avg: {avg_random:.4f} ms")

        # Compare each model against random
        result = {
            "kernel": kernel_key,
            "model_results": model_results,
            "best_random": best_random,
            "avg_random": avg_random,
        }

        for model_name, model_data in model_results.items():
            model_time = model_data["time"]
            speedup_vs_best = best_random / model_time
            speedup_vs_avg = avg_random / model_time

            print(f"\n{model_name}:")
            print(
                f"  Speedup vs best: {speedup_vs_best:.2f}×  vs avg: {speedup_vs_avg:.2f}×"
            )

            if speedup_vs_best >= 1.0:
                print("  ✅ Beats or ties best random")
            elif speedup_vs_avg >= 1.0:
                print("  ✓ Beats average random")
            else:
                print("  ⚠ Random was better")

            result[f"{model_name}_speedup_vs_best"] = speedup_vs_best
            result[f"{model_name}_speedup_vs_avg"] = speedup_vs_avg
            result[f"{model_name}_beats_best"] = speedup_vs_best >= 1.0
            result[f"{model_name}_beats_avg"] = speedup_vs_avg >= 1.0

        return result

    def run_all_benchmarks(self):
        print("=" * 70)
        print("BENCHMARK: Suggested vs Random Configs")
        print("=" * 70)

        results = []
        kernels = self.test_suite.get("kernels", {})

        for i, (key, info) in enumerate(kernels.items(), 1):
            print(f"\n[{i}/{len(kernels)}]", end=" ")
            result = self.benchmark_kernel(key, info)
            if result:
                results.append(result)

        self.print_summary(results)
        return results

    def print_summary(self, results):
        if not results:
            print("\n\nNo successful benchmarks")
            return

        print("\n\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        total = len(results)

        # Collect stats for each model
        model_names = set()
        for r in results:
            model_names.update(r.get("model_results", {}).keys())

        model_stats = {}
        for model_name in sorted(model_names):
            beats_best = sum(
                1 for r in results if r.get(f"{model_name}_beats_best", False)
            )
            beats_avg = sum(
                1 for r in results if r.get(f"{model_name}_beats_avg", False)
            )

            speedups_best = [
                r[f"{model_name}_speedup_vs_best"]
                for r in results
                if f"{model_name}_speedup_vs_best" in r
            ]
            speedups_avg = [
                r[f"{model_name}_speedup_vs_avg"]
                for r in results
                if f"{model_name}_speedup_vs_avg" in r
            ]

            if speedups_best and speedups_avg:
                model_stats[model_name] = {
                    "beats_best": beats_best,
                    "beats_avg": beats_avg,
                    "avg_speedup_best": statistics.mean(speedups_best),
                    "avg_speedup_avg": statistics.mean(speedups_avg),
                    "geom_mean_speedup": statistics.geometric_mean(speedups_avg),
                }

        print(f"\nKernels tested: {total}")

        for model_name, stats in model_stats.items():
            print(f"\n{model_name}:")
            print(
                f"  Beats best random: {stats['beats_best']}/{total} ({100*stats['beats_best']/total:.0f}%)"
            )
            print(
                f"  Beats avg random:  {stats['beats_avg']}/{total} ({100*stats['beats_avg']/total:.0f}%)"
            )
            print(f"  Avg speedup (vs best): {stats['avg_speedup_best']:.2f}×")
            print(f"  Avg speedup (vs avg):  {stats['avg_speedup_avg']:.2f}×")
            print(f"  Geometric mean:        {stats['geom_mean_speedup']:.2f}×")

        print("\n" + "─" * 70)
        print("Per-Kernel Results:")
        print("─" * 70)
        for r in results:
            name = r["kernel"].split("::")[-1][:30]
            print(f"\n{name}")
            for model_name in sorted(model_names):
                if f"{model_name}_speedup_vs_avg" in r:
                    speedup = r[f"{model_name}_speedup_vs_avg"]
                    beats_best = r.get(f"{model_name}_beats_best", False)
                    beats_avg = r.get(f"{model_name}_beats_avg", False)
                    status = "✅" if beats_best else ("✓" if beats_avg else "⚠")
                    print(f"  {model_name:15} {speedup:.2f}× {status}")
        print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-suite", default="test_suite_clean.json")
    parser.add_argument("--random-trials", type=int, default=10)
    parser.add_argument("--runs-per-config", type=int, default=5)
    args = parser.parse_args()

    try:
        subprocess.run(["nvcc", "--version"], capture_output=True, check=True)
    except:
        print("Error: nvcc not found")
        return 1

    benchmarker = CUDABenchmarker(
        args.test_suite, args.random_trials, args.runs_per_config
    )
    benchmarker.run_all_benchmarks()
    return 0


if __name__ == "__main__":
    exit(main())
