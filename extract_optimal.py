#!/usr/bin/env python3
"""
Extract optimal configurations from full KLARAPTOR data
Finds the config with minimum exec_time for each (kernel, N) or (kernel, N, GPU)
"""

import pandas as pd
import sys


def extract_optimal_configs(input_csv, output_csv=None, per_gpu=False):
    """
    Extract optimal configurations

    Args:
        input_csv: Path to CSV with all configs
        output_csv: Path to save optimal configs (default: optimal_configs.csv)
        per_gpu: If True, find optimal per GPU. If False, find global optimal.
    """
    # Load data
    df = pd.read_csv(input_csv)

    print(f"Loaded {len(df):,} total configurations")
    print(
        f"Kernels: {df['kernel_name' if 'kernel_name' in df.columns else 'kernel'].nunique()}"
    )
    print(f"Data sizes: {sorted(df['N'].unique())}")

    # Handle column naming
    kernel_col = "kernel_name" if "kernel_name" in df.columns else "kernel"

    # Group columns
    if per_gpu and "gpu" in df.columns:
        group_cols = [kernel_col, "N", "gpu"]
        print(f"GPUs: {df['gpu'].nunique()}")
        print("Finding optimal config per (kernel, N, GPU)...")
    else:
        group_cols = [kernel_col, "N"]
        print("Finding optimal config per (kernel, N) across all GPUs...")

    # Find optimal configurations
    optimal_rows = []
    stats = {"speedups": [], "configs_tested": []}

    for group_key, group in df.groupby(group_cols):
        # Find row with minimum execution time
        best_idx = group["exec_time"].idxmin()
        best_row = group.loc[best_idx].copy()

        # Calculate statistics
        avg_time = group["exec_time"].mean()
        worst_time = group["exec_time"].max()
        best_time = best_row["exec_time"]

        speedup_vs_avg = avg_time / best_time
        speedup_vs_worst = worst_time / best_time

        best_row["speedup_vs_avg"] = speedup_vs_avg
        best_row["speedup_vs_worst"] = speedup_vs_worst
        best_row["configs_tested"] = len(group)
        best_row["worst_exec_time"] = worst_time
        best_row["avg_exec_time"] = avg_time

        stats["speedups"].append(speedup_vs_avg)
        stats["configs_tested"].append(len(group))

        optimal_rows.append(best_row)

    df_optimal = pd.DataFrame(optimal_rows)

    # Print summary
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal optimal configs: {len(df_optimal):,}")
    print(
        f"Reduction: {len(df) / len(df_optimal):.1f}x (from {len(df):,} to {len(df_optimal):,})"
    )

    print(f"\nConfigs tested per group:")
    print(f"  Mean: {sum(stats['configs_tested']) / len(stats['configs_tested']):.1f}")
    print(f"  Min: {min(stats['configs_tested'])}")
    print(f"  Max: {max(stats['configs_tested'])}")

    print(f"\nSpeedup (optimal vs average):")
    print(f"  Mean: {sum(stats['speedups']) / len(stats['speedups']):.2f}x")
    print(f"  Min: {min(stats['speedups']):.2f}x")
    print(f"  Max: {max(stats['speedups']):.2f}x")

    # Show examples of biggest improvements
    print(f"\nTop 5 biggest improvements (optimal vs worst):")
    print("-" * 70)
    top_improvements = df_optimal.nlargest(5, "speedup_vs_worst")
    for _, row in top_improvements.iterrows():
        kernel = row[kernel_col]
        n = row["N"]
        speedup = row["speedup_vs_worst"]
        bx = row["bx"] if "bx" in row else row.get("block_x", "?")
        by = row["by"] if "by" in row else row.get("block_y", "?")
        print(f"  {kernel:25s} N={n:4d}: {speedup:5.2f}x faster (block={bx}x{by})")

    # Save optimal configs
    if output_csv is None:
        output_csv = "optimal_configs.csv"

    df_optimal.to_csv(output_csv, index=False)
    print(f"\n✓ Saved optimal configs to: {output_csv}")

    # Show sample
    print(f"\nSample optimal configurations:")
    print("-" * 70)
    print(
        df_optimal.head(10)[
            [
                kernel_col,
                "N",
                "bx" if "bx" in df_optimal.columns else "block_x",
                "by" if "by" in df_optimal.columns else "block_y",
                "exec_time",
                "speedup_vs_avg",
            ]
        ].to_string(index=False)
    )

    return df_optimal


def compare_gpus(input_csv):
    """Compare optimal configs across different GPUs"""
    df = pd.read_csv(input_csv)

    if "gpu" not in df.columns:
        print("No GPU column found in data")
        return

    kernel_col = "kernel_name" if "kernel_name" in df.columns else "kernel"

    print("\n" + "=" * 70)
    print("GPU COMPARISON")
    print("=" * 70)

    # For each (kernel, N), compare optimal config across GPUs
    differences = []

    for (kernel, n), group in df.groupby([kernel_col, "N"]):
        if group["gpu"].nunique() < 2:
            continue

        # Find optimal for each GPU
        gpu_optima = {}
        for gpu, gpu_group in group.groupby("gpu"):
            best_idx = gpu_group["exec_time"].idxmin()
            best_row = gpu_group.loc[best_idx]
            bx = best_row["bx"] if "bx" in best_row else best_row.get("block_x", 0)
            by = best_row["by"] if "by" in best_row else best_row.get("block_y", 0)
            gpu_optima[gpu] = (bx, by)

        # Check if optimal config is same across GPUs
        configs = list(gpu_optima.values())
        if len(set(configs)) > 1:  # Different optimal configs
            differences.append({"kernel": kernel, "N": n, "gpu_optima": gpu_optima})

    print(f"\nCases where optimal config differs across GPUs: {len(differences)}")
    print(f"Total (kernel, N) pairs: {df.groupby([kernel_col, 'N']).ngroups}")
    print(
        f"Agreement rate: {100 * (1 - len(differences) / df.groupby([kernel_col, 'N']).ngroups):.1f}%"
    )

    if differences:
        print("\nExample differences:")
        print("-" * 70)
        for diff in differences[:5]:
            print(f"{diff['kernel']:25s} N={diff['N']:4d}:")
            for gpu, (bx, by) in diff["gpu_optima"].items():
                print(f"  {gpu:25s}: block=({bx}, {by})")
            print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract optimal configurations for LLM/NN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract global optimal (best across all GPUs)
  python3 extract_optimal_configs.py klaraptor_enriched_data.csv
  
  # Extract optimal per GPU
  python3 extract_optimal_configs.py klaraptor_enriched_data.csv --per-gpu
  
  # Custom output file
  python3 extract_optimal_configs.py klaraptor_enriched_data.csv \\
      --output optimal_global.csv
  
  # Compare GPUs
  python3 extract_optimal_configs.py klaraptor_enriched_data.csv --compare-gpus
        """,
    )

    parser.add_argument("input_csv", help="Input CSV with all configurations")
    parser.add_argument("--output", "-o", help="Output CSV file")
    parser.add_argument(
        "--per-gpu",
        action="store_true",
        help="Find optimal config per GPU (not global optimal)",
    )
    parser.add_argument(
        "--compare-gpus",
        action="store_true",
        help="Compare optimal configs across GPUs",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("OPTIMAL CONFIG EXTRACTOR")
    print("=" * 70)
    print()

    # Extract optimal configs
    df_optimal = extract_optimal_configs(
        args.input_csv, output_csv=args.output, per_gpu=args.per_gpu
    )

    # Compare GPUs if requested
    if args.compare_gpus:
        compare_gpus(args.input_csv)

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\nTo train Neural Network:")
    print(
        f"  python3 train_nn_block_predictor.py {args.output or 'optimal_configs.csv'}"
    )

    print("\nTo fine-tune LLM:")
    print(
        f"  python3 finetune_llm_block_prediction.py {args.output or 'optimal_configs.csv'} gpt2-medium"
    )

    print("\nTo use the full dataset (including suboptimal for NN):")
    print(f"  python3 train_nn_block_predictor.py {args.input_csv}")


if __name__ == "__main__":
    main()
