import re
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict


def parse_cuda_results_file(filepath: str) -> List[Dict]:
    """
    Parse a cuda_results_*.txt file

    Format:
    checking log point [N bx by bz]
    [trace: n=N, bx=X, by=Y, elapsed_KERNEL=TIME (ms)] ... PASS
    or
    [trace: n=N, bx=X, by=Y, elapsed_KERNEL1=TIME1 (ms), elapsed_KERNEL2=TIME2 (ms)] ... PASS

    Returns list of dicts with: kernel, N, bx, by, bz, exec_time, gpu
    """
    results = []

    with open(filepath, "r") as f:
        content = f.read()

    # Extract GPU name from first line
    gpu_match = re.search(r"\[running on device \d+: ([^\]]+)\]", content)
    gpu_name = gpu_match.group(1) if gpu_match else "unknown"

    # Clean up GPU name
    gpu_name = gpu_name.replace(" ", "_").lower()

    # Find all log points and their corresponding traces
    # Pattern: checking log point [N bx by bz]
    log_points = re.finditer(
        r"checking log point \[(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]", content
    )

    for match in log_points:
        n = int(match.group(1))
        bx = int(match.group(2))
        by = int(match.group(3))
        bz = int(match.group(4))

        # Find the trace line immediately after this log point
        trace_start = match.end()
        trace_text = content[
            trace_start : trace_start + 800
        ]  # Look ahead 800 chars for multiple kernels

        # Find the trace line
        trace_line_match = re.search(r"\[trace:[^\]]+\]", trace_text)
        if not trace_line_match:
            continue

        trace_line = trace_line_match.group(0)

        # Extract ALL elapsed_KERNEL=TIME pairs from the trace line
        # Pattern: elapsed_KERNEL_NAME=TIME
        kernel_timings = re.findall(r"elapsed_([^=]+)=([\d.]+)\s*\(ms\)", trace_line)

        # Create a result entry for EACH kernel in this trace
        for kernel_name, exec_time in kernel_timings:
            kernel_name = kernel_name.strip()
            exec_time = float(exec_time)

            results.append(
                {
                    "kernel_name": kernel_name,
                    "N": n,
                    "block_x": bx,
                    "block_y": by,
                    "block_z": bz,
                    "total_threads": bx * by * bz,
                    "exec_time": exec_time,
                    "gpu": gpu_name,
                }
            )

    return results


def parse_kernel_table_file(filepath: str) -> List[Dict]:
    """
    Parse kernel_*_table_*.txt files

    Format:
    [N: 128]
    [b0: 2]
    [b1: 16]
    [Exec_cycles_app: 22608601.2254]
    [occupancy: 0.31]
    [Threads_per_block: 32]
    ...

    Returns list of dicts with: kernel_name, N, bx, by, exec_time, occupancy, etc.
    """
    results = []

    # Extract kernel name from filename
    # Format: kernel_KERNELNAME_table_GPU.txt
    filename = Path(filepath).name
    match = re.match(r"kernel_(.+)_table_(.+)\.txt", filename)
    if not match:
        return results

    kernel_name = match.group(1)
    gpu_name = match.group(2)

    with open(filepath, "r") as f:
        content = f.read()

    # Split into individual entries (each starts with [N: ...])
    entries = re.split(r"(?=\[N: \d+\])", content)

    gpu_freq_ghz = 1.5  # Default GPU frequency for cycle-to-time conversion

    for entry in entries:
        if not entry.strip():
            continue

        # Extract all fields from this entry
        n_match = re.search(r"\[N: (\d+)\]", entry)
        b0_match = re.search(r"\[b0: (\d+)\]", entry)
        b1_match = re.search(r"\[b1: (\d+)\]", entry)
        exec_cycles_match = re.search(r"\[Exec_cycles_app: ([\d.]+)\]", entry)
        occupancy_match = re.search(r"\[occupancy: ([\d.]+)\]", entry)
        threads_match = re.search(r"\[Threads_per_block: (\d+)\]", entry)

        if n_match and b0_match and b1_match and exec_cycles_match:
            n = int(n_match.group(1))
            bx = int(b0_match.group(1))
            by = int(b1_match.group(1))
            exec_cycles = float(exec_cycles_match.group(1))
            occupancy = float(occupancy_match.group(1)) if occupancy_match else 0.0
            threads_per_block = (
                int(threads_match.group(1)) if threads_match else bx * by
            )

            # Convert cycles to milliseconds
            exec_time = (exec_cycles / (gpu_freq_ghz * 1e9)) * 1000  # Convert to ms

            results.append(
                {
                    "kernel_name": kernel_name,
                    "N": n,
                    "block_x": bx,
                    "block_y": by,
                    "block_z": 1,
                    "total_threads": threads_per_block,
                    "exec_time": exec_time,
                    "gpu": gpu_name,
                    "occupancy": occupancy,
                }
            )

    return results


def parse_kernel_results_file(filepath: str) -> List[Dict]:
    """
    Parse kernel_*_results_*.txt files

    Format:
    [ N:128, b0:  1, b1: 32, ..., Exec_cycles_app:977131.6052, occupancy:0.09, ... ]

    Returns list of dicts with: kernel_name, N, bx, by, exec_time, occupancy, etc.
    """
    results = []

    # Extract kernel name from filename
    filename = Path(filepath).name
    match = re.match(r"kernel_(.+)_results_(.+)\.txt", filename)
    if not match:
        return results

    kernel_name = match.group(1)
    gpu_name = match.group(2)

    with open(filepath, "r") as f:
        content = f.read()

    # Pattern to match each entry
    # [ N:128, b0:  1, b1: 32, ..., Exec_cycles_app:977131.6052, ... ]
    entries = re.findall(
        r"\[\s*N:\s*(\d+),\s*b0:\s*(\d+)\s*,\s*b1:\s*(\d+)\s*,.*?Exec_cycles_app:\s*([\d.]+).*?occupancy:\s*([\d.]+)",
        content,
        re.DOTALL,
    )

    gpu_freq_ghz = 1.5  # Default GPU frequency

    for n, b0, b1, exec_cycles, occupancy in entries:
        n = int(n)
        bx = int(b0)
        by = int(b1)
        exec_cycles = float(exec_cycles)
        occupancy = float(occupancy)

        # Convert cycles to milliseconds
        exec_time = (exec_cycles / (gpu_freq_ghz * 1e9)) * 1000

        results.append(
            {
                "kernel_name": kernel_name,
                "N": n,
                "block_x": bx,
                "block_y": by,
                "block_z": 1,
                "total_threads": bx * by,
                "exec_time": exec_time,
                "gpu": gpu_name,
                "occupancy": occupancy,
            }
        )

    return results


def extract_all_klaraptor_data(klaraptor_dir: str) -> pd.DataFrame:
    """
    Extract ALL data from KLARAPTORresults directory

    Args:
        klaraptor_dir: Path to KLARAPTORresults directory

    Returns:
        DataFrame with all timing data
    """
    all_data = []

    klaraptor_path = Path(klaraptor_dir)

    # Find all benchmark directories
    benchmark_dirs = [
        d
        for d in klaraptor_path.iterdir()
        if d.is_dir() and d.name.startswith("polybench_")
    ]

    print("=" * 70)
    print("KLARAPTOR DATA EXTRACTION")
    print("=" * 70)
    print()
    print(f"Found {len(benchmark_dirs)} benchmark directories")
    print()

    for bench_dir in sorted(benchmark_dirs):
        bench_name = bench_dir.name
        print(f"Processing {bench_name}...")

        samples_before = len(all_data)

        # Strategy 1: Try cuda_results_*.txt files first (most comprehensive)
        result_files = list(bench_dir.glob("cuda_results_*.txt"))
        if result_files:
            for result_file in result_files:
                try:
                    data = parse_cuda_results_file(str(result_file))
                    if data:
                        print(f"  ✓ {result_file.name}: {len(data)} samples")
                        all_data.extend(data)
                except Exception as e:
                    print(f"  ✗ {result_file.name}: {e}")

        # Strategy 2: If no cuda_results data, try kernel_*_table_*.txt files
        if len(all_data) == samples_before:
            table_files = list(bench_dir.glob("kernel_*_table_*.txt"))
            if table_files:
                for table_file in table_files:
                    try:
                        data = parse_kernel_table_file(str(table_file))
                        if data:
                            print(
                                f"  ✓ {table_file.name}: {len(data)} samples (from table)"
                            )
                            all_data.extend(data)
                    except Exception as e:
                        print(f"  ✗ {table_file.name}: {e}")

        # Strategy 3: If still no data, try kernel_*_results_*.txt files
        if len(all_data) == samples_before:
            results_files = list(bench_dir.glob("kernel_*_results_*.txt"))
            if results_files:
                for results_file in results_files:
                    try:
                        data = parse_kernel_results_file(str(results_file))
                        if data:
                            print(
                                f"  ✓ {results_file.name}: {len(data)} samples (from results)"
                            )
                            all_data.extend(data)
                    except Exception as e:
                        print(f"  ✗ {results_file.name}: {e}")

        if len(all_data) == samples_before:
            print(f"  ⚠  No data extracted from {bench_name}")

    print()
    print("=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    if not all_data:
        print("ERROR: No data extracted!")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    print(f"Total samples extracted: {len(df)}")
    print(f"Unique kernels: {df['kernel_name'].nunique()}")
    print(f"Kernel breakdown:")
    for kernel, count in df["kernel_name"].value_counts().head(20).items():
        print(f"  {kernel:30s}: {count:4d} samples")
    print(f"Unique GPUs: {df['gpu'].nunique()}")
    print(f"GPUs: {sorted(df['gpu'].unique())}")
    print(f"Problem sizes (N): {sorted(df['N'].unique())}")
    print()

    return df


def add_kernel_features(
    df: pd.DataFrame, polybench_dir: str = "PolybenchCUDA"
) -> pd.DataFrame:
    """
    Add static kernel features by analyzing source code

    This is similar to what cuda_anal.py does
    """
    from cuda_anal import CUDAKernelAnalyzer

    # Map kernel names to their source files
    kernel_to_file = {
        "Convolution2D_kernel": "stencils/convolution-2d/2DConvolution.cu",
        "convolution3D_kernel": "stencils/convolution-3d/3DConvolution.cu",
        "mm2_kernel1": "linear-algebra/kernels/2mm/2mm.cu",
        "mm2_kernel2": "linear-algebra/kernels/2mm/2mm.cu",
        "mm3_kernel1": "linear-algebra/kernels/3mm/3mm.cu",
        "mm3_kernel2": "linear-algebra/kernels/3mm/3mm.cu",
        "mm3_kernel3": "linear-algebra/kernels/3mm/3mm.cu",
        "atax_kernel1": "linear-algebra/kernels/atax/atax.cu",
        "atax_kernel2": "linear-algebra/kernels/atax/atax.cu",
        "bicg_kernel1": "linear-algebra/kernels/bicg/bicg.cu",
        "bicg_kernel2": "linear-algebra/kernels/bicg/bicg.cu",
        "gemm_kernel": "linear-algebra/kernels/gemm/gemm.cu",
        "gesummv_kernel": "linear-algebra/kernels/gesummv/gesummv.cu",
        "mvt_kernel1": "linear-algebra/kernels/mvt/mvt.cu",
        "mvt_kernel2": "linear-algebra/kernels/mvt/mvt.cu",
        "syr2k_kernel": "linear-algebra/kernels/syr2k/syr2k.cu",
        "syrk_kernel": "linear-algebra/kernels/syrk/syrk.cu",
        "gramschmidt_kernel1": "linear-algebra/solvers/gramschmidt/gramschmidt.cu",
        "gramschmidt_kernel2": "linear-algebra/solvers/gramschmidt/gramschmidt.cu",
        "gramschmidt_kernel3": "linear-algebra/solvers/gramschmidt/gramschmidt.cu",
        "corr_kernel": "datamining/correlation/correlation.cu",
        "mean_kernel": "datamining/correlation/correlation.cu",
        "std_kernel": "datamining/correlation/correlation.cu",
        "reduce_kernel": "datamining/correlation/correlation.cu",
        "covar_kernel": "datamining/covariance/covariance.cu",
        "fdtd_step1_kernel": "stencils/fdtd-2d/fdtd2d.cu",
        "fdtd_step2_kernel": "stencils/fdtd-2d/fdtd2d.cu",
        "fdtd_step3_kernel": "stencils/fdtd-2d/fdtd2d.cu",
    }

    # Analyze each kernel once
    kernel_features = {}

    print("\nAnalyzing kernel source code...")

    for kernel_name, rel_path in kernel_to_file.items():
        full_path = os.path.join(polybench_dir, rel_path)

        if not os.path.exists(full_path):
            print(f"  Warning: {full_path} not found")
            continue

        try:
            parser = CUDAKernelAnalyzer(full_path)
            parser.analyze()

            # Find the specific kernel
            kernel_info = None
            for k in parser.kernels:
                # Match kernel name flexibly
                k_name_normalized = k["name"].replace("_", "").lower()
                kernel_name_normalized = kernel_name.replace("_", "").lower()
                if (
                    kernel_name_normalized in k_name_normalized
                    or k_name_normalized in kernel_name_normalized
                ):
                    kernel_info = k
                    break
            if kernel_info:
                # Extract compute intensity - check both old and new key names
                compute_intensity = kernel_info["metrics"].get(
                    "compute_intensity"
                ) or kernel_info["metrics"].get("compute_intensity_flops_per_byte", 0)

                kernel_features[kernel_name] = {
                    "dimensionality": kernel_info["thread_usage"]["dimensionality"],
                    "compute_intensity": compute_intensity,
                    "has_shared_memory": int(
                        kernel_info["memory_access"]["shared_memory"]
                    ),
                    "global_reads": kernel_info["memory_access"]["global_reads"],
                    "global_writes": kernel_info["memory_access"]["global_writes"],
                    "arithmetic_ops": kernel_info["operations"]["arithmetic"],
                    "memory_ops": kernel_info["operations"]["memory"],
                    # New metrics from cuda_anal.py
                    "control_flow_ops": kernel_info["operations"].get(
                        "control_flow", 0
                    ),
                    "loop_ops": kernel_info["operations"].get("loops", 0),
                    "uses_syncthreads": int(
                        kernel_info["metrics"].get("uses_syncthreads", False)
                    ),
                    "estimated_flops": kernel_info["metrics"].get("estimated_flops", 0),
                    "estimated_memory_bytes": kernel_info["metrics"].get(
                        "estimated_memory_bytes", 0
                    ),
                    # Thread dimension usage
                    "uses_threadIdx_x": int(
                        kernel_info["thread_usage"].get("uses_threadIdx_x", False)
                    ),
                    "uses_threadIdx_y": int(
                        kernel_info["thread_usage"].get("uses_threadIdx_y", False)
                    ),
                    "uses_threadIdx_z": int(
                        kernel_info["thread_usage"].get("uses_threadIdx_z", False)
                    ),
                    "uses_blockIdx_x": int(
                        kernel_info["thread_usage"].get("uses_blockIdx_x", False)
                    ),
                    "uses_blockIdx_y": int(
                        kernel_info["thread_usage"].get("uses_blockIdx_y", False)
                    ),
                    "uses_blockIdx_z": int(
                        kernel_info["thread_usage"].get("uses_blockIdx_z", False)
                    ),
                    "uses_blockDim_x": int(
                        kernel_info["thread_usage"].get("uses_blockDim_x", False)
                    ),
                    "uses_blockDim_y": int(
                        kernel_info["thread_usage"].get("uses_blockDim_y", False)
                    ),
                    "uses_blockDim_z": int(
                        kernel_info["thread_usage"].get("uses_blockDim_z", False)
                    ),
                }
                print(f"  ✓ {kernel_name}")
            else:
                print(f"  ✗ {kernel_name} - kernel not found in file")

        except Exception as e:
            print(f"  ✗ {kernel_name} - {e}")

    # Add features to dataframe (note: column is 'kernel' not 'kernel_name' after rename)
    feature_columns = [
        "dimensionality",
        "compute_intensity",
        "has_shared_memory",
        "global_reads",
        "global_writes",
        "arithmetic_ops",
        "memory_ops",
        # New metrics
        "control_flow_ops",
        "loop_ops",
        "uses_syncthreads",
        "estimated_flops",
        "estimated_memory_bytes",
        # Thread dimension usage
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

    for feature in feature_columns:
        df[feature] = df["kernel"].map(
            lambda k: kernel_features.get(k, {}).get(feature, 0)
        )

    print(
        f"\n✓ Added {len(feature_columns)} features for {len(kernel_features)} kernels"
    )

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract ALL KLARAPTOR data for neural network training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        This extracts MUCH more data than your current enriched CSV:
        - All block configurations tested (not just optimal)
        - Multiple GPUs (c2075, gtx1080ti, rtx2070super)
        - All data sizes (32, 64, 128, 256, 512, 1024, 2048)
        
        This gives you thousands of training samples instead of hundreds!

        Examples:
        # Extract all timing data
        python3 extract_all_klaraptor_data.py KLARAPTORresults
        
        # Also add kernel features from source code
        python3 extract_all_klaraptor_data.py KLARAPTORresults --add-features
        
        # Specify output file
        python3 extract_klaraptor_data.py KLARAPTORresults \\
            --output full_klaraptor_data.csv
        """,
    )

    parser.add_argument("klaraptor_dir", help="Path to KLARAPTORresults directory")
    parser.add_argument(
        "--add-features",
        action="store_true",
        help="Add kernel features by analyzing source code",
    )
    parser.add_argument(
        "--polybench-dir",
        default="PolybenchCUDA",
        help="Path to PolybenchCUDA directory (for feature extraction)",
    )
    parser.add_argument(
        "--output", "-o", default="klaraptor_enriched_data.csv", help="Output CSV file"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("KLARAPTOR DATA EXTRACTION")
    print("=" * 70)
    print()

    # Extract timing data
    df = extract_all_klaraptor_data(args.klaraptor_dir)

    if len(df) == 0:
        print("\nERROR: No data extracted!")
        return

    # Rename columns to match expected format
    df = df.rename(
        columns={
            "kernel_name": "kernel",
            "block_x": "bx",
            "block_y": "by",
            "block_z": "bz",
        }
    )

    # Ensure all required columns exist
    if "bz" not in df.columns:
        df["bz"] = 1
    if "total_threads" not in df.columns:
        df["total_threads"] = df["bx"] * df["by"] * df["bz"]

    # Add kernel features if requested
    if args.add_features:
        if not os.path.exists(args.polybench_dir):
            print(f"\nWarning: {args.polybench_dir} not found")
            print("Skipping feature extraction")
        else:
            df = add_kernel_features(df, args.polybench_dir)

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\n✓ Saved {len(df)} samples to {args.output}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    print(f"\nTotal samples: {len(df)}")
    print(f"Unique kernels: {df['kernel'].nunique()}")
    print(f"Unique GPUs: {df['gpu'].nunique()}")

    print(f"\nKernels:")
    for kernel, count in df["kernel"].value_counts().items():
        print(f"  {kernel}: {count} samples")

    print(f"\nGPUs:")
    for gpu, count in df["gpu"].value_counts().items():
        print(f"  {gpu}: {count} samples")

    print(f"\nData sizes: {sorted(df['N'].unique())}")

    print(f"\nBlock configurations tested per kernel:")
    configs_per_kernel = df.groupby("kernel").size()
    print(f"  Mean: {configs_per_kernel.mean():.0f}")
    print(f"  Min: {configs_per_kernel.min()}")
    print(f"  Max: {configs_per_kernel.max()}")

    # Show sample
    print(f"\nSample data:")
    print(df.head(10).to_string())

    print("\n" + "=" * 70)
    print("READY FOR TRAINING")
    print("=" * 70)
    print(f"\nTo add kernel features and prepare for ML:")
    print(f"  python3 enrich_data_correct.py")
    print(f"\nThen train the model:")
    print(f"  python3 train_model_fixed.py")


if __name__ == "__main__":
    main()
