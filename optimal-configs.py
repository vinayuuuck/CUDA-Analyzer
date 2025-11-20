import os
import sys
import json
import argparse
from pathlib import Path

IGNORE_LIST = {
    "fopen",
    "fclose",
    "printf",
    "fprintf",
    "fscanf",
    "fwrite",
    "fread",
    "fflush",
    "setbuf",
    "setvbuf",
    "tmpfile",
    "tmpnam",
    "fseek",
    "ftell",
    "rewind",
    "fgetpos",
    "fsetpos",
    "clearerr",
    "feof",
    "ferror",
    "perror",
    "getc",
    "getchar",
    "fgetc",
    "fgets",
    "putc",
    "putchar",
    "fputc",
    "fputs",
    "ungetc",
    "scanf",
    "sscanf",
    "vprintf",
    "vfprintf",
    "vsprintf",
    "malloc",
    "calloc",
    "realloc",
    "free",
    "abort",
    "exit",
    "atexit",
    "system",
    "getenv",
    "bsearch",
    "qsort",
    "abs",
    "div",
    "labs",
    "ldiv",
    "rand",
    "srand",
    "mblen",
    "mbtowc",
    "wctomb",
    "mbstowcs",
    "wcstombs",
    "atof",
    "atoi",
    "atol",
    "strtod",
    "strtol",
    "strtoul",
    "__inline_signbitl",
    "initstate",
    "main",  # Added main just in case
}

try:
    from main import MultiModelSuggester
    from cuda_anal import CUDAKernelAnalyzer
except ImportError:
    print("ERROR: Could not import 'main.py' or 'cuda_anal.py'.")
    print("   Make sure this script is in the same folder as your main code.")
    sys.exit(1)


def analyze_and_clean(cuda_dir, output_file):
    print(f" Starting Analysis on directory: {cuda_dir}")

    cuda_files = []
    for root, dirs, files in os.walk(cuda_dir):
        for file in files:
            if file.endswith(".cu"):
                cuda_files.append(os.path.join(root, file))

    if not cuda_files:
        print(f"❌ No .cu files found in {cuda_dir}")
        return

    print(f"   Found {len(cuda_files)} CUDA files.")

    try:
        suggester = MultiModelSuggester()
        print("✅ AI Model Loaded Successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    results = {"kernels": {}}

    valid_kernels_count = 0

    for i, cuda_file in enumerate(cuda_files, 1):
        print(f"   [{i}/{len(cuda_files)}] Analyzing {os.path.basename(cuda_file)}...")

        try:
            analyzer = CUDAKernelAnalyzer(cuda_file)
            found_kernels = analyzer.analyze()

            if not found_kernels:
                print(f"      ⚠️ No kernels found.")
                continue

            kernel_results = suggester.evaluate_fixed_Ns_and_pick(
                cuda_file, strategy="geom_mean", topk_per_N=1, verbose=False
            )

            for k_res in kernel_results:
                k_name = k_res["kernel"]

                if k_name in IGNORE_LIST or k_name.startswith("__"):
                    print(f"      🧹 Ignoring ghost kernel: {k_name}")
                    continue

                full_key = f"{os.path.basename(cuda_file)}::{k_name}"

                chosen = k_res["chosen"]
                model_recs = k_res.get("model_recommendations", {})

                ensemble_config = None
                if "ensemble" in model_recs:
                    ens = model_recs["ensemble"]
                    ensemble_config = {
                        "block_x": int(ens["block_x"]),
                        "block_y": int(ens["block_y"]),
                        "block_z": int(ens["block_z"]),
                    }

                rf_config = None
                if "random_forest" in model_recs:
                    rf = model_recs["random_forest"]
                    rf_config = {
                        "block_x": int(rf["block_x"]),
                        "block_y": int(rf["block_y"]),
                        "block_z": int(rf["block_z"]),
                    }

                results["kernels"][full_key] = {
                    "kernel_name": k_name,
                    "cuda_file": cuda_file,
                    "optimal_config": {
                        "block_x": int(chosen["block_x"]),
                        "block_y": int(chosen["block_y"]),
                        "block_z": int(chosen["block_z"]),
                    },
                    "ensemble_config": ensemble_config,
                    "random_forest_config": rf_config,
                }

                print(f"      ✅ Added {k_name}")
                if ensemble_config:
                    print(
                        f"         Ensemble DNN: ({ensemble_config['block_x']}, {ensemble_config['block_y']}, {ensemble_config['block_z']})"
                    )
                if rf_config:
                    print(
                        f"         Random Forest: ({rf_config['block_x']}, {rf_config['block_y']}, {rf_config['block_z']})"
                    )

                valid_kernels_count += 1

        except Exception as e:
            print(f"      ❌ Error analyzing file: {e}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 40)
    print(f"🎉 ANALYSIS COMPLETE")
    print(f"   Total Valid Kernels: {valid_kernels_count}")
    print(f"   Results Saved to:    {output_file}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", default="test-files-forced", help="Folder containing CUDA files"
    )
    parser.add_argument(
        "--output", default="test_suite_clean.json", help="Output JSON file"
    )
    args = parser.parse_args()

    analyze_and_clean(args.dir, args.output)
