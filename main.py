import sys
import os
import subprocess
import re
from pathlib import Path
from typing import Dict, List

try:
    from cuda_anal import CUDAKernelAnalyzer
except ImportError:
    print("Error: cuda_anal.py not found")
    sys.exit(1)


class CodeInstrumenter:
    """Instruments CUDA source code by inserting optimization calls"""

    def __init__(self, cuda_file: str):
        self.cuda_file = cuda_file
        self.source_code = ""
        self.instrumented_code = ""

    def read_source(self) -> str:
        """Read the original source code"""
        with open(self.cuda_file, "r") as f:
            self.source_code = f.read()
        return self.source_code

    def insert_optimization_calls(self, kernel_info: List[Dict]) -> str:
        """
        Insert printf statements before kernel launches
        This demonstrates the instrumentation concept
        """
        code = self.source_code

        # Find kernel launch patterns: kernelName<<<grid, block>>>
        # We'll insert a printf before each one

        for kernel in kernel_info:
            kernel_name = kernel["name"]

            # Pattern to find kernel launches
            # Matches: kernelName<<<...>>>(...);
            pattern = rf"({kernel_name}\s*<<<[^>]+>>>)"

            # Replacement: insert printf before the launch
            replacement = (
                f"// INSTRUMENTATION: Optimizing {kernel_name}\n"
                f'    printf("ORC: Optimizing kernel {kernel_name} (dim=%dD, compute=%.2f)\\n", '
                f'{kernel["thread_usage"]["dimensionality"]}, '
                f'{kernel["metrics"]["compute_intensity"]:.2f}f);\n'
                f"    \\1"
            )

            code = re.sub(pattern, replacement, code)

        self.instrumented_code = code
        return code

    def save_instrumented_code(self, output_file: str) -> str:
        """Save the instrumented code to a file"""
        with open(output_file, "w") as f:
            f.write(self.instrumented_code)
        return output_file

    def compile_code(self, source_file: str, output_binary: str) -> bool:
        """Compile the instrumented code"""

        # Try nvcc first (if CUDA is available), then fall back to clang/gcc
        compilers = [
            (["nvcc", "-o", output_binary, source_file], "NVCC"),
            (
                [
                    "clang++",
                    "-x",
                    "c++",
                    "-o",
                    output_binary,
                    source_file,
                    "-Wno-everything",
                    "-D__CUDACC__",
                    "-D__global__=",
                    "-D__device__=",
                    "-D__host__=",
                    "-Ddim3=int",
                ],
                "Clang++",
            ),
            (
                [
                    "g++",
                    "-x",
                    "c++",
                    "-o",
                    output_binary,
                    source_file,
                    "-w",
                    "-D__CUDACC__",
                    "-D__global__=",
                    "-D__device__=",
                    "-D__host__=",
                    "-Ddim3=int",
                ],
                "G++",
            ),
        ]

        for cmd, compiler_name in compilers:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    print(f"✓ Compiled with {compiler_name}: {output_binary}")
                    return True

            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
            except Exception as e:
                continue

        print(f"✗ Compilation failed with all compilers")
        return False


class OrcMain:
    """Main orchestrator"""

    def __init__(self, cuda_file: str, output_dir: str = "output"):
        self.cuda_file = cuda_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.analyzer = None
        self.instrumenter = None
        self.kernel_info = []

    def run_analysis(self) -> bool:
        """Analyze CUDA kernels"""
        print("Phase 1: Analyzing kernels...")

        try:
            self.analyzer = CUDAKernelAnalyzer(self.cuda_file)
            self.kernel_info = self.analyzer.analyze()

            if not self.kernel_info:
                print("✗ No kernels found")
                return False

            print(f"✓ Found {len(self.kernel_info)} kernel(s)")

            # Save analysis
            analysis_file = self.output_dir / "analysis.json"
            self.analyzer.generate_report(str(analysis_file))

            return True

        except Exception as e:
            print(f"✗ Analysis failed: {e}")
            return False

    def run_instrumentation(self) -> bool:
        """Instrument the code"""
        print("\nPhase 2: Instrumenting code...")

        try:
            self.instrumenter = CodeInstrumenter(self.cuda_file)
            self.instrumenter.read_source()

            # Insert optimization calls
            instrumented = self.instrumenter.insert_optimization_calls(self.kernel_info)

            # Save instrumented code
            base_name = Path(self.cuda_file).stem
            instrumented_file = self.output_dir / f"{base_name}_instrumented.cu"
            self.instrumenter.save_instrumented_code(str(instrumented_file))

            print(f"✓ Instrumented code saved: {instrumented_file}")

            # Show what was inserted
            print("\nInstrumentation summary:")
            for kernel in self.kernel_info:
                print(f"  • {kernel['name']}: Added optimization call")

            return True

        except Exception as e:
            print(f"✗ Instrumentation failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def compile_instrumented_code(self) -> bool:
        """Compile the instrumented code"""
        print("\nPhase 3: Compiling instrumented code...")

        try:
            base_name = Path(self.cuda_file).stem
            instrumented_file = self.output_dir / f"{base_name}_instrumented.cu"
            output_binary = self.output_dir / f"{base_name}_instrumented"

            if not instrumented_file.exists():
                print(f"✗ Instrumented file not found: {instrumented_file}")
                return False

            success = self.instrumenter.compile_code(
                str(instrumented_file), str(output_binary)
            )

            if success and output_binary.exists():
                print(f"\n✓ You can run: ./{output_binary}")
                return True
            else:
                print("\n⚠ Compilation not successful (this is OK for demo)")
                print("  The instrumented source code is ready in output/")
                return True  # Still consider it a success

        except Exception as e:
            print(f"⚠ Compilation error: {e}")
            print("  (This is expected without CUDA runtime)")
            return True  # Still OK for demo

    def print_summary(self):
        """Print final summary"""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for i, kernel in enumerate(self.kernel_info, 1):
            print(f"\n{i}. {kernel['name']}")
            print(f"   Dimensionality: {kernel['thread_usage']['dimensionality']}D")
            print(f"   Compute Intensity: {kernel['metrics']['compute_intensity']:.2f}")
            print(
                f"   Memory Accesses: {kernel['memory_access']['global_reads']}R / {kernel['memory_access']['global_writes']}W"
            )
            print(
                f"   Shared Memory: {'Yes' if kernel['memory_access']['shared_memory'] else 'No'}"
            )

        print(f"\n✓ Results saved to: {self.output_dir}/")
        print(f"  - analysis.json: Detailed kernel metrics")

        base_name = Path(self.cuda_file).stem
        instrumented_file = self.output_dir / f"{base_name}_instrumented.cu"
        if instrumented_file.exists():
            print(f"  - {instrumented_file.name}: Instrumented source code")
            print(f"\nView instrumented code:")
            print(f"  cat {instrumented_file}")


def main():
    print("CUDA Kernel Optimizer - Code Instrumentation Demo\n")

    if len(sys.argv) < 2:
        print("Usage: python3 main.py <cuda_file.cu> [output_dir]")
        print("\nExample:")
        print("  python3 main.py vec_add.cu")
        sys.exit(1)

    cuda_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    if not os.path.exists(cuda_file):
        print(f"Error: '{cuda_file}' not found")
        sys.exit(1)

    # Run the pipeline
    tool = OrcMain(cuda_file, output_dir)

    if not tool.run_analysis():
        sys.exit(1)

    if not tool.run_instrumentation():
        sys.exit(1)

    tool.compile_instrumented_code()
    tool.print_summary()

    print()


if __name__ == "__main__":
    main()
