import clang.cindex
from clang.cindex import CursorKind, TypeKind
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CUDAKernelAnalyzer:
    def __init__(self, cuda_file: str):
        self.cuda_file = cuda_file
        self.kernels = []

        # Initialize libclang - will auto-detect on macOS
        self.index = clang.cindex.Index.create()

    def parse_file(self) -> clang.cindex.TranslationUnit:
        """Parse the CUDA file using clang"""
        args = [
            "-x",
            "cuda",
            "--cuda-gpu-arch=sm_75",
            "--cuda-host-only",
            "-nocudainc",
            "-nocudalib",
            "--no-cuda-version-check",
            "-std=c++11",
            "-Wno-everything",
            "-fsyntax-only",  # Only parse, don't compile
            "-I.",  # Include current directory
        ]

        # Try with resource dir to find system headers
        import platform

        if platform.system() == "Darwin":  # macOS
            # Common paths for macOS system headers
            system_includes = [
                "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include",
                "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include",
                "/usr/local/include",
            ]
            for inc_path in system_includes:
                if os.path.exists(inc_path):
                    args.append(f"-I{inc_path}")
                    break

        tu = self.index.parse(
            self.cuda_file,
            args=args,
            options=clang.cindex.TranslationUnit.PARSE_INCOMPLETE,
        )

        if not tu:
            raise Exception("Failed to parse CUDA file")

        return tu

    def is_cuda_kernel(self, cursor) -> bool:
        """Check if a function is a CUDA kernel (__global__)"""
        tokens = list(cursor.get_tokens())
        for token in tokens:
            if token.spelling == "__global__":
                return True
        return False

    def analyze_memory_accesses(self, cursor) -> Dict:
        """Analyze memory access patterns in the kernel"""
        memory_info = {
            "global_reads": 0,
            "global_writes": 0,
            "shared_memory": False,
            "shared_arrays": [],
            "array_accesses": [],
        }

        # Get source text for simpler pattern matching
        import re
        tokens = list(cursor.get_tokens())
        source_text = " ".join([t.spelling for t in tokens])
        
        # Count array writes: pattern like "array[index] ="
        write_pattern = r'\w+\s*\[\s*[^\]]+\s*\]\s*='
        writes = re.findall(write_pattern, source_text)
        memory_info["global_writes"] = len(writes)
        
        # Count all array accesses
        array_pattern = r'\w+\s*\[\s*[^\]]+\s*\]'
        all_accesses = re.findall(array_pattern, source_text)
        
        # Reads = total accesses - writes
        memory_info["global_reads"] = max(len(all_accesses) - len(writes), 0)

        def visit_node(node):
            # Check for shared memory declarations
            if node.kind == CursorKind.VAR_DECL:
                tokens = list(node.get_tokens())
                token_strs = [t.spelling for t in tokens]
                if "__shared__" in token_strs:
                    memory_info["shared_memory"] = True
                    memory_info["shared_arrays"].append(node.spelling)

            # Collect detailed array access info
            if node.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
                memory_info["array_accesses"].append(
                    {
                        "line": node.location.line,
                        "array": (
                            list(node.get_children())[0].spelling
                            if list(node.get_children())
                            else "unknown"
                        ),
                    }
                )

            for child in node.get_children():
                visit_node(child)

        visit_node(cursor)
        return memory_info

    def analyze_thread_indexing(self, cursor) -> Dict:
        """Analyze how thread indices are used"""
        thread_info = {
            "uses_threadIdx_x": False,
            "uses_threadIdx_y": False,
            "uses_threadIdx_z": False,
            "uses_blockIdx_x": False,
            "uses_blockIdx_y": False,
            "uses_blockIdx_z": False,
            "uses_blockDim_x": False,
            "uses_blockDim_y": False,
            "uses_blockDim_z": False,
            "dimensionality": 1,
        }

        # Also get all tokens from the function to catch cases where
        # clang doesn't parse threadIdx/blockIdx as expected
        all_tokens = list(cursor.get_tokens())
        source_text = " ".join([t.spelling for t in all_tokens])
        
        # Direct text search as backup
        thread_info["uses_threadIdx_x"] = "threadIdx.x" in source_text or "threadIdx . x" in source_text
        thread_info["uses_threadIdx_y"] = "threadIdx.y" in source_text or "threadIdx . y" in source_text
        thread_info["uses_threadIdx_z"] = "threadIdx.z" in source_text or "threadIdx . z" in source_text
        thread_info["uses_blockIdx_x"] = "blockIdx.x" in source_text or "blockIdx . x" in source_text
        thread_info["uses_blockIdx_y"] = "blockIdx.y" in source_text or "blockIdx . y" in source_text
        thread_info["uses_blockIdx_z"] = "blockIdx.z" in source_text or "blockIdx . z" in source_text
        thread_info["uses_blockDim_x"] = "blockDim.x" in source_text or "blockDim . x" in source_text
        thread_info["uses_blockDim_y"] = "blockDim.y" in source_text or "blockDim . y" in source_text
        thread_info["uses_blockDim_z"] = "blockDim.z" in source_text or "blockDim . z" in source_text

        # Determine dimensionality
        if thread_info["uses_threadIdx_z"] or thread_info["uses_blockIdx_z"]:
            thread_info["dimensionality"] = 3
        elif thread_info["uses_threadIdx_y"] or thread_info["uses_blockIdx_y"]:
            thread_info["dimensionality"] = 2
        else:
            thread_info["dimensionality"] = 1

        return thread_info

    def count_operations(self, cursor) -> Dict:
        """Count arithmetic and logical operations"""
        ops = {"arithmetic": 0, "memory": 0, "control_flow": 0, "loops": 0}

        def visit_node(node):
            if node.kind == CursorKind.BINARY_OPERATOR:
                ops["arithmetic"] += 1
            elif node.kind == CursorKind.UNARY_OPERATOR:
                ops["arithmetic"] += 1
            elif node.kind in [
                CursorKind.ARRAY_SUBSCRIPT_EXPR,
                CursorKind.MEMBER_REF_EXPR,
            ]:
                ops["memory"] += 1
            elif node.kind in [CursorKind.IF_STMT, CursorKind.SWITCH_STMT]:
                ops["control_flow"] += 1
            elif node.kind in [
                CursorKind.FOR_STMT,
                CursorKind.WHILE_STMT,
                CursorKind.DO_STMT,
            ]:
                ops["loops"] += 1

            for child in node.get_children():
                visit_node(child)

        visit_node(cursor)
        return ops

    def extract_kernel_parameters(self, cursor) -> List[Dict]:
        """Extract kernel function parameters"""
        params = []
        for arg in cursor.get_arguments():
            param_info = {
                "name": arg.spelling,
                "type": arg.type.spelling,
                "is_pointer": arg.type.kind == TypeKind.POINTER,
                "is_const": arg.type.is_const_qualified(),
            }
            params.append(param_info)
        return params

    def analyze_kernel(self, cursor) -> Dict:
        """Perform complete analysis of a kernel"""
        kernel_info = {
            "name": cursor.spelling,
            "location": {
                "file": (
                    cursor.location.file.name if cursor.location.file else "unknown"
                ),
                "line": cursor.location.line,
            },
            "parameters": self.extract_kernel_parameters(cursor),
            "thread_usage": self.analyze_thread_indexing(cursor),
            "memory_access": self.analyze_memory_accesses(cursor),
            "operations": self.count_operations(cursor),
        }

        # Calculate complexity metrics
        kernel_info["metrics"] = {
            "compute_intensity": kernel_info["operations"]["arithmetic"]
            / max(
                kernel_info["memory_access"]["global_reads"]
                + kernel_info["memory_access"]["global_writes"],
                1,
            ),
            "has_shared_memory": kernel_info["memory_access"]["shared_memory"],
            "dimensionality": kernel_info["thread_usage"]["dimensionality"],
        }

        return kernel_info

    def analyze(self) -> List[Dict]:
        """Main analysis function"""
        try:
            tu = self.parse_file()
        except Exception as e:
            print(f"Warning: Parse failed ({e}), trying fallback method...")
            # Fallback: use regex to find kernels
            return self._fallback_analysis()

        def visit_cursor(cursor):
            if cursor.kind == CursorKind.FUNCTION_DECL:
                if self.is_cuda_kernel(cursor):
                    try:
                        kernel_info = self.analyze_kernel(cursor)
                        self.kernels.append(kernel_info)
                    except Exception as e:
                        print(
                            f"Warning: Could not analyze kernel {cursor.spelling}: {e}"
                        )

            for child in cursor.get_children():
                visit_cursor(child)

        visit_cursor(tu.cursor)
        return self.kernels

    def _fallback_analysis(self) -> List[Dict]:
        """Fallback method using regex when clang parsing fails"""
        import re

        with open(self.cuda_file, "r") as f:
            content = f.read()

        # Find __global__ functions with their bodies
        # Match: __global__ void kernelName(...) { ... }
        pattern = r"__global__\s+\w+\s+(\w+)\s*\([^)]*\)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}"
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            kernel_name = match.group(1)
            kernel_body = match.group(2) if len(match.groups()) > 1 else match.group(0)
            
            # Analyze this specific kernel's body
            uses_threadIdx_x = "threadIdx.x" in kernel_body
            uses_threadIdx_y = "threadIdx.y" in kernel_body
            uses_threadIdx_z = "threadIdx.z" in kernel_body
            uses_blockIdx_x = "blockIdx.x" in kernel_body
            uses_blockIdx_y = "blockIdx.y" in kernel_body
            uses_blockIdx_z = "blockIdx.z" in kernel_body
            
            # Determine dimensionality from this kernel only
            if uses_threadIdx_z or uses_blockIdx_z:
                dimensionality = 3
            elif uses_threadIdx_y or uses_blockIdx_y:
                dimensionality = 2
            else:
                dimensionality = 1
            
            # Count operations in this kernel body
            array_accesses = kernel_body.count("[")
            arithmetic_ops = kernel_body.count("+") + kernel_body.count("-") + kernel_body.count("*") + kernel_body.count("/")
            
            # Estimate reads vs writes (assignments with = typically mean writes)
            # Count array accesses on left side of = as writes
            write_count = len(re.findall(r'\w+\[[^\]]+\]\s*=', kernel_body))
            read_count = max(array_accesses - write_count, 1)
            
            has_shared = "__shared__" in kernel_body
            
            # Create minimal kernel info
            kernel_info = {
                "name": kernel_name,
                "location": {
                    "file": self.cuda_file,
                    "line": content[: match.start()].count("\n") + 1,
                },
                "parameters": [],
                "thread_usage": {
                    "uses_threadIdx_x": uses_threadIdx_x,
                    "uses_threadIdx_y": uses_threadIdx_y,
                    "uses_threadIdx_z": uses_threadIdx_z,
                    "uses_blockIdx_x": uses_blockIdx_x,
                    "uses_blockIdx_y": uses_blockIdx_y,
                    "uses_blockIdx_z": uses_blockIdx_z,
                    "dimensionality": dimensionality,
                },
                "memory_access": {
                    "global_reads": read_count,
                    "global_writes": max(write_count, 1),
                    "shared_memory": has_shared,
                    "shared_arrays": [],
                },
                "operations": {
                    "arithmetic": arithmetic_ops,
                    "memory": array_accesses,
                    "control_flow": kernel_body.count("if"),
                    "loops": kernel_body.count("for") + kernel_body.count("while"),
                },
            }
            
            # Calculate compute intensity
            total_memory_ops = read_count + max(write_count, 1)
            kernel_info["metrics"] = {
                "compute_intensity": arithmetic_ops / total_memory_ops if total_memory_ops > 0 else 1.0,
                "has_shared_memory": has_shared,
                "dimensionality": dimensionality,
            }
            
            self.kernels.append(kernel_info)

        return self.kernels

    def generate_report(self, output_file: Optional[str] = None):
        """Generate a detailed analysis report"""
        report = {
            "source_file": self.cuda_file,
            "num_kernels": len(self.kernels),
            "kernels": self.kernels,
        }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_file}")

        return report

    def print_summary(self):
        """Print a human-readable summary"""
        print(f"\n{'='*60}")
        print(f"CUDA Kernel Analysis Report")
        print(f"{'='*60}")
        print(f"Source file: {self.cuda_file}")
        print(f"Kernels found: {len(self.kernels)}\n")

        for i, kernel in enumerate(self.kernels, 1):
            print(f"Kernel #{i}: {kernel['name']}")
            print(f"  Location: Line {kernel['location']['line']}")
            print(f"  Parameters: {len(kernel['parameters'])}")
            print(f"  Dimensionality: {kernel['thread_usage']['dimensionality']}D")
            print(
                f"  Memory accesses: {kernel['memory_access']['global_reads']} reads, {kernel['memory_access']['global_writes']} writes"
            )
            print(
                f"  Shared memory: {'Yes' if kernel['memory_access']['shared_memory'] else 'No'}"
            )
            print(f"  Compute intensity: {kernel['metrics']['compute_intensity']:.2f}")
            print(f"  Operations: {kernel['operations']}")
            print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 cuda_analyzer.py <cuda_file.cu> [output.json]")
        sys.exit(1)

    cuda_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(cuda_file):
        print(f"Error: File '{cuda_file}' not found")
        sys.exit(1)

    analyzer = CUDAKernelAnalyzer(cuda_file)
    analyzer.analyze()
    analyzer.print_summary()

    if output_file:
        analyzer.generate_report(output_file)


if __name__ == "__main__":
    main()
