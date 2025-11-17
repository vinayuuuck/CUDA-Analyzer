import clang.cindex
from clang.cindex import CursorKind, TypeKind
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import re


class CUDAKernelAnalyzer:
    def __init__(self, cuda_file: str):
        self.cuda_file = cuda_file
        self.kernels = []
        self.index = clang.cindex.Index.create()

    def parse_file(self) -> clang.cindex.TranslationUnit:
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
            "-fsyntax-only",
            "-I.",
        ]

        import platform

        if platform.system() == "Darwin":
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

    def _get_source_from_extent(self, extent) -> str:
        try:
            start_line = extent.start.line
            end_line = extent.end.line
            with open(self.cuda_file, "r") as f:
                lines = f.readlines()
            return "".join(lines[max(0, start_line - 1) : end_line])
        except Exception:
            with open(self.cuda_file, "r") as f:
                return f.read()

    def is_cuda_kernel(self, cursor) -> bool:
        if cursor.kind != CursorKind.FUNCTION_DECL:
            return False

        tokens = [t.spelling for t in cursor.get_tokens()]
        if "__global__" in tokens:
            return True

        for ch in cursor.get_children():
            try:
                child_tokens = [t.spelling for t in ch.get_tokens()]
                if "__global__" in child_tokens:
                    return True
            except Exception:
                pass

        try:
            src = self._get_source_from_extent(cursor.extent)
            if "__global__" in src.split("\n", 1)[0]:
                return True
        except Exception:
            pass

        return False

    def _traverse_for_memory(self, node, memory_info: Dict):
        if node.kind == CursorKind.VAR_DECL:
            try:
                token_strs = [t.spelling for t in node.get_tokens()]
                if "__shared__" in token_strs:
                    memory_info["shared_memory"] = True
                    memory_info["shared_arrays"].append(node.spelling)
            except Exception:
                pass

        if node.kind == CursorKind.CALL_EXPR:
            try:
                tok = [t.spelling for t in node.get_tokens()]
                if any("__syncthreads" == s or "syncthreads" in s for s in tok):
                    memory_info["uses_syncthreads"] = True
            except Exception:
                pass

        if node.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            try:
                children = list(node.get_children())
                arr = children[0].spelling if children else "unknown"
                memory_info["array_accesses"].append(
                    {
                        "line": node.location.line,
                        "array": arr,
                    }
                )
            except Exception:
                pass

        for child in node.get_children():
            self._traverse_for_memory(child, memory_info)

    def analyze_memory_accesses(self, cursor) -> Dict:
        memory_info = {
            "global_reads": 0,
            "global_writes": 0,
            "shared_memory": False,
            "shared_arrays": [],
            "array_accesses": [],
            "uses_syncthreads": False,
        }

        tokens = [t.spelling for t in cursor.get_tokens()]
        source_text = " ".join(tokens)

        write_pattern = r"\w+\s*\[\s*[^\]]+\s*\]\s*(?:=|\+=|-=|\*=|/=)"
        writes = re.findall(write_pattern, source_text)
        memory_info["global_writes"] = len(writes)

        array_pattern = r"\w+\s*\[\s*[^\]]+\s*\]"
        all_accesses = re.findall(array_pattern, source_text)
        memory_info["global_reads"] = max(len(all_accesses) - len(writes), 0)

        self._traverse_for_memory(cursor, memory_info)

        return memory_info

    def _traverse_and_count_ops(self, node, ops: Dict):
        if node.kind == CursorKind.BINARY_OPERATOR:
            ops["arithmetic"] += 1
        elif node.kind == CursorKind.UNARY_OPERATOR:
            ops["arithmetic"] += 1
        elif node.kind in [CursorKind.ARRAY_SUBSCRIPT_EXPR, CursorKind.MEMBER_REF_EXPR]:
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
            self._traverse_and_count_ops(child, ops)

    def count_operations(self, cursor) -> Dict:
        ops = {"arithmetic": 0, "memory": 0, "control_flow": 0, "loops": 0}
        self._traverse_and_count_ops(cursor, ops)
        return ops

    def extract_kernel_parameters(self, cursor) -> List[Dict]:
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

    def analyze_thread_indexing(self, cursor) -> Dict:
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

        all_tokens = [t.spelling for t in cursor.get_tokens()]
        source_text = " ".join(all_tokens)

        thread_info["uses_threadIdx_x"] = (
            "threadIdx.x" in source_text or "threadIdx . x" in source_text
        )
        thread_info["uses_threadIdx_y"] = (
            "threadIdx.y" in source_text or "threadIdx . y" in source_text
        )
        thread_info["uses_threadIdx_z"] = (
            "threadIdx.z" in source_text or "threadIdx . z" in source_text
        )
        thread_info["uses_blockIdx_x"] = (
            "blockIdx.x" in source_text or "blockIdx . x" in source_text
        )
        thread_info["uses_blockIdx_y"] = (
            "blockIdx.y" in source_text or "blockIdx . y" in source_text
        )
        thread_info["uses_blockIdx_z"] = (
            "blockIdx.z" in source_text or "blockIdx . z" in source_text
        )
        thread_info["uses_blockDim_x"] = (
            "blockDim.x" in source_text or "blockDim . x" in source_text
        )
        thread_info["uses_blockDim_y"] = (
            "blockDim.y" in source_text or "blockDim . y" in source_text
        )
        thread_info["uses_blockDim_z"] = (
            "blockDim.z" in source_text or "blockDim . z" in source_text
        )

        if thread_info["uses_threadIdx_z"] or thread_info["uses_blockIdx_z"]:
            thread_info["dimensionality"] = 3
        elif thread_info["uses_threadIdx_y"] or thread_info["uses_blockIdx_y"]:
            thread_info["dimensionality"] = 2
        else:
            thread_info["dimensionality"] = 1

        return thread_info

    def estimate_compute_intensity(self, cursor, memory_info: Dict, ops: Dict) -> Dict:
        tokens = [t.spelling for t in cursor.get_tokens()]
        src_text = " ".join(tokens)

        mul_div_count = tokens.count("*") + tokens.count("/")
        add_sub_count = tokens.count("+") + tokens.count("-")

        estimated_flops = ops.get("arithmetic", 0) + 2 * mul_div_count + add_sub_count

        bytes_per_element = 4
        if re.search(r"\bdouble\b", src_text) or any(
            "double" in p["type"] for p in self.extract_kernel_parameters(cursor)
        ):
            bytes_per_element = 8

        total_mem_accesses = max(
            memory_info.get("global_reads", 0) + memory_info.get("global_writes", 0), 1
        )
        estimated_memory_bytes = total_mem_accesses * bytes_per_element

        compute_intensity = estimated_flops / max(estimated_memory_bytes, 1)

        return {
            "estimated_flops": int(estimated_flops),
            "estimated_memory_bytes": int(estimated_memory_bytes),
            "flops_per_byte": float(compute_intensity),
        }

    def analyze_kernel(self, cursor) -> Dict:
        params = self.extract_kernel_parameters(cursor)
        thread_usage = self.analyze_thread_indexing(cursor)
        memory_access = self.analyze_memory_accesses(cursor)
        operations = self.count_operations(cursor)
        metrics = self.estimate_compute_intensity(cursor, memory_access, operations)

        kernel_info = {
            "name": cursor.spelling,
            "location": {
                "file": (
                    cursor.location.file.name if cursor.location.file else "unknown"
                ),
                "line": cursor.location.line,
            },
            "parameters": params,
            "thread_usage": thread_usage,
            "memory_access": memory_access,
            "operations": operations,
            "metrics": {
                "compute_intensity_flops_per_byte": metrics["flops_per_byte"],
                "estimated_flops": metrics["estimated_flops"],
                "estimated_memory_bytes": metrics["estimated_memory_bytes"],
                "has_shared_memory": memory_access["shared_memory"],
                "dimensionality": thread_usage["dimensionality"],
                "uses_syncthreads": memory_access.get("uses_syncthreads", False),
            },
        }

        return kernel_info

    def analyze(self) -> List[Dict]:
        try:
            tu = self.parse_file()
        except Exception as e:
            print(f"Warning: Parse failed ({e}), trying fallback method...")
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
        with open(self.cuda_file, "r") as f:
            content = f.read()

        pattern = (
            r"__global__\s+\w+\s+(\w+)\s*\([^)]*\)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}"
        )
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            kernel_name = match.group(1)
            kernel_body = match.group(2) if len(match.groups()) > 1 else match.group(0)

            uses_threadIdx_x = "threadIdx.x" in kernel_body
            uses_threadIdx_y = "threadIdx.y" in kernel_body
            uses_threadIdx_z = "threadIdx.z" in kernel_body
            uses_blockIdx_x = "blockIdx.x" in kernel_body
            uses_blockIdx_y = "blockIdx.y" in kernel_body
            uses_blockIdx_z = "blockIdx.z" in kernel_body

            if uses_threadIdx_z or uses_blockIdx_z:
                dimensionality = 3
            elif uses_threadIdx_y or uses_blockIdx_y:
                dimensionality = 2
            else:
                dimensionality = 1

            array_accesses = kernel_body.count("[")
            arithmetic_ops = (
                kernel_body.count("+")
                + kernel_body.count("-")
                + kernel_body.count("*")
                + kernel_body.count("/")
            )

            write_count = len(re.findall(r"\w+\[[^\]]+\]\s*=", kernel_body))
            read_count = max(array_accesses - write_count, 1)

            has_shared = "__shared__" in kernel_body
            uses_syncthreads = (
                "__syncthreads" in kernel_body or "syncthreads" in kernel_body
            )

            total_memory_ops = read_count + max(write_count, 1)
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
                    "uses_syncthreads": uses_syncthreads,
                },
                "operations": {
                    "arithmetic": arithmetic_ops,
                    "memory": array_accesses,
                    "control_flow": kernel_body.count("if"),
                    "loops": kernel_body.count("for") + kernel_body.count("while"),
                },
            }

            bytes_per_element = 8 if "double" in kernel_body else 4
            estimated_flops = arithmetic_ops
            estimated_memory_bytes = total_memory_ops * bytes_per_element
            kernel_info["metrics"] = {
                "compute_intensity_flops_per_byte": estimated_flops
                / max(estimated_memory_bytes, 1),
                "estimated_flops": estimated_flops,
                "estimated_memory_bytes": estimated_memory_bytes,
                "has_shared_memory": has_shared,
                "dimensionality": dimensionality,
                "uses_syncthreads": uses_syncthreads,
            }

            self.kernels.append(kernel_info)

        return self.kernels

    def generate_report(self, output_file: Optional[str] = None):
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
            ma = kernel["memory_access"]
            print(
                f"  Memory accesses: {ma['global_reads']} reads, {ma['global_writes']} writes"
            )
            print(f"  Shared memory: {'Yes' if ma['shared_memory'] else 'No'}")
            print(
                f"  Uses __syncthreads(): {'Yes' if kernel['metrics'].get('uses_syncthreads') else 'No'}"
            )
            print(
                f"  Compute intensity (FLOPs/byte): {kernel['metrics']['compute_intensity_flops_per_byte']:.4f}"
            )
            print(f"  Estimated FLOPs: {kernel['metrics']['estimated_flops']}")
            print(
                f"  Estimated memory bytes: {kernel['metrics']['estimated_memory_bytes']}"
            )
            print(f"  Operations: {kernel['operations']}")
            print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 cuda_anal.py <cuda_file.cu> [output.json]")
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
