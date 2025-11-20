# CUDA Block Configuration Predictor - User Manual

## Project Overview
Machine learning models for predicting optimal CUDA thread block configurations without profiling overhead. Compares Random Forest, Ensemble Deep Neural Networks (6 models), and fine-tuned LLM approaches.

## Authors
- Ujjval Chopra (uc2062@nyu.edu)
- Vinayak Singh Bhadoriya (vb2588@nyu.edu)

## Environment Requirements
- **Platform**: NYU CIMS cuda5.cims.nyu.edu (or equivalent CUDA-enabled machine)
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4070)
- **CUDA**: Version 11.0+ with nvcc compiler
- **Python**: 3.9+
- **Clang/libclang**: Required for AST-based feature extraction

## Dependencies

For inference only (running predictions):
```bash
pip install --user -r requirements_inference.txt
```

This installs:
- onnxruntime (~50MB, for ONNX model inference)
- numpy, pandas (standard libraries)
- scikit-learn (for metadata handling)
- libclang (for CUDA AST parsing)

**Note**: ONNX-based inference runs on CPU without GPU or PyTorch required!

## Project Structure
```
.
├── main.py                               # Main prediction interface
├── cuda_anal.py                          # Static feature extraction (Clang AST traversal)
├── ensemble_predictor_onnx.py            # Lightweight ONNX-only ensemble predictor
├── ensemble_dnn.py                       # Full ensemble implementation (training + PyTorch)
├── klaraptor_enriched_data.csv           # Training dataset (12,687 clean samples)
├── grid_block_model.onnx                 # Pre-trained Random Forest (ONNX, for inference)
├── grid_block_model.pkl                  # Random Forest metadata
├── feature_names.pkl                     # Feature metadata
├── ensemble_models_large/                # Pre-trained Ensemble DNN models
│   ├── fast/
│   │   ├── model.onnx                    # ONNX model (for inference)
│   │   └── metadata.pkl                  # Scaler, features, etc.
│   ├── medium_fast/model.onnx, metadata.pkl
│   ├── medium/model.onnx, metadata.pkl
│   ├── medium_slow/model.onnx, metadata.pkl
│   └── slow/model.onnx, metadata.pkl
├── requirements_inference.txt            # Minimal dependencies (inference only)
├── test-files/                           # Custom CUDA test kernels
│   ├── conv2D.cu
│   ├── conv3D.cu
│   ├── matrix_multiplication.cu
│   └── ...
└── report.pdf                            # Complete technical report
```

## Quick Start: Predict Block Configuration for Your Kernel

The main tool takes a CUDA source file and predicts optimal block dimensions:

```bash
python3 main.py <path_to_kernel.cu>
```

### Examples:
```bash
# Example 1: Simple kernel
python3 main.py test-files/conv2D.cu

# Example 2: PolyBench kernel
python3 main.py PolybenchCUDA/stencils/convolution-2d/2DConvolution.cu

# Example 3: Your custom kernel
python3 main.py my_kernel.cu
```

### What the Tool Does:
1. **Analyzes** your CUDA kernel using static AST analysis (extracts features like loop nests, memory access patterns, arithmetic intensity)
2. **Evaluates** the kernel across multiple problem sizes (default: N=128, 256, 512, 1024, 2048, 4096)
3. **Predicts** execution time for ~100-120 candidate block configurations using:
   - Random Forest baseline (26 features, R²=0.803)
   - Ensemble DNNs (38 features, 6 specialized models by execution time regime)
   - Fine-tuned LLM (optional, if available)
4. **Recommends** the best block configuration (bx, by, bz) that minimizes predicted execution time
5. **Returns** both block dimensions and calculated grid dimensions for easy kernel launch

### Example Output:
```
=== CUDA Kernel Analysis ===
File: test-files/conv2D.cu
Kernel: convolution2D
Dimensionality: 2D
Features extracted: 38 features

=== Random Forest Prediction ===
Best config across problem sizes: (16, 16, 1)
Predicted speedup vs avg random: 1.08×

=== Ensemble DNN Prediction ===
Best config across problem sizes: (32, 8, 1)
Predicted speedup vs avg random: 1.15×
Selected model regime: medium (0.01-0.1s)

Recommendation: Use block=(32, 8, 1) for best performance
Grid dimensions will be calculated as: grid=(⌈N/32⌉, ⌈N/8⌉, 1)
```

### Notes:
- The tool predicts **block dimensions only** (the primary performance factor)
- Grid dimensions are computed automatically from problem size: grid_x = ⌈N/block_x⌉
- No profiling required - predictions complete in <1 second
- Models generalize to unseen kernels without retraining
- Uses ONNX models by default (faster, minimal dependencies, no PyTorch required)

## Pre-Trained Models (Included - Ready to Use)

All models are pre-trained in ONNX format and ready for inference:

### Random Forest
- **File**: `grid_block_model.onnx`
- **Performance**: R² = 0.803 on test set
- **Features**: 26 static + dynamic features

### Ensemble DNNs  
- **Directory**: `ensemble_models_large/`
- **Models**: 5 regime-specific networks (fast, medium_fast, medium, medium_slow, slow)
- **Files**: Each subdirectory contains `model.onnx` and `metadata.pkl`
- **Architecture**: Deep network [512, 512, 256, 256, 128, 128, 64, 64, 32] with attention
- **Features**: 38 features
- **Performance**: Weighted R² ≈ 0.65-0.70, beats Random Forest on 80% of test kernels

## CUDA Kernel Requirements

Your CUDA kernel must:
1. Use `#ifndef BLOCK_DIM_X` style macros for block dimensions:
   ```cuda
   #ifndef BLOCK_DIM_X
   #define BLOCK_DIM_X 16
   #endif
   
   __global__ void myKernel(...) {
       // Your kernel code using BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z
   }
   ```

2. Be compilable with nvcc (standard CUDA C++ syntax)

3. Include a main() function or be a standalone kernel file

See `test-files/*.cu` for examples of properly formatted kernels.

## Troubleshooting

**Error: "libclang.so not found"**
- Install libclang: `pip install --user libclang`
- On CIMS, libclang should be available via module system

**Error: "nvcc not found"**  
- Ensure CUDA toolkit is in PATH: `which nvcc`
- On CIMS: `module load cuda` or check `/usr/local/cuda/bin`

**Error: "Cannot load model file"**
- Verify `grid_block_model.pkl` and `ensemble_models_large/` exist in project directory
- Check PyTorch version compatibility (requires PyTorch 2.0+)

**Poor predictions on your kernel:**
- Check if kernel is very different from training data (PolyBench/GPU benchmarks)
- Verify static features are extracted correctly: run `python3 cuda_anal.py your_kernel.cu`
- Models work best on compute/memory-bound kernels, may struggle with highly divergent code

## Performance Expectations

Based on validation experiments (see report.pdf Table 9):
- **vs Average Random**: Models achieve 1.08-1.10× geometric mean speedup
- **vs Best Random**: Models beat best random on 30-50% of kernels
- **Success Rate**: ~60-80% of predictions within 20% of optimal
- **Overhead**: <1 second prediction time (vs 50-100ms for profiling-based methods like KLARAPTOR)

Models provide good automated starting points but may not always match exhaustive search.

## Citation

If you use this tool in your research, please cite:
```
@article{chopra2025cuda,
  title={Predicting Optimal CUDA Block Configurations: A Machine Learning Comparison Study},
  author={Chopra, Ujjval and Bhadoriya, Vinayak Singh},
  year={2025},
  institution={New York University}
}
```