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
### For Training (Development Machine)
Install full dependencies for training models:
```bash
pip install --user -r requirements_training.txt
```

Or manually:
```bash
pip install --user torch scikit-learn pandas numpy libclang transformers skl2onnx onnxruntime
```

Specific versions used:
- PyTorch 2.1+
- scikit-learn 1.3+
- libclang 16.0+
- transformers 4.30+ (for LLM model, optional)
- pandas, numpy (standard versions)
- skl2onnx 1.15+ (for ONNX export)
- onnxruntime 1.12+ (for ONNX inference)

### For Inference Only (Deployment Machine)
**Minimal dependencies** for running predictions without PyTorch:
```bash
pip install --user -r requirements_inference.txt
```

This installs only:
- onnxruntime (lightweight, ~50MB vs PyTorch ~2GB)
- scikit-learn (for metadata handling)
- numpy, pandas (standard libraries)

**Note**: ONNX-based inference allows you to run predictions on machines without GPU or PyTorch!

## Project Structure
```
.
├── cuda_anal.py                          # Static feature extraction (Clang AST traversal)
├── ensemble_dnn.py                       # Ensemble DNN implementation (6 models)
├── random_forest.py                      # Random Forest training script
├── main.py                               # Main prediction interface
├── benchmark.py                          # Benchmark validation script
├── extract_klaraptor_data.py             # Data preprocessing from KLARAPTOR traces
├── klaraptor_enriched_data.csv           # Training dataset (12,687 clean samples)
├── grid_block_model.pkl                  # Pre-trained Random Forest model (R²=0.803)
├── grid_block_model.onnx                 # ONNX version of Random Forest (for deployment)
├── feature_names.pkl                     # Feature metadata for Random Forest
├── ensemble_models_large/                # Pre-trained Ensemble DNN models (6 regimes)
│   ├── base_pretrained/
│   │   ├── model.pth                     # PyTorch weights
│   │   ├── model.onnx                    # ONNX version (for deployment)
│   │   └── metadata.pkl
│   ├── fast/
│   │   ├── model.pth
│   │   ├── model.onnx
│   │   └── metadata.pkl
│   ├── medium_fast/model.pth, model.onnx, metadata.pkl
│   ├── medium/model.pth, model.onnx, metadata.pkl
│   ├── medium_slow/model.pth, model.onnx, metadata.pkl
│   └── slow/model.pth, model.onnx, metadata.pkl
├── export_to_onnx.py                     # Export existing models to ONNX
├── requirements_training.txt             # Full dependencies (training)
├── requirements_inference.txt            # Minimal dependencies (inference only)
├── test-files/                           # Custom CUDA test kernels for validation
│   ├── conv2D.cu
│   ├── conv3D.cu
│   ├── matrix_multiplication.cu
│   ├── reduction.cu
│   └── ...
├── PolybenchCUDA/                        # Benchmark suite source code
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
- By default, uses ONNX models (faster, minimal dependencies)
- Use `--no-onnx` flag to force PyTorch inference (requires torch)

## Deployment: ONNX Export for Lightweight Inference

### Why ONNX?
ONNX (Open Neural Network Exchange) allows you to:
- **Run on machines without PyTorch** (PyTorch ~2GB vs ONNX Runtime ~50MB)
- **Faster inference** (optimized C++ runtime)
- **No GPU required** for inference
- **Smaller deployment footprint**

### Step 1: Export Models to ONNX (on development machine)

After training your models, export them to ONNX format:

```bash
python3 export_to_onnx.py
```

This will:
- Convert Random Forest to ONNX (grid_block_model.onnx)
- Convert all 6 Ensemble DNN models to ONNX (model.onnx in each subdirectory)
- Verify that all exported models can be loaded

**Requirements for export**:
```bash
pip install --user skl2onnx onnx torch
```

### Step 2: Deploy to Inference Machine

Copy these files to your deployment/inference machine:

**Essential files**:
- `main.py`, `cuda_anal.py`, `ensemble_dnn.py`
- `grid_block_model.onnx` (Random Forest)
- `grid_block_model.pkl`, `feature_names.pkl` (metadata)
- `ensemble_models_large/*/model.onnx` (all 6 DNN models)
- `ensemble_models_large/*/metadata.pkl` (all 6 metadata files)

**Optional** (only if you want PyTorch fallback):
- `*.pth` files (PyTorch weights)

### Step 3: Install Minimal Dependencies (on inference machine)

```bash
pip install --user -r requirements_inference.txt
```

This installs only:
- `onnxruntime` (~50MB)
- `numpy`, `pandas`, `scikit-learn` (standard libraries)
- **No PyTorch required!**

### Step 4: Run Predictions

```bash
# Using ONNX (default, lightweight)
python3 main.py your_kernel.cu

# Force PyTorch (if you have it installed)
python3 main.py your_kernel.cu --no-onnx
```

### Comparison: ONNX vs PyTorch

| Aspect | ONNX Runtime | PyTorch |
|--------|--------------|---------|
| Installation size | ~50 MB | ~2 GB |
| GPU required | No | Optional |
| Inference speed | Fast (C++) | Moderate (Python) |
| Dependencies | Minimal | Heavy |
| Deployment | Easy | Complex |

**Recommendation**: Use ONNX for deployment, PyTorch only for training.

## Benchmark Validation

To reproduce the experimental results from the paper:

```bash
python3 benchmark.py
```

This script:
1. Loads test kernels from `test_suite_clean.json` (10 custom kernels)
2. For each kernel, generates 10 random baseline configurations
3. Compiles each configuration with nvcc using `-DBLOCK_DIM_X/Y/Z` flags
4. Executes each binary 5 independent runs to reduce timing variance
5. Compares DNN predictions vs RF predictions vs random baselines
6. Outputs results showing which model wins on each kernel

Expected runtime: ~10-15 minutes (compiling + executing 10 kernels × ~15 configs × 5 runs each)

Results are printed on stdout

## Pre-Trained Models (Included - No Training Needed)

All models are pre-trained and ready to use:

### Random Forest
- **File**: `grid_block_model.pkl`
- **Performance**: R² = 0.803 on test set
- **Features**: 26 static + dynamic features
- **Training data**: 12,687 configurations from PolyBench/GPU

### Ensemble DNNs  
- **Directory**: `ensemble_models_large/`
- **Models**: 6 regime-specific networks:
  - `base_pretrained/` - Warm-start model trained on all data
  - `fast/` - Specialized for <0.01s kernels
  - `medium_fast/` - For 0.01-0.05s range
  - `medium/` - For 0.05-0.1s range
  - `medium_slow/` - For 0.1-0.5s range
  - `slow/` - For >0.5s kernels
- **Architecture**: [512, 512, 256, 256, 128, 128, 64, 64, 32] with multi-head attention
- **Features**: 38 features (includes all RF features + additional kernel properties)
- **Performance**: Weighted R² ≈ 0.65-0.70, beats Random Forest on 80% of test kernels

### LLM (Optional)
- **Model**: Qwen 1.5-0.5B (500M parameters, fully fine-tuned)
- **Directory**: `cuda_exec_time_predictor_llm/` (if available)
- **Note**: Experimental - shows mode collapse (R²=-0.0099), included for comparison only

## Optional: Retraining Models from Scratch

Pre-trained models are included, but if you want to retrain all models from scratch (e.g., on new data or modified hyperparameters), follow these steps:

### Step-by-Step Training Process

**Prerequisites:**
- Ensure you have `klaraptor_enriched_data.csv` (included) or generate new data from KLARAPTOR traces
- Verify all dependencies are installed (see Dependencies section)
- Training requires ~2-4 hours on RTX 4070 GPU

#### Step 1: Verify/Generate Training Data

If you have KLARAPTOR trace files and want to regenerate the dataset:

```bash
# Extract features from KLARAPTOR results
python3 extract_klaraptor_data.py
```

This processes all benchmark results in `KLARAPTORresults/polybench_*/` directories and creates `klaraptor_enriched_data.csv` with 38 features per configuration.

**Expected output:** `klaraptor_enriched_data.csv` (~12,687 rows after outlier removal)

#### Step 2: Train Random Forest Model

```bash
# Train Random Forest baseline
python3 random_forest.py
```

**What this does:**
- Loads `klaraptor_enriched_data.csv`
- Applies outlier removal (removes top 20% slowest configs)
- Trains Random Forest with:
  - 150 trees
  - max_depth=25
  - Log-transform on execution times
- Saves model to `grid_block_model.pkl`
- Saves feature names to `feature_names.pkl`

**Expected time:** ~2-5 minutes  
**Expected output:** 
```
Training Random Forest...
R² Score: 0.803
Mean Absolute Error: 0.XXX
Model saved to grid_block_model.pkl
```

#### Step 3: Train Ensemble DNNs

```bash
# Train all 6 Ensemble DNN models
python3 ensemble_dnn.py
```

**What this does:**
- Loads `klaraptor_enriched_data.csv`
- Splits data into 5 execution time regimes (fast, medium_fast, medium, medium_slow, slow)
- Trains base pretrained model on all data (warm start)
- Trains 5 regime-specific models, each initialized from base model
- Each model uses:
  - Architecture: [512, 512, 256, 256, 128, 128, 64, 64, 32]
  - Multi-head attention (8 heads)
  - Dropout: 0.15
  - AdamW optimizer with ReduceLROnPlateau scheduler
- Saves models to `ensemble_models_large/{base_pretrained,fast,medium_fast,medium,medium_slow,slow}/model.pth`

**Expected time:** ~20-30 minutes
**Expected output:**
```
Training base model on all data...
Epoch 1/100: train_loss=X.XXX, val_loss=X.XXX
...
Base model saved to ensemble_models_large/base_pretrained/model.pth

Training regime: fast (<0.01s)
...
Model saved to ensemble_models_large/fast/model.pth

[Repeats for each regime]
```

#### Step 4: (Optional) Fine-Tune LLM

**Warning:** LLM training is experimental and currently shows poor performance (R²=-0.0099). Only attempt if you want to reproduce full experimental comparison.

Requirements:
- Download Qwen 1.5-0.5B base model
- ~16GB GPU memory
- 6-8 hours training time
- Jupyter notebook environment

```bash
# Open the fine-tuning notebook
jupyter notebook finetune_llm.ipynb
```

Then run all cells in the notebook to:
- Load Qwen 1.5-0.5B base model from HuggingFace
- Prepare training data with prompts
- Fine-tune all parameters (full fine-tuning, not LoRA)
- Training parameters: batch_size=1, gradient_accumulation=16, bf16=auto, epochs=3
- Save fine-tuned model to `cuda_exec_time_predictor_llm/`

**Not recommended for production use** - included only for research comparison.

#### Step 5: Verify Models Work

Test your newly trained models:

```bash
# Run on a test kernel
python3 main.py test-files/conv2D.cu
```

Expected output should show predictions from both Random Forest and Ensemble DNNs.

### Training Data Requirements

If creating your own training dataset, ensure:
- CSV format with columns: `N`, `block_x`, `block_y`, `block_z`, `exec_time_ms`, plus 38 feature columns
- Minimum ~5,000 samples for reasonable generalization
- Multiple problem sizes (N) per kernel
- Multiple kernels with diverse characteristics (compute-bound, memory-bound, different dimensionality)

### Hyperparameter Tuning

To modify training hyperparameters, edit the respective files:

**Random Forest** (`random_forest.py`):
- Line ~50: `n_estimators=150` (number of trees)
- Line ~51: `max_depth=25` (tree depth)

**Ensemble DNNs** (`ensemble_dnn.py`):
- Line ~80-90: Network architecture layers
- Line ~150: `dropout=0.15`
- Line ~200: `n_heads=8` (attention heads)
- Line ~300: Learning rate, batch size, epochs

## Quick Retrain Commands (Summary)

```bash
# Full retraining pipeline (if klaraptor_enriched_data.csv exists)
python3 random_forest.py              
python3 ensemble_dnn.py 

# Or regenerate data first
python3 extract_klaraptor_data.py         
python3 random_forest.py
python3 ensemble_dnn.py
```

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