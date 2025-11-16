import pandas as pd
import sys
import os

# Add path to your analyzer
sys.path.insert(0, 'CUDA-Analyzer')

from cuda_anal import CUDAKernelAnalyzer

def add_static_features():
    # Load KLARAPTOR timing data
    df = pd.read_csv('klaraptor_timing_data.csv')
    print(f"Loaded {len(df)} timing samples\n")

    # Map kernel names to actual PolyBench source files (from find output)
    kernel_sources = {
        'GEMM': 'PolyBenchCUDA/linear-algebra/kernels/gemm/gemm.cu',
        'ATAX': 'PolyBenchCUDA/linear-algebra/kernels/atax/atax.cu',
        'MVT': 'PolyBenchCUDA/linear-algebra/kernels/mvt/mvt.cu',
        '2MM': 'PolyBenchCUDA/linear-algebra/kernels/2mm/2mm.cu',
        '3MM': 'PolyBenchCUDA/linear-algebra/kernels/3mm/3mm.cu',
        'BICG': 'PolyBenchCUDA/linear-algebra/kernels/bicg/bicg.cu',
        'CORR': 'PolyBenchCUDA/datamining/correlation/correlation.cu',
        'COVAR': 'PolyBenchCUDA/datamining/covariance/covariance.cu',
        'GESUMMV': 'PolyBenchCUDA/linear-algebra/kernels/gesummv/gesummv.cu',
        'SYR2K': 'PolyBenchCUDA/linear-algebra/kernels/syr2k/syr2k.cu',
        'SYRK': 'PolyBenchCUDA/linear-algebra/kernels/syrk/syrk.cu',
        '2DCONV': 'PolyBenchCUDA/stencils/convolution-2d/2DConvolution.cu',
        'FDTD_2D': 'PolyBenchCUDA/stencils/fdtd-2d/fdtd2d.cu',
        'GRAMSCHM': 'PolyBenchCUDA/linear-algebra/solvers/gramschmidt/gramschmidt.cu',
    }
    
    # Extract static features for each kernel
    kernel_features = {}
    
    unique_kernels = df['kernel'].unique()
    print(f"Analyzing {len(unique_kernels)} kernels...\n")
    
    for kernel_name in unique_kernels:
        if kernel_name not in kernel_sources:
            print(f"⚠️  {kernel_name}: No source mapping, using defaults")
            kernel_features[kernel_name] = {
                'dimensionality': 2,
                'compute_intensity': 1.0,
                'has_shared_memory': 0,
                'global_reads': 1,
                'global_writes': 1,
                'arithmetic_ops': 10,
                'memory_ops': 10,
            }
            continue
        
        source_file = kernel_sources[kernel_name]
        
        if not os.path.exists(source_file):
            print(f"⚠️  {kernel_name}: File not found: {source_file}")
            kernel_features[kernel_name] = {
                'dimensionality': 2,
                'compute_intensity': 1.0,
                'has_shared_memory': 0,
                'global_reads': 1,
                'global_writes': 1,
                'arithmetic_ops': 10,
                'memory_ops': 10,
            }
            continue
        
        print(f"Analyzing {kernel_name}...", end=' ')
        
        try:
            analyzer = CUDAKernelAnalyzer(source_file)
            analyzer.analyze()
            
            if len(analyzer.kernels) == 0:
                print("✗ No kernels found")
                kernel_features[kernel_name] = {
                    'dimensionality': 2,
                    'compute_intensity': 1.0,
                    'has_shared_memory': 0,
                    'global_reads': 1,
                    'global_writes': 1,
                    'arithmetic_ops': 10,
                    'memory_ops': 10,
                }
                continue
            
            kernel_info = analyzer.kernels[0]
            
            kernel_features[kernel_name] = {
                'dimensionality': kernel_info['thread_usage']['dimensionality'],
                'compute_intensity': kernel_info['metrics']['compute_intensity'],
                'has_shared_memory': int(kernel_info['memory_access']['shared_memory']),
                'global_reads': kernel_info['memory_access']['global_reads'],
                'global_writes': kernel_info['memory_access']['global_writes'],
                'arithmetic_ops': kernel_info['operations']['arithmetic'],
                'memory_ops': kernel_info['operations']['memory'],
            }
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            kernel_features[kernel_name] = {
                'dimensionality': 2,
                'compute_intensity': 1.0,
                'has_shared_memory': 0,
                'global_reads': 1,
                'global_writes': 1,
                'arithmetic_ops': 10,
                'memory_ops': 10,
            }
    
    # Add features to dataframe
    print("\nAdding features to dataset...")
    for feature in ['dimensionality', 'compute_intensity', 'has_shared_memory',
                   'global_reads', 'global_writes', 'arithmetic_ops', 'memory_ops']:
        df[feature] = df['kernel'].map(lambda k: kernel_features.get(k, {}).get(feature, 0))
    
    # Save
    output_file = 'klaraptor_enriched_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Enriched dataset saved!")
    print(f"{'='*70}")
    print(f"Total samples: {len(df)}")
    print(f"File: {output_file}\n")
    
    print("Sample with features:")
    print(df[['kernel', 'N', 'bx', 'by', 'exec_time', 'dimensionality', 'compute_intensity']].head(10))
    
    # Show feature variation to verify they're not all defaults
    print(f"\n{'='*70}")
    print("FEATURE VARIATION CHECK (should see different values!)")
    print(f"{'='*70}")
    print(f"Dimensionality values: {sorted(df['dimensionality'].unique())}")
    print(f"Compute intensity range: {df['compute_intensity'].min():.2f} - {df['compute_intensity'].max():.2f}")
    print(f"Compute intensity unique values: {sorted(df['compute_intensity'].unique())}")
    print(f"Has shared memory values: {sorted(df['has_shared_memory'].unique())}")
    print(f"Global reads range: {df['global_reads'].min()} - {df['global_reads'].max()}")
    print(f"{'='*70}\n")
    
    return df

if __name__ == '__main__':
    add_static_features()
