#ifndef KLARAPTOR_DRIVER_H
#define KLARAPTOR_DRIVER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

/**
 * KLARAPTOR Driver Interface
 * 
 * This header provides the runtime parameter selection mechanism
 * similar to KLARAPTOR's LLVM Pass instrumentation.
 * 
 * The selectOptimalParams() function uses kernel characteristics
 * and a performance model to determine optimal grid/block sizes.
 */

// Configuration for parameter search
typedef struct {
    int min_block_size;
    int max_block_size;
    int block_size_step;
    int max_grid_size;
    int use_performance_model;  // 0 = search, 1 = model-based
} KlaraptorConfig;

// Global configuration (can be modified before kernel launches)
extern KlaraptorConfig klaraptor_config;

/**
 * Select optimal kernel launch parameters
 * 
 * @param kernel_name Name of the kernel being launched
 * @param orig_grid Original grid dimension
 * @param orig_block Original block dimension
 * @param opt_grid Output: optimal grid dimension
 * @param opt_block Output: optimal block dimension
 * @param dimensionality Kernel dimensionality (1D, 2D, or 3D)
 * @param compute_intensity Ratio of compute to memory operations
 * @param has_shared_memory Whether kernel uses shared memory
 * @param global_reads Number of global memory reads
 * @param global_writes Number of global memory writes
 */
void selectOptimalParams(
    const char* kernel_name,
    dim3 orig_grid, dim3 orig_block,
    dim3* opt_grid, dim3* opt_block,
    int dimensionality,
    float compute_intensity,
    int has_shared_memory,
    int global_reads,
    int global_writes
);

/**
 * Performance model prediction (simplified MWP-CWP approach)
 * 
 * Predicts execution time based on kernel characteristics
 * Returns estimated execution time in microseconds
 */
float predictPerformance(
    int block_size,
    int grid_size,
    int dimensionality,
    float compute_intensity,
    int has_shared_memory,
    int global_reads,
    int global_writes
);

/**
 * Initialize KLARAPTOR runtime
 * Should be called once at program start
 */
void klaraptorInit();

/**
 * Cleanup KLARAPTOR runtime
 * Should be called before program exit
 */
void klaraptorFinalize();

#endif // KLARAPTOR_DRIVER_H
