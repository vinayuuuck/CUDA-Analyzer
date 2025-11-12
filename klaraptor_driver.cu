#include "klaraptor_driver.h"
#include <string.h>
#include <stdlib.h>

// Default configuration
KlaraptorConfig klaraptor_config = {
    .min_block_size = 32,
    .max_block_size = 1024,
    .block_size_step = 32,
    .max_grid_size = 65535,
    .use_performance_model = 1  // Use model by default
};

// Cache for storing optimal parameters per kernel
typedef struct {
    char kernel_name[256];
    dim3 opt_grid;
    dim3 opt_block;
    float best_time;
} KernelParamCache;

static KernelParamCache* param_cache = NULL;
static int cache_size = 0;
static int cache_capacity = 0;

void klaraptorInit() {
    cache_capacity = 100;
    param_cache = (KernelParamCache*)malloc(sizeof(KernelParamCache) * cache_capacity);
    cache_size = 0;
    printf("[KLARAPTOR] Runtime initialized\n");
}

void klaraptorFinalize() {
    if (param_cache) {
        printf("[KLARAPTOR] Optimal parameters found:\n");
        for (int i = 0; i < cache_size; i++) {
            printf("  %s: grid(%d,%d,%d) block(%d,%d,%d) time=%.3fms\n",
                   param_cache[i].kernel_name,
                   param_cache[i].opt_grid.x, param_cache[i].opt_grid.y, param_cache[i].opt_grid.z,
                   param_cache[i].opt_block.x, param_cache[i].opt_block.y, param_cache[i].opt_block.z,
                   param_cache[i].best_time);
        }
        free(param_cache);
        param_cache = NULL;
    }
}

// Simple performance model (placeholder for MWP-CWP)
float predictPerformance(
    int block_size,
    int grid_size,
    int dimensionality,
    float compute_intensity,
    int has_shared_memory,
    int global_reads,
    int global_writes
) {
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Get clock rates using cudaDeviceGetAttribute (CUDA 13+ compatible)
    int clockRateKHz, memClockRateKHz;
    cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&memClockRateKHz, cudaDevAttrMemoryClockRate, 0);
    
    int total_threads = block_size * grid_size;
    int num_blocks = grid_size;
    
    // Calculate occupancy
    float max_blocks_per_sm = (float)prop.maxThreadsPerMultiProcessor / block_size;
    float occupancy = fminf(max_blocks_per_sm * prop.multiProcessorCount / num_blocks, 1.0f);
    
    // Memory bandwidth factor
    float memory_ops = global_reads + global_writes;
    float memory_time = memory_ops / (prop.memoryBusWidth / 8.0f * memClockRateKHz / 1000.0f);
    
    // Compute time (simplified)
    float compute_time = total_threads / (clockRateKHz / 1000.0f * prop.multiProcessorCount * occupancy);
    
    // Shared memory benefit
    float shared_mem_factor = has_shared_memory ? 0.7f : 1.0f;
    
    // Total estimated time
    float estimated_time = (compute_time + memory_time * shared_mem_factor) / compute_intensity;
    
    return estimated_time;
}

void selectOptimalParams(
    const char* kernel_name,
    dim3 orig_grid, dim3 orig_block,
    dim3* opt_grid, dim3* opt_block,
    int dimensionality,
    float compute_intensity,
    int has_shared_memory,
    int global_reads,
    int global_writes
) {
    // Check cache first
    for (int i = 0; i < cache_size; i++) {
        if (strcmp(param_cache[i].kernel_name, kernel_name) == 0) {
            *opt_grid = param_cache[i].opt_grid;
            *opt_block = param_cache[i].opt_block;
            return;
        }
    }
    
    printf("[KLARAPTOR] Optimizing parameters for kernel: %s\n", kernel_name);
    printf("  Original: grid(%d,%d,%d) block(%d,%d,%d)\n",
           orig_grid.x, orig_grid.y, orig_grid.z,
           orig_block.x, orig_block.y, orig_block.z);
    
    dim3 best_grid = orig_grid;
    dim3 best_block = orig_block;
    float best_time = 1e9;
    
    if (klaraptor_config.use_performance_model) {
        // Use performance model to predict optimal parameters
        int total_threads = orig_grid.x * orig_grid.y * orig_grid.z * 
                           orig_block.x * orig_block.y * orig_block.z;
        
        // Search over different block sizes
        for (int block_size = klaraptor_config.min_block_size; 
             block_size <= klaraptor_config.max_block_size; 
             block_size += klaraptor_config.block_size_step) {
            
            if (block_size > total_threads) continue;
            
            int grid_size = (total_threads + block_size - 1) / block_size;
            
            // Limit grid size
            if (grid_size > klaraptor_config.max_grid_size) {
                grid_size = klaraptor_config.max_grid_size;
            }
            
            float predicted_time = predictPerformance(
                block_size, grid_size,
                dimensionality, compute_intensity,
                has_shared_memory, global_reads, global_writes
            );
            
            if (predicted_time < best_time) {
                best_time = predicted_time;
                
                // For 1D kernels
                if (dimensionality == 1) {
                    best_block = dim3(block_size, 1, 1);
                    best_grid = dim3(grid_size, 1, 1);
                }
                // For 2D kernels
                else if (dimensionality == 2) {
                    int block_x = (int)sqrt(block_size);
                    int block_y = block_size / block_x;
                    best_block = dim3(block_x, block_y, 1);
                    
                    int grid_x = (int)sqrt(grid_size);
                    int grid_y = (grid_size + grid_x - 1) / grid_x;
                    best_grid = dim3(grid_x, grid_y, 1);
                }
                // For 3D kernels
                else {
                    int block_x = (int)cbrt(block_size);
                    int block_y = (int)sqrt(block_size / block_x);
                    int block_z = block_size / (block_x * block_y);
                    best_block = dim3(block_x, block_y, block_z);
                    
                    int grid_x = (int)cbrt(grid_size);
                    int grid_y = (int)sqrt(grid_size / grid_x);
                    int grid_z = (grid_size + grid_x * grid_y - 1) / (grid_x * grid_y);
                    best_grid = dim3(grid_x, grid_y, grid_z);
                }
            }
        }
        
        printf("  Optimal (predicted): grid(%d,%d,%d) block(%d,%d,%d) est_time=%.3f\n",
               best_grid.x, best_grid.y, best_grid.z,
               best_block.x, best_block.y, best_block.z,
               best_time);
    } else {
        // Use original parameters (no optimization)
        best_grid = orig_grid;
        best_block = orig_block;
    }
    
    // Store in cache
    if (cache_size < cache_capacity) {
        strncpy(param_cache[cache_size].kernel_name, kernel_name, 255);
        param_cache[cache_size].opt_grid = best_grid;
        param_cache[cache_size].opt_block = best_block;
        param_cache[cache_size].best_time = best_time;
        cache_size++;
    }
    
    *opt_grid = best_grid;
    *opt_block = best_block;
}
