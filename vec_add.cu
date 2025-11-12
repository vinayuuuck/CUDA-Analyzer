// Example CUDA kernels for testing the analyzer

#include <cuda_runtime.h>

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

// Matrix multiplication kernel with shared memory
__global__ void matrixMul(const float *A, const float *B, float *C, int M,
                          int N, int K) {
  __shared__ float sharedA[16][16];
  __shared__ float sharedB[16][16];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0f;

  for (int tile = 0; tile < (K + 15) / 16; ++tile) {
    // Load data into shared memory
    if (row < M && (tile * 16 + threadIdx.x) < K) {
      sharedA[threadIdx.y][threadIdx.x] = A[row * K + tile * 16 + threadIdx.x];
    } else {
      sharedA[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (col < N && (tile * 16 + threadIdx.y) < K) {
      sharedB[threadIdx.y][threadIdx.x] =
          B[(tile * 16 + threadIdx.y) * N + col];
    } else {
      sharedB[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute partial sum
    for (int k = 0; k < 16; ++k) {
      sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// Image convolution kernel
__global__ void convolution2D(const float *input, float *output,
                              const float *kernel, int width, int height,
                              int kernelSize) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float sum = 0.0f;
  int halfKernel = kernelSize / 2;

  for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
    for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
      int ix = x + kx;
      int iy = y + ky;

      if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
        int inputIdx = iy * width + ix;
        int kernelIdx = (ky + halfKernel) * kernelSize + (kx + halfKernel);
        sum += input[inputIdx] * kernel[kernelIdx];
      }
    }
  }

  output[y * width + x] = sum;
}

// Reduction kernel
__global__ void reduce(const float *input, float *output, int N) {
  __shared__ float sharedData[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  sharedData[tid] = (idx < N) ? input[idx] : 0.0f;
  __syncthreads();

  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sharedData[tid] += sharedData[tid + stride];
    }
    __syncthreads();
  }

  // Write result
  if (tid == 0) {
    output[blockIdx.x] = sharedData[0];
  }
}
