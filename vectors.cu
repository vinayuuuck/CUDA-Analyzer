#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define RANGE 17.78

__global__ void vecGPU(int n, float* ad, float* bd, float* cd);

void printarr(int x, float* arr){
	for(int i = 0; i < x; i++){
		printf("Element number %d is: %f\n", i, arr[i]);
	}
}

int main(int argc, char *argv[]){

	int n = 0; //number of elements in the arrays
	int i;  //loop index
	float *a, *b, *c; // The arrays that will be processed in the host.
	float *temp;  //array in host used in the sequential code.
	float *ad, *bd, *cd; //The arrays that will be processed in the device.
	clock_t start, end; // to meaure the time taken by a specific part of code
	
	if(argc != 2){
		printf("usage:  ./vectorprog n\n");
		printf("n = number of elements in each vector\n");
		exit(1);
		}
		
	n = atoi(argv[1]);
	printf("Each vector will have %d elements\n", n);
	
	
	//Allocating the arrays in the host
	
	if( !(a = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array a\n");
	   exit(1);
	}
	
	if( !(b = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array b\n");
	   exit(1);
	}
	
	if( !(c = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array c\n");
	   exit(1);
	}
	
	if( !(temp = (float *)malloc(n*sizeof(float))) )
	{
	   printf("Error allocating array temp\n");
	   exit(1);
	}
	
	//Fill out the arrays with random numbers between 0 and RANGE;
	srand((unsigned int)time(NULL));
	for (i = 0; i < n;  i++){
        a[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
		b[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
		c[i] = ((float)rand()/(float)(RAND_MAX)) * RANGE;
		temp[i] = c[i]; //temp is just another copy of C
	}
	
    //The sequential part
	start = clock();
	for(i = 0; i < n; i++)
		temp[i] += a[i] * b[i];
	end = clock();
	printf("Total time taken by the sequential part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    /******************  The start GPU part: Do not modify anything in main() above this line  ************/
	//The GPU part
	
		cudaMalloc(&ad, n*sizeof(float));
		cudaMalloc(&bd, n*sizeof(float));
		cudaMalloc(&cd, n*sizeof(float));
		cudaMemcpy(ad, a, n*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(bd, b, n*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(cd, c, n*sizeof(float), cudaMemcpyHostToDevice);

		start = clock();
		
		cudaError_t err;
    int threadsPerBlock = 500;
    int numBlocks = 8;
 
	  vecGPU<<<numBlocks, threadsPerBlock>>>(n, ad, bd, cd);

    err = cudaGetLastError();
    if(err != cudaSuccess){ fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess){ fprintf(stderr, "synchronize failed: %s\n", cudaGetErrorString(err)); return 1; }
		end = clock();
		cudaMemcpy(c, cd, n*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(ad);
		cudaFree(bd);
		cudaFree(cd);
	
	
	printf("Total time taken by the GPU part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
	/******************  The end of the GPU part: Do not modify anything in main() below this line  ************/
	
	//checking the correctness of the GPU part
	for(i = 0; i < n; i++)
	  if( fabs(temp[i] - c[i]) >= 0.009) //compare up to the second degit in floating point
		{
			printf("Element %d in the result array does not match the sequential version\n", i);
		}
		
	// Free the arrays in the host
	free(a); free(b); free(c); free(temp);

	return 0;
}

__global__ void vecGPU(int n, float* ad, float* bd, float* cd){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for(int i = idx; i<n; i+=stride){
		cd[i] += ad[i]*bd[i];
	}
}

