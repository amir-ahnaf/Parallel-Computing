#include <stdio.h> 
#include <math.h> 
#include <cuda_runtime.h> 
#include <time.h> 
// ------------------------------------------------------------------------------------------------------------ 
// Function to find Gaussian Function Integral via CUDA parallel computing 
// ------------------------------------------------------------------------------------------------------------ 
// Parallel Methodology: RectangleMidPointRuleKernel for rectangle area calculation 
//
                       Parallel Reduction for Total area summation 
// Output: total summation of gasussian function integral  
// Required header files: Required header files: stdio.h (for printf), time.h (for clock),  
//
                        cuda_runtime.h (for blockIdx.x), math.h (for exp()) 
// ------------------------------------------------------------------------------------------------------------ 
// gaussianFunction allocated to the device 
__device__ double gaussianFunction(double x) { return exp(-x * x); }

// Find the area of rectangle, stored it in result array 
__global__ void rectangleMidPointRuleKernel(double a, double interval, int n, double* result) { 
    int i = blockDim.x * blockIdx.x + threadIdx.x; 
    if (i < n) { 
        double x_mid = a + interval * (i + 0.5); 
        result[i] = gaussianFunction(x_mid) * interval; 
    } 
} 

// Parallel reduction kernel, use  
__global__ void parallelReduction(double* input, double* output, int n) { 
    extern __shared__ double sdata[]; 
    unsigned int tid = threadIdx.x; 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n) sdata[tid] = input[i]; 
    else sdata[tid] = 0.0; 
    __syncthreads(); 
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { 
        if (tid < s) { 
            sdata[tid] += sdata[tid + s]; 
        } 
    __syncthreads();
    } 
 
    if (tid == 0) output[blockIdx.x] = sdata[0]; 
} 


int main() { 
    // Variable Declaration 
    int n = 1000000; 
    double a = 0.0; 
    double b = 5.0; 
    double interval = (b - a) / n; 
    double* d_result;                                   // Pointer for device memory to store individual rectangle result 
    double* d_partialSums;                              // Pointer for device memory to store the parallel reduction summmation 
 
    // CUDA variables 
    int blockSize = 1024;                               // Number of threads in each block 
    int numBlocks = (n + blockSize - 1) / blockSize;    // number of block allocated 
    int sharedMemSize = blockSize * sizeof(double);     // Shared memory size 
    size_t size = n * sizeof(double); 
    size_t sizeBlocks = numBlocks * sizeof(double); 
 
    // Temporary array allocation 
    double* tempResult = (double*)malloc(size);         // Temporary array for Rectangle Area Calculation 
    double* partialSums = (double*)malloc(sizeBlocks);  // Temporary array for Parallel Reduction 
 
    // Start Computation and clock  
    cudaFree(0); 
    cudaMalloc(&d_result, n * sizeof(double));                  // Allocate memory on the device 
    cudaMalloc(&d_partialSums, numBlocks * sizeof(double));     // Allocate memory on the device 
    cudaDeviceSynchronize(); 
    clock_t begin = clock(); 
 
    // Parallel Component 1: Tringale Calculation  
    rectangleMidPointRuleKernel << <numBlocks, blockSize >> > (a, interval, n, d_result);      // Launch the triangle calculation kernel 
    cudaMemcpy(tempResult, d_result, n * sizeof(double), cudaMemcpyDeviceToHost);           // Copy results from device to host 
 
    // Parallel Component 2: Parallel Reduction 
    parallelReduction << <numBlocks, blockSize, sharedMemSize >> > (d_result, d_partialSums, n);   // Launch the parallel Reduction kernel 
    cudaMemcpy(partialSums, d_partialSums, numBlocks * sizeof(double), cudaMemcpyDeviceToHost); // Copy results from device to host 
 
    // Parallel Reduction Summation
    double result = 0.0; for (int i = 0; i < numBlocks; i++) { result += partialSums[i]; } 
    
    // End Computation  
    cudaDeviceSynchronize(); 
    clock_t end = clock(); 
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; 
    
    // Printing Result 
    printf("Integral of interval: %.16g\n", interval); 
    printf("Integral of Gaussian function: %.16g\n", result * 2); 
    printf("Time taken: %.16g seconds\n", time_spent); 
    
    // Free memory 
    free(tempResult); 
    free(partialSums); 
    cudaFree(d_partialSums); 
    cudaFree(d_result); 

    return 0; 
}