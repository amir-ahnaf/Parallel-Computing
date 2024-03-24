// 2D TLM for CUDA 
//
// Simulate 2D an area divided into NX and NY dimensions of dl
//
// Origin of the area matched to the source impedence
// 
// The area is excited with gaussian function at node Ein
// 
// Line is terminated with a short circuit to ground

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <iomanip>  // for setprecision
#include <time.h>   // for clock

#define M_PI 3.14276        // Pi Value
#define c 299792458         // Speed of light
#define mu0 M_PI*4e-7       
#define eta0 c*mu0

using namespace std;

// TLM Source kernel calculating the energy at the In coordinate and store it to voltage node
__global__ void tlmSource(double* V1, double* V2, double* V3, double* V4, int sourceIdx, int n, double dt, double delay, double width) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double time = n * dt;
        double E0 = (1 / sqrt(2.)) * exp(-(time - delay) * (time - delay) / (width * width));

        // Only one thread updates the source
        V1[sourceIdx] += E0;
        V2[sourceIdx] -= E0;
        V3[sourceIdx] -= E0;
        V4[sourceIdx] += E0;
    }
}

// TLM Scatter kernel distributes the signal and update the voltage for each nnode
__global__ void tlmScatter(double* V1, double* V2, double* V3, double* V4, int NX, int NY, double Z) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = x + NY * y;

    if (x < NX && y < NY) {
        double I = (2 * V1[index] + 2 * V4[index] - 2 * V2[index] - 2 * V3[index]) / (4 * Z);

        // Update voltages for each port
        double V;
        V = 2 * V1[index] - I * Z;
        V1[index] = V - V1[index];      // port 1

        V = 2 * V2[index] + I * Z;
        V2[index] = V - V2[index];      // port 2

        V = 2 * V3[index] + I * Z;
        V3[index] = V - V3[index];      // port 3

        V = 2 * V4[index] - I * Z;
        V4[index] = V - V4[index];      // port 4
    }
}

// TLM Connect Kernel transmit the current voltage to the adjecent node for the next time step
__global__ void tlmConnect(double* V1, double* V2, double* V3, double* V4, int NX, int NY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    double tempV;
    int index = x + NY * y;

    // Horizontal Connection
    if (x > 0 && index < (NY * NX)) {
        tempV = V2[index];
        V2[index] = V4[index - 1];
        V4[index - 1] = tempV;
    }

    // Vertical Connection
    if (y > 0 && index < (NY * NX)) {
        tempV = V1[index];
        V1[index] = V3[x + (y - 1) * NY];
        V3[x + (y - 1) * NY] = tempV;
    }

}

// TLM Boundary Kernel to short the voltage at the end of the area to ground
__global__ void tlmBoundary(double* V1, double* V2, double* V3, double* V4, int NX, int NY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //boundary coefficients
    double rXmin = -1;
    double rXmax = -1;
    double rYmin = -1;
    double rYmax = -1;

    // Horizontal Connection
    if (y == (NX - 1) * NY && x < NX) {
        V3[x + (NY - 1) * NY] *= rYmax;
        V1[x] *= rYmin;
    }

    // Vertical Connection
    if (x == NX - 1 && x < NX) {
        V4[(NX - 1) + y * NY] *= rXmax;
        V2[y * NY] *= rXmin;
    }
}


int main()
{
    // Computation Variables
    clock_t start, end;                     // for clocking 
    int NX = 100;                           // X dimension of the area
    int NY = 100;                           // Y dimension of the area
    int NT = 8000;                          // Time step of the simulation
    double dl = 1;                          // Node length
    double dt = dl / (sqrt(2.) * c);        // Time step duration
    double Z = eta0 / sqrt(2.);             // Impedance 

    // Signal and 
    double width = 20 * dt * sqrt(2.);      // Width of the Gaussian function
    double delay = 100 * dt * sqrt(2.);     // Delay of the Gaussian function
    int Ein[] = { 10,10 };                  // Signal input coordinate
    int Eout[] = { 15,15 };                 // Signal output coordinate
    int sourceIdx = Ein[0] + Ein[1] * NY;   // Source Index in 1D array
    int indexEout = Eout[0] + Eout[1] * NY; // Output Index in 1D array

    // Host array 
    double* h_V2, * h_V4;
    size_t size = NX * NY * sizeof(double);
    h_V2 = (double*)malloc(size);
    h_V4 = (double*)malloc(size);
    double* h_output = (double*)malloc(NT * sizeof(double));

    // Allocate memory for each array on the GPU
    double* V1, * V2, * V3, * V4;
    cudaMalloc(&V1, size);
    cudaMalloc(&V2, size);
    cudaMalloc(&V3, size);
    cudaMalloc(&V4, size);

    // Padding each array to zero
    cudaMemset(V1, 0, size);
    cudaMemset(V2, 0, size);
    cudaMemset(V3, 0, size);
    cudaMemset(V4, 0, size);

    ofstream output("output.out");

    // Starting the clock
    start = clock();

    for (int n = 0; n < NT; n++) {
        // Source
        tlmSource << <1, 1 >> > (V1, V2, V3, V4, sourceIdx, n, dt, delay, width);

        // Set block and grid sizes
        dim3 blockSize(10, 10);
        dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y);

        // Scatter
        tlmScatter << <gridSize, blockSize >> > (V1, V2, V3, V4, NX, NY, Z);
        cudaDeviceSynchronize();

        // Connect
        tlmConnect << <gridSize, blockSize >> > (V1, V2, V3, V4, NX, NY);
        cudaDeviceSynchronize();

        // Boundary
        tlmBoundary << <gridSize, blockSize >> > (V1, V2, V3, V4, NX, NY);
        cudaDeviceSynchronize();

        // Transfer data from device to host for output
        cudaMemcpy(&h_V2[n], &V2[indexEout], sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_V4[n], &V4[indexEout], sizeof(double), cudaMemcpyDeviceToHost);
        h_V2[n] += h_V4[n];                                 // Sum V2 and V4 values for output
        output << n * dt << "  " << h_V2[n] << endl;        // Output saved in the file

    }

    output.close();

    // Display the time period
    std::cout << "Done" << endl;
    end = clock();
    std::cout << '\n' << ((end - start) / (double)CLOCKS_PER_SEC) << ' s\n';

    cudaFree(V1);
    cudaFree(V2);
    cudaFree(V3);
    cudaFree(V4);
    free(h_V2);
    free(h_V4);
    free(h_output);

}

