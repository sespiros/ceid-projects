#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "common.h"
#include "optimizedKernel.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// atomicAdd for doubles
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

#define TILE_SIZE 512
__global__ void matVectorMulOpt(double* res, double* mat, double* vec, sizeInfo size)
{
    __shared__ double v[TILE_SIZE];
    __shared__ int tileElt, xstartPos, ystartPos;

    if (threadIdx.x == 0) {
        if ((blockIdx.x + 1) * TILE_SIZE <= size.cols) {
            tileElt = TILE_SIZE;
        } else {
            tileElt = size.cols % TILE_SIZE;
        }
        xstartPos = blockIdx.x * TILE_SIZE; // column starting point
        ystartPos = blockIdx.y * blockDim.x; // row starting point
    }


    __syncthreads();

    // use each thread in block
    // to copy one element from the vector
    // to shared memory; limit to number of
    // elements needed
    if (threadIdx.x < tileElt) {
        v[threadIdx.x] = vec[xstartPos + threadIdx.x];
    }

    __syncthreads();

    double sum = 0.0;
    int rowIdx = ystartPos + threadIdx.x;
    if (rowIdx < size.rows) {
        for (int i = 0; i < tileElt; i++) {
            // mat is column major
            sum += mat[rowIdx + (xstartPos + i) * size.rows] * v[i];
        }
        atomicAdd(res + rowIdx, sum);
    }
}

void matVectorMulHostOpt(double *mat, double *vec, double *res, sizeInfo size)
{
    double sum = 0.0;

    for (int i = 0; i < size.rows; i++) {
        sum = 0.0;
        for (int j = 0; j < size.cols; j++) {
            sum += mat[i * size.cols + j] * vec[j];
        }
        res[i] = sum;
    }
}

int optimizedKernelSetup(int rows, int cols, bool runCPU)
{
    double *matrix, *v, *result, *result_cpu;
    double *dev_matrix, *dev_v, *dev_result;
    sizeInfo sizes;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    sizes.rows = rows;
    sizes.cols = cols;

    std::cout << "\nRunning MV multiplication for a " << rows << "x" << cols << " matrix..." << std::endl;

    int matrixSize = rows * cols * sizeof(double);
    int vectorSize = cols * sizeof(double);
    int resultSize = rows * sizeof(double);

    // allocate matrix & vector memory
    matrix = (double *) malloc(matrixSize);
    v = (double *) malloc(vectorSize);
    result = (double *) malloc(resultSize);
    result_cpu = (double *) malloc(resultSize);

    // allocate cuda memory
    gpuErrchk( cudaMalloc(&dev_matrix, matrixSize) );
    gpuErrchk( cudaMalloc(&dev_v, vectorSize) );
    gpuErrchk( cudaMalloc(&dev_result, resultSize) );

    // randomize matrix elements
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            /*matrix[i * cols + j] = ceil(rand() / (RAND_MAX * 1.0) * 10.0);*/
            matrix[j * rows + i] = ceil(rand() / (RAND_MAX * 1.0) * 10.0);
        }
    }

    // randomize vector elements
    for (int i = 0; i < cols; i++) {
        v[i] = rand() / (RAND_MAX * 1.0) * 2.0 - 1.0;
        /*v[i] = ceil(rand() / (RAND_MAX * 1.0) * 10.0);*/
    }

    // copy from host to device
    gpuErrchk( cudaMemcpy(dev_matrix, matrix, matrixSize, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dev_v, v, vectorSize, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemset(dev_result, 0, resultSize) );

    // define grid, block sizes
    int blockSize = 1024;
    dim3 block(blockSize);
    dim3 grid(cols / (TILE_SIZE + 1) + 1, rows / (blockSize + 1) + 1);

    gpuErrchk( cudaEventRecord(start) );
    matVectorMulOpt<<<grid, block>>>(dev_result, dev_matrix, dev_v, sizes);

    gpuErrchk( cudaEventRecord(stop) );

    gpuErrchk( cudaPeekAtLastError() );

    gpuErrchk( cudaMemcpy(result, dev_result, resultSize, cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaEventSynchronize(stop) );
    float msec = 0.0f;
    gpuErrchk( cudaEventElapsedTime(&msec, start, stop) );
    std::cout << msec << "ms elapsed for kernel." << std::endl;

    if (runCPU) {
        // run same multiplication on CPU
        matVectorMulHostOpt(matrix, v, result_cpu, sizes);
    }

    gpuErrchk( cudaFree(dev_matrix) );
    gpuErrchk( cudaFree(dev_v) );
    gpuErrchk( cudaFree(dev_result) );

    free(matrix);
    free(v);
    free(result);
    free(result_cpu);

    return 0;
}
