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

#define TILE_SIZE 16
__global__ void matVectorMulOpt(double* mat, double* vec, double *res, sizeInfo size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double v[TILE_SIZE];

    if (tid < size.rows * size.cols) {
        res[tid] = 0.0;
        for (int m = 0; m < size.cols / TILE_SIZE; m++) {
            // each thread loads a value to the shared array
            v[threadIdx.x] = vec[m * TILE_SIZE + threadIdx.x];
            __syncthreads(); // wait for loads to finish

            for (int i = 0; i < TILE_SIZE; i++) {
                res[tid] += mat[tid * size.cols + m * TILE_SIZE + i] * v[i];
            }
        }
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
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            /*matrix[i * cols + j] = rand() / (RAND_MAX * 1.0) * 2.0 - 1.0;*/
            matrix[i * cols + j] = ceil(rand() / (RAND_MAX * 1.0) * 10.0);
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

    gpuErrchk( cudaEventRecord(start) );
    matVectorMulOpt<<<cols / 16 + 1, 16>>>(dev_matrix, dev_v, dev_result, sizes);
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
