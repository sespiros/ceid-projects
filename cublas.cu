#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cstdio>
#include <cstdlib>

float runCublas(int rows, int cols)
{
    cublasHandle_t handle;
    cublasStatus_t stat;
    double *a, *v, *res;
    double *dev_a;
    double *dev_v;
    double *dev_z;

    std::cout << "\nRunning MV multiplication in CUBLAS for a " << rows << "x" << cols << " matrix..." << std::endl;

    a = (double *) malloc (rows * cols * sizeof(double));
    v = (double *) malloc (cols * sizeof(double));
    res = (double *) malloc (rows * sizeof(double));
    cudaMalloc((void **)&dev_a, rows * cols * sizeof(double));
    cudaMalloc((void **)&dev_v, cols * sizeof(double));
    cudaMalloc((void **)&dev_z, rows * sizeof(double));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a[i*cols + j] = (i-1) * cols + 2*j;
        }
    }

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("handle creation failed");
    }

    double alpha, beta;
    alpha = 1.0;
    beta = 0.0;
    stat = cublasSetMatrix(rows, cols, sizeof(double), a, rows, dev_a, rows);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("set matrix fail");
    }

    for (int i = 0; i < cols; i++) {
        v[i] = 1.1f;
    }

    stat = cublasSetVector(cols, sizeof(double), v, 1, dev_v, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("set vector fail");
    }

    for (int i = 0; i < rows; i++) {
        res[i] = 0;
    }

    stat = cublasSetVector(rows, sizeof(double), res, 1, dev_z, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("set vector fail");
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stat = cublasDgemv(handle, CUBLAS_OP_T, 
                        rows, cols, 
                        &alpha, 
                        dev_a, rows, 
                        dev_v, 1, 
                        &beta, 
                        dev_z, 1);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("dgemvv fail");
    }

    /*for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", a[i*N+j]);
        }
        printf("\n");
    }
    printf("\n");*/
    
    stat = cublasGetVector(rows, sizeof(double), dev_z, 1, res, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("download fail");
    }

    /*for (int i = 0; i < N; i++) {*/
        /*printf("%f ", v[i]);*/
    /*}*/

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);
    std::cout << elapsedMs << "ms elapsed for cuBLAS." << std::endl;

    cudaFree(dev_a);
    cudaFree(dev_v);
    cudaFree(dev_z);
    cublasDestroy(handle);
    free(a);
    free(v);
    free(res);

    return elapsedMs;
}
