#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// metrics
#include <sys/time.h>

#define N 2047

timespec diff(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}

int main()
{
    timespec kernelStart, kernelEnd;
    cublasHandle_t handle;
    cublasStatus_t stat;
    double *a, *v;
    double *dev_a;
    double *dev_v;
    double *dev_z;

    printf("Matrix size: %dx%d\n", N, N);
    a = (double *) malloc (N * N * sizeof(double));
    v = (double *) malloc (N * sizeof(double));
    cudaMalloc((void **)&dev_a, N * N * sizeof(double));
    cudaMalloc((void **)&dev_v, N * sizeof(double));
    cudaMalloc((void **)&dev_z, N * sizeof(double));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i*N + j] = (i-1) * N + 2*j;
        }
    }

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("handle creation failed");
    }

    double alpha, beta;
    alpha = 1.0;
    beta = 0.0;
    stat = cublasSetMatrix(N, N, sizeof(double), a, N, dev_a, N);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("set matrix fail");
    }

    for (int i = 0; i < N; i++) {
        v[i] = 1.1f;
    }

    stat = cublasSetVector(N, sizeof(double), v, 1, dev_v, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("set vector fail");
    }

    for (int i = 0; i < N; i++) {
        v[i] = 0;
    }

    stat = cublasSetVector(N, sizeof(double), v, 1, dev_z, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("set vector fail");
    }

    clock_gettime(CLOCK_MONOTONIC, &kernelStart);
    stat = cublasDgemv(handle, CUBLAS_OP_T, N, N, &alpha, dev_a, N, dev_v, 1, &beta, dev_z, 1);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &kernelEnd);
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
    
    stat = cublasGetVector(N, sizeof(double), dev_z, 1, v, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("download fail");
    }

    /*for (int i = 0; i < N; i++) {*/
        /*printf("%f ", v[i]);*/
    /*}*/

    timespec elapsed = diff(kernelStart, kernelEnd);
    printf("Time elapsed: %ds %dns\n", elapsed.tv_sec, elapsed.tv_nsec);

    cudaFree(dev_a);
    cudaFree(dev_v);
    cudaFree(dev_z);
    cublasDestroy(handle);
    free(a);
    free(v);
    return 0;
}
