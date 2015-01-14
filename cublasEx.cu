#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// metrics
#include <sys/time.h>

int main(int argc, char *argv[])
{
    cublasHandle_t handle;
    cublasStatus_t stat;
    double *a, *v;
    double *dev_a;
    double *dev_v;
    double *dev_z;

    int N = 1024;
    int matched;
    if (argc > 1) {
        matched = sscanf(argv[1], "%d", &N);
        if (matched < 1) {
            printf("Usage: %s number\n", argv[0]);
            return -1;
        }
    }

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    stat = cublasDgemv(handle, CUBLAS_OP_T, N, N, &alpha, dev_a, N, dev_v, 1, &beta, dev_z, 1);
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
    
    stat = cublasGetVector(N, sizeof(double), dev_z, 1, v, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("download fail");
    }

    /*for (int i = 0; i < N; i++) {*/
        /*printf("%f ", v[i]);*/
    /*}*/

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);
    printf("Time elapsed: %fms\n", elapsedMs);

    cudaFree(dev_a);
    cudaFree(dev_v);
    cudaFree(dev_z);
    cublasDestroy(handle);
    free(a);
    free(v);
    return 0;
}
