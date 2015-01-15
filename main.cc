#include <iostream>

#include "cublas.h"
#include "plainKernel.h"

int main()
{
    int matrixSizes[] = {
        2 << 9, 2 << 9,
        2 << 10, 2 << 10,
        2 << 11, 2 << 11,
        2 << 12, 2 << 12,
        9000, 9000,
        1025, 1025,
        1992, 2001,
        3342, 114
    };
    int nSizes = sizeof(matrixSizes) / sizeof(int);
    nSizes /= 2;

    // call variant 1
    for (int i = 0; i < nSizes; ++i) {
        runCublas(matrixSizes[2*i], matrixSizes[2*i+1]);
    }

    // call variant 2
    for (int i = 0; i < nSizes; i += 1) {
        plainKernelSetup(matrixSizes[2*i], matrixSizes[2*i+1]);
    }

    // call variant 3
}
