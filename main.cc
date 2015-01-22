#include <iostream>

#include "cublas.h"
#include "plainKernel.h"
#include "optimizedKernel.h"

#define KNRM  "\x1B[0m"
#define KGRN  "\x1B[32m"

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

    // vector holding elapsed times
    float *runtimes = new float[3 * nSizes];

    // call variant 1
    std::cout << KGRN "\n== Running cuBLAS MV multiplication tests... ==" KNRM << std::endl;
    for (int i = 0; i < nSizes; ++i) {
        runtimes[i] = runCublas(matrixSizes[2*i], matrixSizes[2*i+1]);
    }

    // call variant 2
    std::cout << KGRN "\n== Running vanilla MV multiplication tests... ==" KNRM << std::endl;
    for (int i = 0; i < nSizes; i += 1) {
        runtimes[nSizes + i] = plainKernelSetup(matrixSizes[2*i], matrixSizes[2*i+1], true);
    }

    // call variant 3
    std::cout << KGRN "\n== Running optimized MV multiplication tests... ==" KNRM << std::endl;
    for (int i = 0; i < nSizes; i++) {
        runtimes[nSizes * 2 + i] = optimizedKernelSetup(matrixSizes[2*i], matrixSizes[2*i+1], true);
    }

    // show results
    std::cout << KGRN "\n== Elapsed time comparison ==\n" KNRM;
    std::cout << "\nMatrix size\t\tcuBLAS\t\tPlain\t\tOptimized\n";
    std::cout.precision(3);
    for (int i = 0; i < nSizes; ++i) {
        std::cout << matrixSizes[2*i] << "x" << matrixSizes[2*i+1];
        for (int k = 0; k < 3; k++)
        {
            std::cout << "\t\t" << runtimes[k * nSizes + i] << "ms";
        }
        std::cout << std::endl;
    }

    delete[] runtimes;
    return 0;
}
