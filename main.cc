#include <iostream>
#include <iomanip>
#include <vector>

#include "cublas.h"
#include "plainKernel.h"
#include "optimizedKernel.h"

#define KNRM  "\x1B[0m"
#define KGRN  "\x1B[32m"

int main()
{
    std::vector<int> matrixDims;
    int p[] = {7, 8, 9, 10, 11, 12, 13};

    for (int i = 0; i < 7; ++i) {
        matrixDims.push_back(2 << (p[i] - 1));
        matrixDims.push_back(2 << (p[i] - 2));

        matrixDims.push_back(2 << (p[i] - 1));
        matrixDims.push_back(2 << (p[i] - 1));
    }
    matrixDims.push_back(9000);
    matrixDims.push_back(9000);
    matrixDims.push_back(3342);
    matrixDims.push_back(114);
    matrixDims.push_back(1025);
    matrixDims.push_back(1025);

    int nDims = matrixDims.size() / 2;

    // vector holding elapsed times
    float *runtimes = new float[3 * nDims];

    // call variant 1
    std::cout << KGRN "\n== Running cuBLAS MV multiplication tests... ==" KNRM << std::endl;
    for (int i = 0; i < nDims; i += 1) {
        runtimes[i] = runCublas(matrixDims[2*i], matrixDims[2*i + 1]);
    }

    // call variant 2
    std::cout << KGRN "\n== Running vanilla MV multiplication tests... ==" KNRM << std::endl;
    for (int i = 0; i < nDims; i += 1) {
        runtimes[nDims + i] = plainKernelSetup(matrixDims[2*i], matrixDims[2*i + 1], true);
    }

    // call variant 3
    std::cout << KGRN "\n== Running optimized MV multiplication tests... ==" KNRM << std::endl;
    for (int i = 0; i < nDims; i += 1) {
        runtimes[nDims * 2 + i] = optimizedKernelSetup(matrixDims[2*i], matrixDims[2*i + 1], true);
    }

    // show results
    std::cout << KGRN "\n== Elapsed time comparison ==\n" KNRM;
    std::cout << "\nMatrix size\tcuBLAS\t\tPlain\t\tOptimized\n";
    std::cout.precision(3);
    for (int i = 0; i < nDims; ++i) {
        std::cout << std::left << std::setw(4) << matrixDims[2*i] << "x" << std::setw(4) << matrixDims[2*i+1];
        for (int k = 0; k < 3; k++)
        {
            std::cout << "\t" << std::setw(6) << runtimes[k * nDims + i] << "ms";
        }
        std::cout << std::endl;
    }

    delete[] runtimes;
    return 0;
}
