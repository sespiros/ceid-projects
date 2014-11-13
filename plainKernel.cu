#include <cstdlib>
#include <iostream>

#include "plainKernel.h"

int plainKernelSetup(int rows, int cols)
{
    double *matrix, *v;
    sizeInfo sizes;

    sizes.rows = rows;
    sizes.cols = cols;

    // allocate matrix & vector memory
    matrix = (double *) malloc(rows * cols * sizeof(double));
    v = (double *) malloc(cols * sizeof(double));

    // randomize matrix elements
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = rand() / (RAND_MAX * 1.0) * 2.0 - 1.0;
        }
    }

    // randomize vector elements
    for (int i = 0; i < cols; i++) {
        v[i] = rand() / (RAND_MAX * 1.0) * 2.0 - 1.0;
    }

    free(matrix);
    free(v);

}
