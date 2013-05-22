g++ -O3 -Wall -Wextra -o streamcluster streamcluster.cpp
g++ -O3 -fopenmp -Wall -Wextra -o streamcluster_omp streamcluster_omp.cpp
export OMP_NUM_THREADS=x

