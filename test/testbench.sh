#serial
rm streamcluster
g++ -Wall -Wextra -o streamcluster streamcluster_omp.cpp
echo serial >> times
./streamcluster 10 20 512 65536 65536 1000 none output_medium.txt >> times

#omp O0 1 thread
export OMP_NUM_THREADS=1
rm streamcluster
g++ -Wall -Wextra -fopenmp -O0 -o streamcluster streamcluster_omp.cpp
echo omp_o0_1thread >> times
./streamcluster 10 20 512 65536 65536 1000 none omp01_output_medium.txt >> times

#omp O0 2 thread
export OMP_NUM_THREADS=2
echo omp_o0_2threads >> times
./streamcluster 10 20 512 65536 65536 1000 none omp02_output_medium.txt >> times

#omp O0 4 thread
export OMP_NUM_THREADS=4
echo omp_o0_4threads >> times
./streamcluster 10 20 512 65536 65536 1000 none omp04_output_medium.txt >> times

#omp O3 1 thread
export OMP_NUM_THREADS=1
rm streamcluster
g++ -Wall -Wextra -fopenmp -O3 -o streamcluster streamcluster_omp.cpp
echo omp_o3_1thread >> times
./streamcluster 10 20 512 65536 65536 1000 none omp31_output_medium.txt >> times

#omp O3 2 thread
export OMP_NUM_THREADS=2
echo omp_o3_2threads >> times
./streamcluster 10 20 512 65536 65536 1000 none omp32_output_medium.txt >> times

#omp O3 4 thread
export OMP_NUM_THREADS=4
echo omp_o3_4threads >> times
./streamcluster 10 20 512 65536 65536 1000 none omp34_output_medium.txt >> times

#simd O0 1 thread
export OMP_NUM_THREADS=1
rm streamcluster
g++ -Wall -Wextra -fopenmp -O0 -msse -DENABLE_SIMD -o streamcluster streamcluster_omp.cpp
echo simd_o0_1thread >> times
./streamcluster 10 20 512 65536 65536 1000 none simd01_output_medium.txt >> times

#simd O0 2 thread
export OMP_NUM_THREADS=2
echo simd_o0_2threads >> times
./streamcluster 10 20 512 65536 65536 1000 none simd02_output_medium.txt >> times

#simd O0 4 thread
export OMP_NUM_THREADS=4
echo simd_o0_4threads >> times
./streamcluster 10 20 512 65536 65536 1000 none simd04_output_medium.txt >> times

#simd O3 1 thread
export OMP_NUM_THREADS=1
rm streamcluster
g++ -Wall -Wextra -fopenmp -O3 -msse -DENABLE_SIMD -o streamcluster streamcluster_omp.cpp
echo simd_o3_1thread >> times
./streamcluster 10 20 512 65536 65536 1000 none simd31_output_medium.txt >> times

#simd O3 2 thread
export OMP_NUM_THREADS=2
echo simd_o3_2threads >> times
./streamcluster 10 20 512 65536 65536 1000 none simd32_output_medium.txt >> times

#simd O3 4 thread
export OMP_NUM_THREADS=4
echo simd_o3_4threads >> times
./streamcluster 10 20 512 65536 65536 1000 none simd34_output_medium.txt >> times
