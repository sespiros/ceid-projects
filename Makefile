CCX=g++
CXXFLAGS=-Wall -Wextra -O0
SIMDFLAGS=-msse -DENABLE_SIMD 
OMPFLAGS=-fopenmp
EXTRAFLAGS=-march=native

streamcluster: streamcluster_omp.cpp
	$(CXX) -ostreamcluster $(CXXFLAGS) streamcluster_omp.cpp

omp: streamcluster_omp.cpp
	$(CXX) -ostreamcluster $(CXXFLAGS) $(OMPFLAGS) streamcluster_omp.cpp

simd: streamcluster_omp.cpp
	$(CXX) -ostreamcluster $(CXXFLAGS) $(SIMDFLAGS) streamcluster_omp.cpp

all: streamcluster_omp.cpp
	$(CXX) -ostreamcluster $(CXXFLAGS) $(SIMDFLAGS) $(OMPFLAGS) streamcluster_omp.cpp

extra: streamcluster_omp.cpp
	$(CXX) -ostreamcluster $(CXXFLAGS) $(SIMDFLAGS) $(OMPFLAGS) $(EXTRAFLAGS) streamcluster_omp.cpp

clean: streamcluster
	rm streamcluster
