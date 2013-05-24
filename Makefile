CXX=g++
CXXFLAGS=-Wall -Wextra

ifeq ($(omp),true)
	CXXFLAGS+=-fopenmp
endif

ifeq ($(simd),true)
	CXXFLAGS+=-DENABLE_SIMD
endif

streamcluster: streamcluster_omp.cpp
	$(CXX) -ostreamcluster $(CXXFLAGS) streamcluster_omp.cpp
