# Optimization of non-parallel streamcluster

Streamcluster is a Recognition, Mining and Synthesis (RMS) applications that solves the Online clustering problem. It's part of the PARSEC suite.

Given a linear version of streamcluster, optimize it using OpenMP, SIMD instructions and anything necessary to run optimally on the target system.

## Compilation instructions

- make streamcluster: serial version
- make omp: OpemMP version
- make simd: SIMD-only version
- make all: OpenMP + SIMD version
- make extra: OpenMP + SIMD version + further machine-specific optimizations
