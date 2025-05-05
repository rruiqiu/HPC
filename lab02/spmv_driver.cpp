// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include <iostream>
#include <benchmark/benchmark.h>

#include "spmv.h"
#include "def.h"
#include "utils.h"
#include "gemv.h"


static void BM_GEMV(benchmark::State &state,
                    void (*gemvImpl1)(int M, int N, const float *A, const float *b, float *c,
                                      swiftware::hpp::ScheduleParams Sp)) {
  int m = state.range(0);
  int n = state.range(1);

  int t1 = state.range(2);
  int t2 = state.range(3);
  //int srPercentage = state.range(5);
  int srPercentage = 20;
  auto *A = new swiftware::hpp::DenseMatrix(m, n);
  auto *b = new swiftware::hpp::DenseMatrix(n, 1);
  auto *c = new swiftware::hpp::DenseMatrix(m, 1);

  for(int i=0;i<m*n;i++){
    A->data[i] = 1.0;
  }

  for(int i=0;i<n;i++){
    b->data[i] =  1.0;
  }

  // TODO: implement the benchmark for GEMV and SPMV Skipping

  // TODO do sampling here
  float samplingRate = (float)srPercentage / 100.; //sampling rate contrls how much of matrix A remains non=zero
  //so if smpaleing rate ie 1 (100%), no elments are set to zero and matrix A remains fully dense
  auto* sampledA = swiftware::hpp::samplingDense(A,samplingRate);

  for (auto _: state) {
    gemvImpl1(m, n, sampledA->data.data(), b->data.data(), c->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
  }

  delete A;
  delete b;
  delete c;
  delete sampledA;


}




static void BM_SPMV(benchmark::State &state,
                    void (*spmvImpl1)(int M, int N, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c,
                                      swiftware::hpp::ScheduleParams Sp)) {


  int m = state.range(0);
  int n = state.range(1);
  int t1 = state.range(3);
  int t2 = state.range(4);
  //int srPercentage = state.range(5);
  int srPercentage = 20;
  auto *A = new swiftware::hpp::DenseMatrix(m, n);
  auto *B = new swiftware::hpp::DenseMatrix(m, 1);
  auto *C = new swiftware::hpp::DenseMatrix(m, 1);
  float samplingRate = (float)srPercentage / 100.;
  // Sample Dense Matrix


  for (int i = 0; i < m * n; ++i) {
    A->data[i] = 1.0;
  }
  for (int i = 0; i < n; ++i) {
    B->data[i] = 1.0;
  }
  auto* sampledA = swiftware::hpp::samplingDense(A,samplingRate);

  // Convert A to CSR
  auto *ACSR = swiftware::hpp::dense2CSR(sampledA);
  for (auto _: state) {
    spmvImpl1(m, n, ACSR->rowPtr.data(), ACSR->colIdx.data(), ACSR->data.data(),  B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
  }

  delete A;
  delete B;
  delete C;
  delete sampledA;
                                      
}

/*
BENCHMARK_CAPTURE(BM_SPMV, baseline_spmv_csr, swiftware::hpp::spmvCSR)
  ->Args({4096, 4096,  -1, -1});

BENCHMARK_CAPTURE(BM_GEMV, spmv_skipping, swiftware::hpp::spmvSkipping)
  ->Args({4096, 4096,  256, 32});

BENCHMARK_CAPTURE(BM_GEMV, gemv_optimized, swiftware::hpp::gemvVec)
  ->Args({4096, 4096,  1024, 32});
*/

BENCHMARK_CAPTURE(BM_SPMV, optimized_spmv_csr, swiftware::hpp::spmvCSROptimized)
  ->Args({4096, 4096,  16, 1})
  ->Args({4096, 4096,  32, 1})
  ->Args({4096, 4096,  64, 1})
  ->Args({4096, 4096,  128, 1})
  ->Args({4096, 4096,  256, 1})
  ->Args({4096, 4096,  512, 1})
  ->Args({4096, 4096,  1024, 1})
  ->Args({4096, 4096,  2048, 1})
  ->Args({4096, 4096,  4096, 1})
  ->Args({4096, 4096, 8000, 1});

/*BENCHMARK_CAPTURE(BM_GEMV, spmv_skipping, swiftware::hpp::spmvSkipping)
  ->Args({4096, 4096,  1024, 100});
*/
BENCHMARK_MAIN();

