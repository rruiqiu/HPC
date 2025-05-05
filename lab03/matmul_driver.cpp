// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include <benchmark/benchmark.h>

#include "spmm.h"
#include "def.h"



#include "gemm.h"
#include "utils.h"

#define NUM_THREADS 8

static void BM_GEMM(benchmark::State &state,
                    void (*gemmImpl1)(int M, int N, int K, const float *A, const float *B, float *C,
                                      swiftware::hpp::ScheduleParams Sp)) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int t1 = state.range(3);
  int t2 = state.range(4);
  int cs = state.range(5); // chunk size
  int nt = state.range(6); // number of threads
  // TOOO : add other parameters if needed

  auto *A = new swiftware::hpp::DenseMatrix(m, k);
  auto *B = new swiftware::hpp::DenseMatrix(k, n);
  auto *C = new swiftware::hpp::DenseMatrix(m, n);
  for (int i = 0; i < m * k; ++i) {
    A->data[i] = 1.0;
  }
  for (int i = 0; i < k * n; ++i) {
    B->data[i] = 1.0;
  }

  for (auto _: state) {
    gemmImpl1(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2, nt, cs));
  }
  delete A;
  delete B;
  delete C;

}




static void BM_SPMM(benchmark::State &state,
                    void (*spmmImpl1)(int M, int N, int K, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C,
                                      swiftware::hpp::ScheduleParams Sp)) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int t1 = state.range(3);
  int t2 = state.range(4);
  int cs = state.range(5); // chunk size
  int nt = state.range(6); // number of threads
  int srPercentage = state.range(7);
  // TOOO : add other parameters if needed
  auto *A = new swiftware::hpp::DenseMatrix(m, k);
  auto *B = new swiftware::hpp::DenseMatrix(k, n);
  auto *C = new swiftware::hpp::DenseMatrix(m, n);
  float samplingRate = (float)srPercentage / 100.;
  // Sample Dense Matrix
  for (int i = 0; i < m * k; ++i) {
    A->data[i] = 1.0;
  }
  
  auto* sampledA = swiftware::hpp::samplingDense(A,samplingRate);

  // Convert A to CSR
  auto *ACSR = swiftware::hpp::dense2CSR(sampledA);

  for (int i = 0; i < k * n; ++i) {
    B->data[i] = 1.0;
  }

  for (auto _: state) {
    spmmImpl1(m, n, k, ACSR->rowPtr.data(), ACSR->colIdx.data(), ACSR->data.data(),  B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2, nt, cs));
  }

  delete A;
  delete B;
  delete C;
  delete sampledA;

}

//Args format (m, n, k, Sp.TileSize1, Sp.TileSize2, chunk size, number of threads, samplingRatePercentage)
BENCHMARK_CAPTURE(BM_GEMM, gemm_sequential, swiftware::hpp::gemmEfficientSequential)->Args({512, 512, 512, 256, 32, 1, 1, 1})
    ->Args({1024, 1024, 1024, 256, 32, 1, 1, 1})
    ->Args({2048, 2048, 2048, 256, 32, 1, 1, 1})
    ->Args({4096, 4096, 4096, 256, 32, 1, 1, 1});

BENCHMARK_CAPTURE(BM_GEMM, gemm_optimized, swiftware::hpp::gemmEfficientParallel)->Args({512, 512, 512, 256, 32, 1, NUM_THREADS, 1})
    ->Args({1024, 1024, 1024, 256, 32, 1, NUM_THREADS, 1})
    ->Args({2048, 2048, 2048, 256, 32, 1, NUM_THREADS, 1})
    ->Args({4096, 4096, 4096, 256, 32, 1, NUM_THREADS, 1});


BENCHMARK_CAPTURE(BM_SPMM, baseline_spmm_csr, swiftware::hpp::spmmCSREfficientSequential)->Args({512, 512, 512, -1, -1, 1, 1, 1})
  ->Args({1024, 1024, 1024, -1, -1, 1, 1, 1})
  ->Args({2048, 2048, 2048, -1, -1, 1, 1, 1})
  ->Args({4096, 4096, 4096, -1, -1, 1, 1, 1});

BENCHMARK_CAPTURE(BM_SPMM, optimized_spmm_csr, swiftware::hpp::spmmCSREfficientParallel)->Args({512, 512, 512, 32, -1, 1, NUM_THREADS, 1})
  ->Args({1024, 1024, 1024, 32, -1, 1, NUM_THREADS, 1})
  ->Args({2048, 2048, 2048, 32, -1, 1, NUM_THREADS, 1})
  ->Args({4096, 4096, 4096, 32, -1, 1, NUM_THREADS, 1});





BENCHMARK_MAIN();

