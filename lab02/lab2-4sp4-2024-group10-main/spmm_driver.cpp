// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include <benchmark/benchmark.h>

#include "spmm.h"
#include "def.h"



#include "gemm.h"
#include "utils.h"


static void BM_GEMM(benchmark::State &state,
                    void (*gemmImpl1)(int M, int N, int K, const float *A, const float *B, float *C,
                                      swiftware::hpp::ScheduleParams Sp)) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int t1 = state.range(3);
  int t2 = state.range(4);
  int srPercentage = state.range(5);
  auto *A = new swiftware::hpp::DenseMatrix(m, k);
  auto *B = new swiftware::hpp::DenseMatrix(k, n);
  auto *C = new swiftware::hpp::DenseMatrix(m, n);

  for (int i = 0; i < m * k; ++i) {
    A->data[i] = (float)i;
  }
  for (int i = 0; i < k * n; ++i) {
    B->data[i] = (float)i;
  }
  float samplingRate = (float)srPercentage / 100.;
  auto* sampledA = swiftware::hpp::samplingDense(A,samplingRate);

  for (auto _: state) {
    gemmImpl1(m, n, k, sampledA->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
  }
  delete A;
  delete B;
  delete C;
  delete sampledA;

}




static void BM_SPMM(benchmark::State &state,
                    void (*spmmImpl1)(int M, int N, int K, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C,
                                      swiftware::hpp::ScheduleParams Sp)) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int t1 = state.range(3);
  int t2 = state.range(4);
  int srPercentage = state.range(5);
  auto *A = new swiftware::hpp::DenseMatrix(m, k);
  auto *B = new swiftware::hpp::DenseMatrix(k, n);
  auto *C = new swiftware::hpp::DenseMatrix(m, n);
  float samplingRate = (float)srPercentage / 100.;


  for (int i = 0; i < m * k; ++i) {
    A->data[i] = 1.0;
  }
  for (int i = 0; i < k * n; ++i) {
    B->data[i] = 1.0;
  }
  // Sample Dense Matrix
  auto* sampledA = swiftware::hpp::samplingDense(A,samplingRate);

  // Convert A to CSR
  auto *ACSR = swiftware::hpp::dense2CSR(sampledA);

  for (auto _: state) {
    spmmImpl1(m, n, k, ACSR->rowPtr.data(), ACSR->colIdx.data(), ACSR->data.data(),  B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
  }

  delete A;
  delete B;
  delete C;
  delete sampledA;

}


//Args format (m, n, k, Sp.TileSize1, Sp.TileSize2, samplingRatePercentage)
//last param is the sampling percentage


// BENCHMARK_CAPTURE(BM_GEMM, spmmSkipping, swiftware::hpp::spmmSkipping)
//   ->Args({4096, 4096, 4096, -1, -1, 1 })
//   ->Args({4096, 4096, 4096, -1, -1, 2 })
//   ->Args({4096, 4096, 4096, -1, -1, 3 })
//   ->Args({4096, 4096, 4096, -1, -1, 4 })
//   ->Args({4096, 4096, 4096, -1, -1, 5 })
//   ->Args({4096, 4096, 4096, -1, -1, 10});
  // ->Args({4096, 4096, 4096, -1, -1, 15})
  // ->Args({4096, 4096, 4096, -1, -1, 18})
  // ->Args({4096, 4096, 4096, -1, -1, 20})
  // ->Args({4096, 4096, 4096, -1, -1, 30})
  // ->Args({4096, 4096, 4096, -1, -1, 40})
  // ->Args({4096, 4096, 4096, -1, -1, 50})
  // ->Args({4096, 4096, 4096, -1, -1, 60});


BENCHMARK_CAPTURE(BM_SPMM, spmmCSR_cache, swiftware::hpp::spmmCSR)
  // ->Args({4096, 4096, 4096, -1, -1, 1 })
  // ->Args({4096, 4096, 4096, -1, -1, 2 })
  // ->Args({4096, 4096, 4096, -1, -1, 3 })
  // ->Args({4096, 4096, 4096, -1, -1, 4 })
  // ->Args({4096, 4096, 4096, -1, -1, 5 })
  // ->Args({4096, 4096, 4096, -1, -1, 10});
  // ->Args({4096, 4096, 4096, -1, -1, 15})
  // ->Args({4096, 4096, 4096, -1, -1, 18})
  // ->Args({4096, 4096, 4096, -1, -1, 20})
  ->Args({4096, 4096, 4096, -1, -1, 30})
  ->Args({4096, 4096, 4096, -1, -1, 50})
  ->Args({4096, 4096, 4096, -1, -1, 60});

// BENCHMARK_CAPTURE(BM_SPMM, spmmCSR_cache, swiftware::hpp::spmmCSR_cache)
//   ->Args({4096, 4096, 4096, 2, -1,    4})
//   ->Args({4096, 4096, 4096, 4, -1,    4})
//   ->Args({4096, 4096, 4096, 8, -1,    4})
//   ->Args({4096, 4096, 4096, 16, -1,   4})
//   ->Args({4096, 4096, 4096, 32, -1,   4})
//   ->Args({4096, 4096, 4096, 64, -1,   4})
//   ->Args({4096, 4096, 4096, 128, -1,  4})
//   ->Args({4096, 4096, 4096, 256, -1,  4})
//   ->Args({4096, 4096, 4096, 512, -1,  4})
//   ->Args({4096, 4096, 4096, 1024, -1, 4})
//   ->Args({4096, 4096, 4096, 2048, -1, 4})
//   ->Args({4096, 4096, 4096, 4096, -1, 4});


// BENCHMARK_CAPTURE(BM_SPMM, spmmCSROptimized_tilled, swiftware::hpp::spmmCSROptimized_tilled)
//   ->Args({4096, 4096, 4096, 2,   -1, 16})
//   ->Args({4096, 4096, 4096, 4,   -1, 16})
//   ->Args({4096, 4096, 4096, 8,   -1, 16})
//   ->Args({4096, 4096, 4096, 16,  -1, 16})
//   ->Args({4096, 4096, 4096, 32,  -1, 16})
//   ->Args({4096, 4096, 4096, 64,  -1, 16})
//   ->Args({4096, 4096, 4096, 128, -1, 16})
//   ->Args({4096, 4096, 4096, 256, -1, 16})
//   ->Args({4096, 4096, 4096, 512, -1, 16})
//   ->Args({4096, 4096, 4096, 1024,-1, 16})
//   ->Args({4096, 4096, 4096, 2048,-1, 16})
//   ->Args({4096, 4096, 4096, 4096,-1, 16});

// BENCHMARK_CAPTURE(BM_SPMM, spmmCSROptimized, swiftware::hpp::spmmCSROptimized)
//   ->Args({4096, 4096, 4096, -1, -1, 1 })
//   ->Args({4096, 4096, 4096, -1, -1, 2 })
//   ->Args({4096, 4096, 4096, -1, -1, 3 })
//   ->Args({4096, 4096, 4096, -1, -1, 4 })
//   ->Args({4096, 4096, 4096, -1, -1, 5 })
//   ->Args({4096, 4096, 4096, -1, -1, 10})
//   ->Args({4096, 4096, 4096, -1, -1, 15})
//   ->Args({4096, 4096, 4096, -1, -1, 18})
//   ->Args({4096, 4096, 4096, -1, -1, 20})
//   ->Args({4096, 4096, 4096, -1, -1, 30})
//   ->Args({4096, 4096, 4096, -1, -1, 50})
//   ->Args({4096, 4096, 4096, -1, -1, 60});
// BENCHMARK_CAPTURE(BM_SPMM, optimized_spmm_csr, swiftware::hpp::spmmCSROptimized)->Args({512, 512, 512, 32, -1, 1})
//   ->Args({1024, 1024, 1024, 32, -1, 1})
//   ->Args({2048, 2048, 2048, 32, -1, 1})
//   ->Args({4096, 4096, 4096, 32, -1, 1});

// BENCHMARK_CAPTURE(BM_GEMM, spmm_skipping, swiftware::hpp::spmmSkipping)->Args({512, 512, 512, 256, 32, 1})
//   ->Args({1024, 1024, 1024, 256, 32, 1})
//   ->Args({2048, 2048, 2048, 256, 32, 1})
//   ->Args({4096, 4096, 4096, 256, 32, 1});

BENCHMARK_MAIN();

