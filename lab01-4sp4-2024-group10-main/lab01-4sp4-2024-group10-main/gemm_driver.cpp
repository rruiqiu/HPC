//
// Created by kazem on 06/09/24.
//

#include <iostream>
#include <benchmark/benchmark.h>

#include "gemm.h"
#include "def.h"


static void BM_GEMM(benchmark::State &state,
                    void (*gemmImpl1)(int M, int N, int K, const float *A, const float *B, float *C,
                                      swiftware::hpp::ScheduleParams Sp)) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    int t1 = state.range(3);
    int t2 = state.range(4);
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
        gemmImpl1(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }
    delete A;
    delete B;
    delete C;

}
/*
BENCHMARK_CAPTURE(BM_GEMM, baseline_gemm, swiftware::hpp::gemm)->Args({512, 512, 512, -1, -1})
        ->Args({1024, 1024, 1024, -1, -1})
        ->Args({2048, 2048, 2048, -1, -1})
        ->Args({4096, 4096, 4096, -1, -1});
*/


// TODO: add benchmark for other implementations of GEMM
BENCHMARK_CAPTURE(BM_GEMM, tiled_gemm, swiftware::hpp::gemmT1)->Args({512, 512, 512, 32, -1})
        ->Args({1024, 1024, 1024, 32, -1})
        ->Args({2048, 2048, 2048, 32, -1})
        ->Args({4096, 4096, 4096, 32, -1});

/*
BENCHMARK_CAPTURE(BM_GEMM, tiled_gemm, swiftware::hpp::gemmT2)->Args({512, 512, 512, 256, 32})
        ->Args({1024, 1024, 1024, 256, 32})
        ->Args({2048, 2048, 2048, 256, 32})
        ->Args({4096, 4096, 4096, 256, 32});
//TODO: add benchmark for vectorized GEMM
*/

BENCHMARK_MAIN();

