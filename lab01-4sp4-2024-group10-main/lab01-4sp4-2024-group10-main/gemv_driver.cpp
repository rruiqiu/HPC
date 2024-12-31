//
// Created by kazem on 06/09/24.
//

#include <benchmark/benchmark.h>
#include "def.h"
#include "gemv.h"


// TODO : Implement the benchmark

static void BM_GEMV(benchmark::State &state, void (*gemvImpl1)(int M, int N, const float *A, const float *x, float *y,
                                                               swiftware::hpp::ScheduleParams Sp)) {

    int m = state.range(0);
    int n = state.range(1);
    int t1 = state.range(2);
    int t2 = state.range(3);

    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *x = new swiftware::hpp::DenseMatrix(n, 1);
    auto *y = new swiftware::hpp::DenseMatrix(m, 1);
    for(int i=0;i<m*n;i++){
        A->data[i] = 1.0;
    }

    for(int i=0;i<n;i++){
        x->data[i] =  1.0;
    }
    for (auto _: state) {
        // TODO
        gemvImpl1(m, n, A->data.data(), x->data.data(), y->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }

    delete A;
    delete x;
    delete y;
}


//gemvT1
BENCHMARK_CAPTURE(BM_GEMV,vec_gemv, swiftware::hpp::gemvVec)
        ->Args({4096, 4096,2096, -1})
        ->Args({4096, 4096,4096, -1});
BENCHMARK_MAIN();