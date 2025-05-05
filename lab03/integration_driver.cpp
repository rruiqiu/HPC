// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#include <benchmark/benchmark.h>
#include "integration.h"

// TODO : Implement the benchmark
#define NUM_THREADS 8

static void BM_INTEGRATION(benchmark::State &state,
                       void (*numericIntegration)(swiftware::hpp::IntegrationParams *params,
                                                                   swiftware::hpp::ScheduleParams Sp)) {
  auto numSteps = state.range(0);
  auto chunkSize = state.range(1);
  auto numThreads = state.range(2);

  //TODO : Implement the benchmark
  swiftware::hpp::IntegrationParams *params;
  swiftware::hpp::ScheduleParams sp(1, 1, numThreads, chunkSize);
  for (auto _: state) {
    // Running the NN function
    swiftware::hpp::IntegrationParams params(numSteps);
    numericIntegration(&params, sp);
  }

}


// Arguments: NumSteps, ChunkSize, NumThreads
/*
BENCHMARK_CAPTURE(BM_INTEGRATION, integration_serial, swiftware::hpp::integrateSequential)
    ->Unit(benchmark::kMillisecond)->Args({1000000, 1, 1})->Args({10000000, 1, 1})->Args({100000000, 1, 1});
BENCHMARK_CAPTURE(BM_INTEGRATION, integration_parallel, swiftware::hpp::integrateParallel)->Unit(benchmark::kMillisecond)
    ->Args({100000000, 1, NUM_THREADS})->Args({100000000, 12500000, NUM_THREADS})->Args({100000000, 100000, NUM_THREADS})->Args({100000000, 15000000, NUM_THREADS});
BENCHMARK_CAPTURE(BM_INTEGRATION, integration_parallel_omp, swiftware::hpp::integrateParallelWithAllOMP)->Unit(benchmark::kMillisecond)
     ->Args({100000000, 1, NUM_THREADS})->Args({100000000, 12500000, NUM_THREADS})->Args({100000000, 100000, NUM_THREADS})->Args({100000000, 15000000, NUM_THREADS});
BENCHMARK_CAPTURE(BM_INTEGRATION, integration_parallel_omp_dy, swiftware::hpp::integrateParallelWithAllOMP_dy)->Unit(benchmark::kMillisecond)
     ->Args({100000000, 1, NUM_THREADS})->Args({100000000, 12500000, NUM_THREADS})->Args({100000000, 100000, NUM_THREADS})->Args({100000000, 15000000, NUM_THREADS});
*/
BENCHMARK_CAPTURE(BM_INTEGRATION, integration_parallel, swiftware::hpp::integrateParallel)->Unit(benchmark::kMillisecond)
    ->Args({100000000000, 1, NUM_THREADS});
BENCHMARK_CAPTURE(BM_INTEGRATION, integration_parallel_omp, swiftware::hpp::integrateParallelWithAllOMP)->Unit(benchmark::kMillisecond)
     ->Args({100000000000, 1, NUM_THREADS});


BENCHMARK_MAIN();