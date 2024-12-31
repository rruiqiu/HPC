// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab
// #define NUM_THREADS 8

#include "spmm.h"
#include <immintrin.h>
namespace swiftware::hpp {

  void spmmCSR(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        C[i * n + j] = 0;
        for (int l = Ap[i]; l < Ap[i + 1]; ++l) {
          C[i * n + j] += Ax[l] * B[Ai[l] * n + j];
        }
      }
    }
  }
  
  void spmmCSREfficientSequential(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp) {
    // TODO: use an efficient sequential spmm implementation
    for (int i = 0; i < m; ++i) {
      for (int j = Ap[i]; j < Ap[i + 1]; ++j){
        auto a_col_length = Ap[i+1] - Ap[i];
        auto min_length = std::min(a_col_length,n);
        auto col_bnd = min_length - min_length%32;
        auto a_val = _mm256_set1_ps(Ax[j]);

        for (int l = 0; l+32 <= (min_length); l+=32) {
          __m256 b_vec1 = _mm256_loadu_ps(&B[Ai[j]*n + l]);
          __m256 b_vec2 = _mm256_loadu_ps(&B[Ai[j]*n + l + 8]);
          __m256 b_vec3 = _mm256_loadu_ps(&B[Ai[j]*n + l + 16]);
          __m256 b_vec4 = _mm256_loadu_ps(&B[Ai[j]*n + l + 24]);

          __m256 mul_vec1 = _mm256_mul_ps(a_val, b_vec1);
          __m256 mul_vec2 = _mm256_mul_ps(a_val, b_vec2);
          __m256 mul_vec3 = _mm256_mul_ps(a_val, b_vec3);
          __m256 mul_vec4 = _mm256_mul_ps(a_val, b_vec4);

          __m256 c_vec1 = _mm256_loadu_ps(&C[i*n + l]);
          __m256 c_vec2 = _mm256_loadu_ps(&C[i*n + l + 8]);
          __m256 c_vec3 = _mm256_loadu_ps(&C[i*n + l + 16]);
          __m256 c_vec4 = _mm256_loadu_ps(&C[i*n + l + 24]);

          c_vec1 = _mm256_add_ps(c_vec1,mul_vec1);
          c_vec2 = _mm256_add_ps(c_vec2,mul_vec2);
          c_vec3 = _mm256_add_ps(c_vec3,mul_vec3);
          c_vec4 = _mm256_add_ps(c_vec4,mul_vec4);

          _mm256_storeu_ps(&C[i*n + l], c_vec1);
          _mm256_storeu_ps(&C[i*n + l + 8], c_vec2);
          _mm256_storeu_ps(&C[i*n + l + 16], c_vec3);
          _mm256_storeu_ps(&C[i*n + l + 24], c_vec4);
        }
        for(int jj = col_bnd; jj < n; jj++){
          C[i*n + jj] +=  Ax[j] * B[Ai[j]*n + jj];
        }
      }
    }
  }

  void spmmCSREfficientParallel(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp) {
    // TODO: Implement skipping version of spmm and optimize it

    int chunk_size = Sp.ChunkSize;
    int num_thread = Sp.NumThreads;

    //dynamcially parallel the outer loop
    #pragma omp parallel for schedule(dynamic,chunk_size) num_threads(num_thread)
    // #pragma omp parallel for schedule(static) num_threads(num_thread)
    for (int i = 0; i < m; ++i) {
      for (int j = Ap[i]; j < Ap[i + 1]; ++j){
        auto a_col_length = Ap[i+1] - Ap[i];
        auto min_length = std::min(a_col_length,n);
        auto col_bnd = min_length - min_length%32;
        auto a_val = _mm256_set1_ps(Ax[j]);
        // #pragma omp parallel for num_threads(num_thread)
        for (int l = 0; l < col_bnd; l+=32) {
          __m256 b_vec1 = _mm256_loadu_ps(&B[Ai[j]*n + l]);
          __m256 b_vec2 = _mm256_loadu_ps(&B[Ai[j]*n + l + 8]);
          __m256 b_vec3 = _mm256_loadu_ps(&B[Ai[j]*n + l + 16]);
          __m256 b_vec4 = _mm256_loadu_ps(&B[Ai[j]*n + l + 24]);

          __m256 mul_vec1 = _mm256_mul_ps(a_val, b_vec1);
          __m256 mul_vec2 = _mm256_mul_ps(a_val, b_vec2);
          __m256 mul_vec3 = _mm256_mul_ps(a_val, b_vec3);
          __m256 mul_vec4 = _mm256_mul_ps(a_val, b_vec4);

          __m256 c_vec1 = _mm256_loadu_ps(&C[i*n + l]);
          __m256 c_vec2 = _mm256_loadu_ps(&C[i*n + l + 8]);
          __m256 c_vec3 = _mm256_loadu_ps(&C[i*n + l + 16]);
          __m256 c_vec4 = _mm256_loadu_ps(&C[i*n + l + 24]);

          c_vec1 = _mm256_add_ps(c_vec1,mul_vec1);
          c_vec2 = _mm256_add_ps(c_vec2,mul_vec2);
          c_vec3 = _mm256_add_ps(c_vec3,mul_vec3);
          c_vec4 = _mm256_add_ps(c_vec4,mul_vec4);

          _mm256_storeu_ps(&C[i*n + l], c_vec1);
          _mm256_storeu_ps(&C[i*n + l + 8], c_vec2);
          _mm256_storeu_ps(&C[i*n + l + 16], c_vec3);
          _mm256_storeu_ps(&C[i*n + l + 24], c_vec4);
        }
        for(int jj = col_bnd; jj < n; jj++){
          C[i*n + jj] +=  Ax[j] * B[Ai[j]*n + jj];
        }
      }
    }
  }

}