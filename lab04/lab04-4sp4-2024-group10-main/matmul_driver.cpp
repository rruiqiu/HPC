// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/opencl.h>
#endif

#include <iostream>
#include <benchmark/benchmark.h>
#include <chrono>

#include "err_code.h"
#include "gemm.h"

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


//single row decomp( done in opencl)
const char *KernelSourceV1 = "\n" \
"__kernel void matmul_v1(                                                   \n" \
"   __global float* a,                                                      \n" \
"   __global float* b,                                                      \n" \
"   __global float* c,                                                      \n" \
"   const int M,                                                            \n" \
"   const int N,                                                            \n" \
"   const int K)                                                            \n" \
"{                                                                          \n" \
"   int i = get_global_id(0);                                               \n" \
"   for (int j=0;j<N;++j){                                                  \n" \
"        float sum = 0.0f;                                                  \n" \
"        for (int l=0;l<K;++l){                                             \n" \
"           sum += a[i * K + l] * b[l * N + j];                             \n" \
"        }                                                                  \n" \
"        c[i*N+j] = sum;                                                    \n"\
"   }                                                                       \n" \
"                                                                           \n" \
"                                                                           \n" \
"}                                                                          \n" \
"\n";


//swtiched to CUDA for this part
const char *KernelSourceV2 = "\n" \
"__kernel void matmul_v1(                                                    \n" \
"    __global float* a,                                                      \n" \
"    __global float* b,                                                      \n" \
"    __global float* c,                                                      \n" \
"    const int M,                                                            \n" \
"    const int N,                                                            \n" \
"    const int K,                                                            \n" \
"    const int tile_size)                                                    \n" \
"{                                                                           \n" \
"    // find the staring row                                                 \n" \
"    int row_start = get_global_id(0) * tile_size;                           \n" \
"    for (int i = 0; i < tile_size; ++i) {                                   \n" \
"        int row = row_start + i;                                            \n" \
"                                                                            \n" \
"        if (row < M) {                                                      \n" \
"            for (int j = 0; j < N; ++j) {                                   \n" \
"                float sum = 0.0f;                                           \n" \
"                for (int l = 0; l < K; ++l) {                               \n" \
"                    sum += a[row * K + l] * b[l * N + j];                   \n" \
"                }                                                           \n" \
"                c[row * N + j] = sum;                                       \n" \
"            }                                                               \n" \
"        }                                                                   \n" \
"    }                                                                       \n" \
"}                                                                          \n" \
"\n";

const char *KernelSourceV3 = "\n" \
"__kernel void matmul_v1(                                                 \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const int M,                                                  \n" \
"   const int N,                                                  \n" \
"   const int K)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"                                                       \n" \
"}                                                                      \n" \
"\n";

static void BM_MATMUL_OPENCL(benchmark::State &state,
                             const char *kernelSource) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    int j1 = state.range(3);

    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);
    auto *expected = new swiftware::hpp::DenseMatrix(m, n);

    for (int i = 0; i < m * k; ++i) {
        A->data[i] = 1.0;
    }
    for (int i = 0; i < k * n; ++i) {
        B->data[i] = 1.0;
    }
    // declare device containers
    // Initialize OpenCL
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;

    // Get the number of platforms
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);

    // Get the platforms
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms,
                     NULL);

    // Find a GPU platform
    cl_bool found_gpu_platform = CL_FALSE;
    for (cl_uint i = 0; i < num_platforms; ++i) {
        cl_uint num_devices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);

        if (num_devices > 0) {
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
            found_gpu_platform = CL_TRUE;
            break;
        }
    }

    if (!found_gpu_platform) {
        std::cerr << "No GPU device found." << std::endl;
    }

    // Get device information
    char device_name[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);


    int err;
// Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");
    // create cl_queue_properties
    cl_queue_properties *properties = new cl_queue_properties[3];
    properties[0] = CL_QUEUE_PROPERTIES;
    properties[1] = CL_QUEUE_PROFILING_ENABLE;
    properties[2] = 0;

    // Create a command queue
    command_queue = clCreateCommandQueueWithProperties(context, device_id,
                                                       properties, &err);
    checkError(err, "Creating command queue");


    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, &err);
    checkError(err, "Creating program");

    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
    }

    // Create the compute kernel from the program
    kernel = clCreateKernel(program, "matmul_v1", &err);
    checkError(err, "Creating kernel");

    // Create device buffers
    cl_mem buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m * k * sizeof(float), A->data.data(), NULL);
    cl_mem buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, k * n * sizeof(float), B->data.data(), NULL);
    cl_mem buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, m * n * sizeof(float), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffer_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&buffer_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&buffer_c);
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&m);
    clSetKernelArg(kernel, 4, sizeof(int), (void *)&n);
    clSetKernelArg(kernel, 5, sizeof(int), (void *)&k);

    //single row decomp
    size_t global_work_size = m;  // total number of work-items
    size_t local_work_size = j1; //j1 work itesm per gorup
    // add gpu name to the log
    state.SetLabel(device_name);
    for (auto _: state) {
        // Execute kernel
        cl_event event;
        cl_ulong time_start = 0;
        cl_ulong time_end = 0;

        clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_size1, 0, NULL, NULL);

        // Wait for kernel to finish
        clFinish(command_queue);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        cl_ulong elapsed = time_end - time_start;
        // Read results back to host
        clEnqueueReadBuffer(command_queue, buffer_c, CL_TRUE, 0, m * sizeof(float), C->data.data(), 0, NULL, NULL);
        
        swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), expected->data.data(), swiftware::hpp::ScheduleParams(32, 32, 8, 1));
        // Print first 10 elements of the result
        
        //test the code
        /*
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                auto c_val = C->data.data()[i];
                auto c_exp = expected->data.data()[i];
                if (c_val != c_exp){
                    std::cout << "Result: " <<  C->data.data()[i] << "Expected: " << expected->data.data()[i] <<std::endl;
                }
            }
        }
        std::cout<<"done checking"<<std::endl;
        */
        

        state.SetIterationTime(elapsed);

    }

    // Clean up
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    clReleaseMemObject(buffer_a);
    clReleaseMemObject(buffer_b);
    clReleaseMemObject(buffer_c);
    free(platforms);
    delete A;
    delete B;
    delete C;

}



extern float matmul_single_row_Wrapper(float* h_A, float* h_B, float* h_C, int m, int n, int k);
extern float matmul_mutiple_rows_Wrapper(float* h_A, float* h_B, float* h_C, int m, int n, int k, int tiling_size);
extern float matmul_mutiple_rowA_colB_Wrapper(float* h_A, float* h_B, float* h_C, int m, int n, int k, int tiling_size_row_A,int tiling_size_col_B);

static void BM_VECMUL_CUDA_mutiple_rows_colB(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    int t1 = state.range(3);
    int t2 = state.range(4);
    // int cs = state.range(5); // chunk size
    // int nt = state.range(6); // number of threads
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

    const char *device_name = "GPU";

    // add gpu name to the log
    state.SetLabel(device_name);
    for (auto _: state) {
        float elapsed = matmul_mutiple_rowA_colB_Wrapper(A->data.data(), B->data.data(), C->data.data(), m, n, k,t1,t2);
        state.SetIterationTime(elapsed*1e-3);
    }
    delete A;
    delete B;
    delete C;

}


static void BM_VECMUL_CUDA_mutiple_rows(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    int t1 = state.range(3);
    // int t2 = state.range(4);
    // int cs = state.range(5); // chunk size
    // int nt = state.range(6); // number of threads
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

    const char *device_name = "GPU";

    // add gpu name to the log
    state.SetLabel(device_name);
    for (auto _: state) {
        float elapsed = matmul_mutiple_rows_Wrapper(A->data.data(), B->data.data(), C->data.data(), m, n, k,t1);
        state.SetIterationTime(elapsed*1e-3);
    }
    delete A;
    delete B;
    delete C;

}

static void BM_VECMUL_CUDA(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    // int t1 = state.range(3);
    // int t2 = state.range(4);
    // int cs = state.range(5); // chunk size
    // int nt = state.range(6); // number of threads
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

    const char *device_name = "GPU";

    // add gpu name to the log
    state.SetLabel(device_name);
    for (auto _: state) {
        float elapsed = matmul_single_row_Wrapper(A->data.data(), B->data.data(), C->data.data(), m, n, k);
        state.SetIterationTime(elapsed*1e-3);
    }
    delete A;
    delete B;
    delete C;

}


//Args format (m, n, k, Sp.TileSize1, Sp.TileSize2, chunk size, number of threads, samplingRatePercentage)
BENCHMARK_CAPTURE(BM_GEMM, gemm_optimized, swiftware::hpp::gemmEfficientParallel)
    // ->Args({512, 512, 512, 256, 32, 1, NUM_THREADS, 1})
    // ->Args({1024, 1024, 1024, 256, 32, 1, NUM_THREADS, 1})
    // ->Args({2048, 2048, 2048, 256, 32, 1, NUM_THREADS, 1})
    ->Args({32,4096,  4096, 256, 32, 1, NUM_THREADS, 1});

// BENCHMARK_CAPTURE(BM_MATMUL_OPENCL, opencl_matmul_v1, KernelSourceV1)->Args({512, 512, 512})->UseManualTime()->Iterations(100);
BENCHMARK_CAPTURE(BM_MATMUL_OPENCL, opencl_matmul_v1, KernelSourceV1)
->Args({40, 4096, 4096, 1})->UseManualTime()->Iterations(1)
->Args({40, 4096, 4096, 16})->UseManualTime()->Iterations(1)
->Args({40, 4096, 4096, 32})->UseManualTime()->Iterations(1)
->Args({40, 4096, 4096, 64})->UseManualTime()->Iterations(1)
->Args({40, 4096, 4096, 128})->UseManualTime()->Iterations(1)
->Args({40, 4096, 4096, 256})->UseManualTime()->Iterations(1);

BENCHMARK(BM_VECMUL_CUDA)
//     ->Args({512, 512, 512})
//     ->Args({1024, 1024, 1024})
//     ->Args({2048, 2048, 2048})
    ->Args({32,4096,  4096})
    // ->Iterations(1)
    ->UseManualTime();

BENCHMARK(BM_VECMUL_CUDA_mutiple_rows_colB)
    ->Args({32,4096,  4096,1,1})
    ->Args({32,4096,  4096,1,2})
    ->Args({32,4096,  4096,1,4})
    ->Args({32,4096,  4096,1,8})
    ->Args({32,4096,  4096,1,16})
    ->Args({32,4096,  4096,1,32})
    ->Args({32,4096,  4096,1,64})
        ->Args({32,4096,  4096,1,128})
            ->Args({32,4096,  4096,1,256})
                ->Args({32,4096,  4096,1,512})

    ->UseManualTime();

// BENCHMARK(BM_VECMUL_CUDA_mutiple_rows)
//     // ->Args({4096, 32, 4096,2})
//     // ->Args({4096, 32, 4096,4})
//     // ->Args({4096, 32, 4096,8})
//     // ->Args({4096, 32, 4096,16})
//     // ->Args({4096, 32, 4096,32})
//     // ->Args({4096, 32, 4096,64})
//     // ->Args({4096, 32, 4096,128})
//     // ->Args({4096, 32, 4096,256})
//     // ->Args({4096, 32, 4096,512})
//     ->Args({32, 4096, 4096,2})
//     ->Args({32, 4096, 4096,4})
//     ->Args({32, 4096, 4096,8})
//     ->Args({32, 4096, 4096,16})
//     ->Args({32, 4096, 4096,32})
//     ->Args({32, 4096, 4096,64})
//     ->Args({32, 4096, 4096,128})
//     ->Args({32, 4096, 4096,256})
//     ->Args({32, 4096, 4096,512})
//     ->UseManualTime();





// TODO add other versions of matmul



BENCHMARK_MAIN();

