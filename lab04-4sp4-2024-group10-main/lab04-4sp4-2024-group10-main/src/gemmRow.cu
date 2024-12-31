#include <iostream>

//col order
__global__ void matmul_single_row(int m, int n, int k, const float *A, const float *B, float *C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread processes only valid rows
    if (row < m) {
        // Traverse through shared dimension (k) and columns of B (n)
        for (int j = 0; j < n; ++j) { // Iterate over columns of B
            float sum = 0.0;
            for (int l = 0; l < k; ++l) {
                sum += A[row * k + l] * B[l * n + j];
            }
            C[row * n + j] = sum;
        }
    }
}
float matmul_single_row_Wrapper(float* h_A, float* h_B, float* h_C, int m, int n, int k) {

    // Calculate sizes for matrices
    size_t size_A = m * k * sizeof(float); // Size of matrix A
    size_t size_B = k * n * sizeof(float); // Size of matrix B
    size_t size_C = m * n * sizeof(float); // Size of matrix C

    // Allocate memory on the device (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice); // Initialize C to 0 on device

    // Launch the kernel with enough blocks and threads
    int threadsPerBlock = 256;  //define how many threads per block within the cuda, max cuda support is 1024
    int blocksPerGrid = (m + threadsPerBlock-1) / threadsPerBlock;

    // CUDA events to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);
    // Launch the kernel
    matmul_single_row<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, d_A, d_B, d_C);


    // Record the stop event
    cudaEventRecord(stop, 0);

    // Record stop event and synchronize
    cudaEventSynchronize(stop);
    // cudaEventCreate(&stop);

    // Calculate elapsed time
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Cleanup: Destroy events and free device memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return elapsed; // Return elapsed time in milliseconds
}

__global__ void matmul_mutiple_row(int m, int n, int k, const float *A, const float *B, float *C,int tiling_size) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index

    int start_row = thread_id * tiling_size;           // Starting row for this thread
    int end_row = min(start_row + tiling_size, m);     // Ending row (exclusive)

    for (int row = start_row; row < end_row; ++row) {      // Iterate through assigned rows
        for (int j = 0; j < n; ++j) {                     // Iterate through columns of B
            float sum = 0.0;
            for (int l = 0; l < k; ++l) {                 // Shared dimension
                sum += A[row * k + l] * B[l * n + j];
            }
            C[row * n + j] = sum;                         // Write result to C
        }
    }
}



float matmul_mutiple_rows_Wrapper(float* h_A, float* h_B, float* h_C, int m, int n, int k, int tiling_size) {

    // Calculate sizes for matrices
    size_t size_A = m * k * sizeof(float); // Size of matrix A
    size_t size_B = k * n * sizeof(float); // Size of matrix B
    size_t size_C = m * n * sizeof(float); // Size of matrix C

    // Allocate memory on the device (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice); // Initialize C to 0 on device

    // Launch the kernel with enough blocks and threads
    int threadsPerBlock = 256;  //define how many threads per block within the cuda, max cuda support is 1024
    int rows_per_thread = tiling_size; //define how many rows each thread process

    int total_rows = (m + rows_per_thread - 1) / rows_per_thread;
    int blocksPerGrid = (total_rows + threadsPerBlock-1) / threadsPerBlock;

    // CUDA events to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // Launch the kernel
    matmul_mutiple_row<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, d_A, d_B, d_C,rows_per_thread);

    // Record stop event and synchronize
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Cleanup: Destroy events and free device memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // std::cout << elapsed;
    return elapsed; // Return elapsed time in milliseconds
}



__global__ void matmul_multiple_rowA_colB(int m, int n, int k, const float *A, const float *B, float *C,int tiling_size_row_A,int tiling_size_col_B) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index
    
    int col_per_thread = n / tiling_size_col_B;

    int start_row = thread_id / tiling_size_col_B;           // Starting row for this thread
    int start_col = (thread_id % tiling_size_col_B) * col_per_thread;         //define the start row
    
    int end_row = min(start_row + tiling_size_row_A, m);     // Ending row (exclusive)
    
    // int end_col = min(start_col + col_per_thread, n);
    int end_col = thread_id % tiling_size_col_B == tiling_size_col_B-1 ? n : start_col + col_per_thread; //handle the last columns of B


    // printf("Thread ID: %d, Start Row: %d, Start Col: %d\n", thread_id, start_row, start_col); 
    for (int row = start_row; row < end_row; ++row) {      // Iterate through assigned rows
        for (int j = start_col; j < end_col; ++j) {                     // Iterate through columns of B
            float sum = 0.0;
            for (int l = 0; l < k; ++l) {                 // Shared dimension
                sum += A[row * k + l] * B[l * n + j];
            }
            C[row * n + j] = sum;                         // Write result to C
        }
    }
}

float matmul_mutiple_rowA_colB_Wrapper(float* h_A, float* h_B, float* h_C, int m, int n, int k, int tiling_size_row_A,int tiling_size_col_B) {

    // Calculate sizes for matrices
    size_t size_A = m * k * sizeof(float); // Size of matrix A
    size_t size_B = k * n * sizeof(float); // Size of matrix B
    size_t size_C = m * n * sizeof(float); // Size of matrix C

    // Allocate memory on the device (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice); // Initialize C to 0 on device

    // Calculate the total number of tiles
    int num_tiles_row = (m + tiling_size_row_A - 1) / tiling_size_row_A; // Number of row tiles
    int num_tiles_col = (n + tiling_size_col_B - 1) / tiling_size_col_B; // Number of column tiles
    int total_tiles = num_tiles_row * num_tiles_col;                     // Total number of tiles

    // Configure threads and blocks
    int threadsPerBlock = 256;  // Number of threads per block
    int blocksPerGrid = (total_tiles + threadsPerBlock - 1) / threadsPerBlock;



    // CUDA events to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);


    // Launch the kernel
    matmul_multiple_rowA_colB<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, d_A, d_B, d_C, tiling_size_row_A, tiling_size_col_B);

    // Record stop event and synchronize
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Cleanup: Destroy events and free device memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return elapsed; // Return elapsed time in milliseconds
}