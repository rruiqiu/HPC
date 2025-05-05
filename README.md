# High Performance Computing Labs

## Lab 1 – Dense Linear Algebra Optimization  
Optimized General Matrix-Matrix Multiplication (GEMM) and General Matrix-Vector Multiplication (GEMV) using cache-aware tiling and AVX vectorization for improved CPU performance. View the full report [here]()

### Performance Speedup summary on Dense Neural Network: 

#### **GEMM (General Matrix-Matrix Multiplication)**

- Baseline (GEMM): 6,851,770,000 ns (6.85 seconds)
- Optimized (GEMMVec with tiling and AVX vectorization): 4,016,030,000 ns (4.02 seconds)
- Improvement: 2,835,740,000 ns (2.84 seconds)
- Speedup: Approximately **41.4%**

#### **GEMV (General Matrix-Vector Multiplication)**

- Baseline (GEMV): 4,554,450 ns (0.00455 seconds)
- Optimized (GEMVVec with AVX vectorization): 1,395,850 ns (0.00140 seconds)
- Improvement: 3,155,950 ns (0.00316 seconds)
- Speedup: Approximately **69.4%**



## Lab 2 – Sparse Matrix Operations  
Implemented Sparse Matrix-Matrix Multiplication (SpMM) and Sparse Matrix-Vector Multiplication (SpMV) using efficient sparse data structures, tiling, and vectorization to accelerate computation on sparse inputs. View the full report [here]()

#### SpMM (Sparse Matrix-Matrix Multiplication)

- **CSR-optimized SpMM** achieved up to **3× speedup** over the **baseline SpMM** at low sampling percentages (e.g., 3–5%).
- For **sparse matrices (sampling percentage < 6%)**, CSR-based SpMM **outperformed GEMM** in execution time.
- At **sampling percentage > 30%**, GEMM became more efficient than sparse methods due to the diminishing advantage of sparsity and increased cache locality in dense access patterns.

#### SpMV (Sparse Matrix-Vector Multiplication)

- **Optimized SpMV (vectorized and tiled)** achieved over **2× speedup** compared to **baseline SpMV** on 4k×4k matrices with 20% density.
- **GEMV** outperformed **SpMV** once density exceeded **50%**, due to:
  - Lower memory access overhead (3 loads vs. 4)
  - Better prefetching and SIMD utilization in GEMV
- **SpMV optimized implementation** showed **near 0% cache miss rate** at high sparsity, leading to reduced execution time in sparse conditions.

## Lab 3 – Parallel Matrix Multiplication with OpenMP  
Introduced OpenMP-based parallelism to speed up GEMM and SpMM computations, with configurable thread count and loop scheduling strategy. View the full report [here]()

#### GEMM (General Matrix-Matrix Multiplication with OpenMP)

- **Square matrix (4096×4096):** Achieved up to **~8× speedup** with 8 threads compared to the baseline sequential implementation.
- **Tall/Skinny matrix (4096×40):** Also reached **~8× speedup**, confirming effective utilization of i-loop parallelism and cache-friendly access.
- **Short/Wide matrix (40×4096):** Reached **~6× speedup**, with optimal results from j-loop parallelism due to better thread utilization and reduced cache conflict.

#### SPMM (Sparse Matrix-Matrix Multiplication)

- For **sparse square matrices (4096×4096)** with sampling percentages between **6%–9%**, the best chunk size was **32**, achieving:
  - **Up to ~6.6× speedup** using **dynamic scheduling**
  - **Up to ~5.9× speedup** using **static scheduling**
- **Tall/Skinny SPMM (4096×32)**:
  - Best speedup (~**6.6×**) achieved with **chunk size = 256**
- **Short/Wide SPMM (32×4096)**:
  - Parallelizing the inner loop (columns) resulted in only **~0.33× speedup**, so outer-loop (i-loop) parallelism was used instead.
  - Dynamic and static scheduling performed similarly due to uniform sparsity.

#### Numerical Integration (Pi Calculation)

- Parallel OpenMP implementation of numerical integration achieved a **~6.9× speedup** with 8 threads over the sequential version.
- **Static scheduling** with default chunk size yielded the best runtime due to uniform iteration cost and minimal overhead.

## Lab 4 – GPU Acceleration with CUDA and OpenCL  
Developed and optimized GEMM implementations using both CUDA and OpenCL, leveraging GPU parallelism and memory management for high-throughput matrix computation. View the full report [here]()

#### Implementation 1: OpenCL – Row-Based Decomposition

- **Square Matrix (4096×4096)**: Best local workgroup size was **64**, achieving **up to 2× speedup** over OpenMP baseline.
- **Tall/Skinny Matrix (e.g., 4096×32)**: Achieved best performance with workgroup size **32**, due to high thread utilization on row-dominant matrices.
- **Short/Fat Matrix (e.g., 32×4096)**: Best performance at **workgroup size = 1**, due to limited active rows and underutilization of larger thread blocks.

#### Implementation 2: CUDA – 1D Tiling on Rows of A

- **Block size 256** provided the best speedup (~**2×**) on square matrices using RTX 4060.
- Increasing **tiled row size** from 1 to 512 resulted in **decreasing speedup**, due to more work per thread and reduced GPU parallelism.

#### Implementation 3: CUDA – 1D Tiling on A + 1D Tiling on B

- **Tall/Skinny Matrix (4096×32)** with **row tile size = 1** and **column tile size = 4** achieved maximum speedup of **~28.88×**.
- **Short/Wide Matrix (32×4096)** with **row tile size = 1** and **column tile size = 256** achieved speedup of **~6.95×**.
- Best overall performance was achieved with:
  - **Row tile size = 1**
  - **Column tile size = 4**
