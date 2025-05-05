[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/80hIyIza)

# CE 4SP4 - Lab3

## Description
In Lab 3, you will implement parallel matrix-matrix multiplication and integration calculation using OpenMP.

## Objectives
- Parallel programming in multi-core CPUs using OpenMP
- Profiling the performance of the implementations and analyzing the results

## Some notes
* You will need to first finish the TODO items in the code and then change your implementations. Please remove TODOs 
once implement them.
 to answer following questions in the report (see questions in the following parts). Your implementations should work correctly for ANY cornel cases.
* Please push a PDF of your report in the `doc` folder.
The `README` file in the `doc` folder provides some hints how to prepare a good report. 
* The evaluation is based on the quality of the report and 
the correctness/efficiency of the implementation. So make sure to provide all the necessary details in the report.
The report should be single column with font size 11 or 12. The report should be at most 6 pages long, 
including figures, references, tables,.... . A good report should be clear and concise and backed up with data/plots.
* A good sample for plots and tables are those provided in lecture slides.
* When profiling with `perf`, you may not be able to collect all counters at once.
* All group members should contribute to all parts of the lab. If contributions are not equal, please specify at the 
end of the report. 

## Installation and Running the Code
The installation instructions for the teach cluster is provided as a bash script. You will need to clone 
the repository. Then you can emit the 
following commands to install the code on the teach cluster after cloning the repository:

```bash
cd 4sp4-lab03 folder
bash build.sh
```

To run the code on the teach cluster:
```bash
sbatch run_teach_cluster.sh mm
```
For running the second part, use `integration` instead of `mm`.

**Please do NOT edit SLURM part of the script.**

## Part 1: Parallel Matrix-Matrix Multiplication
- Implement each of the following matrix-matrix multiplication algorithms in parallel using OpenMP. 
  For each implementation, pick an optimized sequential baseline and make it parallel. 
  Your implementations should be efficient for square matrices, tall skinny matrices, and short wide matrices. 
  In other word, you should check dimension of matrices and pick a proper implementation for these three categories of matrices.
    - 1-1-Dense Matrix-Matrix Multiplication (GEMM) 
    - 1-2- Sparse Matrix-Matrix Multiplication (SpMM)
- Note: the ratios of number of rows to number of columns in tall skinny matrices and short wide matrices are 100 and 0.01, respectively.
- In your report, for each of thw two kernel, you will need to discuss:
    - Speedup plots over the sequential baseline for each implementation.
    - The effect of scheduling and chunk size on the performance of the parallel implementation.
    - For SpMM only, the effect of the sparsity of the matrices on the performance of the parallel implementation.
    - Scalability up to 8 threads on the teach cluster. 


## Part 2: Integration calculation 
- Implement the code to calculate Pi and make it parallel using OpenMP using two approaches: 
  - 1- You are only allowed to use `pragma omp parallel` and `omp_get_thread_num()`. - You should map each step of the integration calculation to a thread. Consecutive steps should be assigned to different threads. For example, step 1 should be assigned to thread 1, step 2 to thread 2, and so on.
  - 2- You are allowed to use any OpenMP directive.
- Plot the speedup over the sequential baseline and discuss your task decomposition and scheduling strategy in the report.
- Scalability up to 8 threads on the teach cluster.

### Integration Description:
The goal of integration in part 2 is to calculate the value of Pi using the following integral:
```
Pi = 4 * integral(0, 1, 1/(1+x^2) dx)
```
The integral is calculated using a simplified trapezoidal rule. 
The integral is divided into n equal subintervals. 
The width of each subinterval is `h = 1/n`, where `h` is the step size and `n` is the number of steps. 
The integral is then approximated by the sum of the areas of the rectangles formed by the function and the x-axis over each subinterval.
In other words, the integral is approximated by the sum of the areas of the rectangles:
```
Pi = F(0.5h) * h + F(1.5h) * h + F(2.5h) * h + ... + F(((n-1)+0.5)h)) * h
```
where `F(x) = 1/(1+x^2)` is the function to be integrated.
The width of each rectangle is `h` and the height is the value of the function at the midpoint of the subinterval.