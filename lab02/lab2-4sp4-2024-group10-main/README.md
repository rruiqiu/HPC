[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/mTCelUc_)

# CE 4SP4 - Lab2

## Description
In Lab 2, you will learn how to use sparsity to optimize matrix-matrix and matrix-vector multiplication.
You will also (optionally) implement a sparse neural network using your optimized matrix-matrix and matrix-vector multiplication.

## Objectives
- Using data structures for efficient sparse matrix representation and operations
- Implement sparse matrix-matrix multiplication (SpMM) and sparse matrix-vector multiplication with tiling and vectorization
- Using SpMM inside a neural network (excluded)
- Profiling the performance of the implementations and analyzing the results

## Some notes
* You will need to first finish the TODO items in lab 2 and then change your implementations. Please remove TODOs 
once implement them.
 to answer following questions in the report (see questions in the three parts). Your implementations should work for ANY cornel cases.
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
cd 4sp4-lab02
bash build.sh
```

To run the code on the teach cluster:
```bash
sbatch run_teach_cluster.sh spmm
```
For running other experiments use spnn or spmv instead of spmm.

**Please do NOT edit SLURM part of the script.**

## Part 1: Sparse Matrix-Matrix Multiplication (SpMM)
- Optimize SpMM and explain your tiling and vectorization strategy and discuss your optimization strategy.
- Compare your SpMM performance with your optimized GEMM (from Lab1) and GEMM Skipping variant performance and explain the difference.
- How does the sparsity of the matrix affect the performance of SpMM compared to GEMM?
- Please provide necessary profiling and experiment to back up your discussions above.

## Part 2: Sparse Matrix-Vector Multiplication (SpMV)
- Implement SpMV and explain your tiling and vectorization strategy and discuss your optimization strategy.
- Compare your SpMV performance with your optimized GEMV (from Lab1) and GEMV Skipping variant performance and explain the difference.
- How does the sparsity of the matrix affect the performance of SpMV compared to other implementations?
- Please provide necessary profiling and experiment to back up your discussions above.

## Part 3: Sparse Neural Network (not graded)

NOTE: This part of lab 2 is not graded. You can skip this part. If you like to do it, you can do it for fun.

For this part, you will need to copy weight matrices and biases from the provided shared directory in
the teach cluster in the `data` folder:
 ```bash
cp -r /home/l/lcl_uotce4sp4/ce4sp4starter/data .
```
This folder contains sparse trained weight matrices and the MNIST dataset. Functions for reading these files are provided.
The MNIST dataset includes features and labels. The labels are located in the first column of the dataset after loading.

- Implement a sparse neural network using your optimized SpMM. 
- Provide a breakdown of execution times of different calls in sparse NN. How does breakdown change with your optimized code?
- How much does your optimization improve the performance NN end to end? 


### DNN Description:
A DNN with two linear layers and two non-linear activation functions. This model is identical to lab 1, 
but the matrix-matrix and matrix-vector multiplications are replaced with the optimized sparse versions. This is 
because the weight matrices, `W` and `W2` are sparse.

#### Mathematical Representation:

Let's denote:

`X:` Input vector (n-dimensional)

`W1:` Weights matrix for the first layer (h x n)

`b1:` Biases vector for the first layer (h-dimensional)

`W2:` Weights matrix for the second layer (m x h)

`b2:` Biases vector for the second layer (m-dimensional)

`H:` Hidden layer activations (h-dimensional)

`Z:` Output vector before softmax (m-dimensional)

`Y:` Output vector (m-dimensional)

The forward pass of a DNN with two layers and softmax can be represented as follows:

`H = tanh(X * W1^T + b1)`

`Z = sigmoid(H * W2^T + b2)`

`Y = argmax(Z)`

Where:


tanh is an activation function, defined as:
`softmax(Z) = (exp(Z) - exp(-Z)) / (exp(Z) + exp(-Z))`.
sigmoid is another commonly used activation function in neural networks, defined as `1/(1 + exp(-z))`. argmax returns the index of the maximum element in the vector.

Explanation:

*Input Layer:* The input data is fed into the input layer.

*Hidden Layer:* The input is multiplied by the weights of the first layer, and the biases are added. The result is passed through the tanh activation function.

*Output Layer:* The activations from the hidden layer are multiplied by the weights of the second layer, and the biases are added. The result is passed through the sigmoid activation function to normalize the outputs into a probability distribution.

*Prediction:* Eventually argmax is used to predict the class label based on the output vector Z.
