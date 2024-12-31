[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/muUorIj7)
# CE 4SP4 - Lab1

## Description
In Lab 1, you will optimize neural network on a single thread through optimizing matrix multiplication. Deep neural networks are composed of many layers of matrix multiplications and
non-linear operations. In each layer, the input is multiplied by a weight matrix and then passed
through a non-linear activation function, e.g. softmax. The output of the layer is then used as input to the next
layer. The matrix multiplication is the most computationally expensive operation in a neural
network. In this lab, you will practice how to optimize matrix multiplication for dense matrices on 
single-core CPU. You will implement tiling and vectorization for matrix-matrix and matrix-vector 
multiplication.

## Objectives
- Implement matrix-matrix multiplication with tiling and vectorization
- Implement matrix-vector multiplication with tiling and vectorization
- Using both matrix-matrix and matrix-vector multiplications inside a neural network
- Profiling the performance of the implementations and analyzing the results

## Some notes
* You will need to first finish the TODO items in lab 1 and then change your implementations
 to answer following questions in the report. Your implementations should work for ANY cornel cases.
* Please push a PDF of your report in the `doc` folder.
The `README` file in the `doc` folder provides some hints how to prepare a good report. 
* The evaluation is based on the quality of the report and 
the correctness/efficiency of the implementation. So make sure to provide all the necessary details in the report.
The report should be single column with font size 11 or 12. The report should be at most 6 pages long, 
including figures, references, tables,.... . A good report should be clear and concise and backed up with data/plots.
* A good sample for plots and tables are those provided in lecture slides.
* When profiling with `perf`, you may not be able to collect all counters at once.
* Do NOT test more than 4K x 4K matrices. It is not needed.
* All group members should contribute to all parts of the lab. If contributions are not equal, please specify at the 
end of the report. 
* When doing vectorization, please make sure to pick instructions supported by the teach CPU.

## Installation and Running the Code
The installation instructions for the teach cluster is provided as a bash script. You will need to clone 
the repository. Then you can emit the 
following commands to install the code on the teach cluster after cloning the repository:

```bash
cd 4sp4-lab01
bash build.sh
```

To run the code on the teach cluster:
```bash
sbatch run_teach_cluster.sh gemm
```
For running other experiments use gemv or nn instead of gemm.

**Please do NOT edit SLURM part of the script.**
## Part 1: Matrix-Matrix Multiplication
Please check the MatMul slides for the details of tiling and vectorization in matrix multiplication.

- Explain how you decide about tiling strategy and tile size.
- Explain how you decide about vectorization strategy.
- Please provide necessary profiling and experiment to back up your discussions above.


## Part 2: Matrix-Vector Multiplication
Please check the MatMul slides for the details of tiling and vectorization in matix multiplication.

- Explain how you decide about tiling strategy and tile size.
- Explain how you decide about vectorization strategy.
- Please provide necessary profiling and experiment to back up your discussions above.
- How does matrix-matrix multiplication different from matrix-vector multiplication in terms of tiling and vectorization?



## Part 3: Dense Neural Network
For this part, you will need to copy weight matrices and biases from the provided shared directory in
the teach cluster in the `data` folder:
 ```bash
cp -r /home/l/lcl_uotce4sp4/ce4sp4starter/data .
```
This folder contains trained weight matrices and the MNIST dataset. Functions for reading these files are provided.
The MNIST dataset includes features and labels. The labels are located in the first column of the dataset after loading.

** NOTE: the code assumes the data is in the `data` folder. 
If you are setting up in a different place than the server, make sure to update the paths. **

- Implement a dense neural network using your optimized matrix-matrix and matrix-vector multiplication. For matrix-vector, only test it for the first 10 features of the MNIST dataset.
- Provide a breakdown of execution times of different calls in dense NN. How does breakdown change with your optimized code?
- How much does your optimization improve the performance NN end to end? 


### DNN Description:
A DNN with two linear layers and two non-linear activation functions.

#### Mathematical Representation:

Let's denote:

`X:` Input (One vector (i.e., one row of a matrix) with n elements.)

`W1:` Weights matrix for the first layer (h x n)

`b1:` Biases for the first layer (One vector (i.e., one row of a matrix) with h elements)

`W2:` Weights matrix for the second layer (m x h)

`b2:` Biases for the second layer (One vector (i.e., one row of a matrix) with m elements)

`H:` Hidden layer activations (One vector (i.e., one row of a matrix) with h elements )

`Z:` Output before softmax (One vector (i.e., one row of a matrix) with m elements)

`Y:` Output (One vector (i.e., one row of a matrix) with m elements)

The forward pass of a DNN with two layers and softmax can be represented as follows:

`H = tanh(X * W1^T + b1)`

`Z = sigmoid(H * W2^T + b2)`

`Y = argmax(Z)`

Where:


tanh is an activation function, defined as:
`softmax(Z) = (exp(Z) - exp(-Z)) / (exp(Z) + exp(-Z))`.
sigmoid is another commonly used activation function in neural networks, defined as `1/(1 + exp(-z))`. argmax returns the index of the maximum element in the vector.
The operator `*` refers to a dot product. 

Explanation:

*Input Layer:* The input data is fed into the input layer.

*Hidden Layer:* The input is multiplied by the weights of the first layer, and the biases are added. The result is passed through the tanh activation function.

*Output Layer:* The activations from the hidden layer are multiplied by the weights of the second layer, and the biases are added. The result is passed through the sigmoid activation function to normalize the outputs into a probability distribution.

*Prediction:* Eventually argmax is used to predict the class label based on the output vector Z.

* The matrix notations are provided as input and outputs for one sample/input. 
However, in implementation, a group of samples are stored as a matrix with batchsize rows. 

* For the GEMV case, you should perform one input at a time.

* The target accuracy is expected to be 100% for all the samples in the MNIST dataset.

