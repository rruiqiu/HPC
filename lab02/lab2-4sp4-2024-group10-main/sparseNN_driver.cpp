// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#include <benchmark/benchmark.h>
#include <iostream>
#include "sparse_nn.h"
#include "utils.h"

// TODO : Implement the benchmark

static void BM_SPARSENN(benchmark::State &state,
                        swiftware::hpp::DenseMatrix* (*sparseNNImpl)(swiftware::hpp::DenseMatrix *InData,
                                                                     swiftware::hpp::CSR *W1,
                                                                     swiftware::hpp::CSR *W2,
                                                                     swiftware::hpp::DenseMatrix *B1,
                                                                     swiftware::hpp::DenseMatrix *B2,
                                                                     swiftware::hpp::ScheduleParams Sp)) {
  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
  auto *labels = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
  auto *features = new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);
  //TODO: Extract labels and features from mnist dataset

  auto *weightsOutput = swiftware::hpp::readCSV("./data/model/weights_output.csv");
  auto *weightsHidden = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
  auto *biasesHidden = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
  auto *biasesOutput = swiftware::hpp::readCSV("./data/model/biases_output.csv");

  auto *weightsOutputCSR = swiftware::hpp::dense2CSR(weightsOutput);
  auto *weightsHiddenCSR = swiftware::hpp::dense2CSR(weightsHidden);

  swiftware::hpp::ScheduleParams scheduleParams(32, 32);
  //TODO : Implement the benchmark

  for (auto _: state) {
    // Running the NN function
    auto *result = sparseNNImpl(features, weightsHiddenCSR, weightsOutputCSR, biasesHidden, biasesOutput, scheduleParams);
    // TODO: Calculate accuracy

  }

}

static void BM_DENSENN(benchmark::State &state,
                       swiftware::hpp::DenseMatrix* (*denseNNImpl)(swiftware::hpp::DenseMatrix *InData,
                                                                   swiftware::hpp::DenseMatrix *W1,
                                                                   swiftware::hpp::DenseMatrix *W2,
                                                                   swiftware::hpp::DenseMatrix *B1,
                                                                   swiftware::hpp::DenseMatrix *B2,
                                                                   swiftware::hpp::ScheduleParams Sp)) {
  auto *mnistData = swiftware::hpp::readCSV("../data/mnist_train.csv", true);
  auto *labels = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
  auto *features = new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);
  //TODO: Extract labels and features from mnist dataset

  auto *weightsOutput = swiftware::hpp::readCSV("./data/model/weights_output.csv");
  auto *weightsHidden = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
  auto *biasesHidden = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
  auto *biasesOutput = swiftware::hpp::readCSV("./data/model/biases_output.csv");

  swiftware::hpp::ScheduleParams scheduleParams(32, 32);
  //TODO : Implement the benchmark

  for (auto _: state) {
    // Running the NN function
    auto *result = denseNNImpl(features, weightsHidden, weightsOutput, biasesHidden, biasesOutput, scheduleParams);
    // TODO: Calculate accuracy
  }

}


BENCHMARK_CAPTURE(BM_SPARSENN, sparse_nn_spmm, swiftware::hpp::sparseNNSpmm);

BENCHMARK_CAPTURE(BM_DENSENN, dense_nn_spmm, swiftware::hpp::denseNNGemm);

BENCHMARK_MAIN();