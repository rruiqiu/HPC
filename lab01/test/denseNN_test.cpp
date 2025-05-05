// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab
#include <gtest/gtest.h>
#include "dense_nn.h"
#include "utils.h"
#include <iostream>
namespace swiftware::hpp {
 // TODO
  TEST(dense_nn_gemm, TestAll){
    std::string dataFolder = "../data/data"; //change this to accommodate folder struct
    auto *mnistData = swiftware::hpp::readCSV(dataFolder + "/mnist_train.csv", true);
    std::cout << "Reading the mnist data done " << std::endl;
    auto *labels = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
    auto *features = new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);
    //TODO: Extract labels and features from mnist dataset
    //m= numbers or rows, n = number of cols
    for (int i = 0; i < mnistData->m; ++i) {
      labels->data[i] = mnistData->data[i * mnistData->n];  // Copy label from first column
      //the rest of the cols contain the features
      for (int j = 1; j < mnistData->n; ++j) {
          features->data[i * features->n + (j - 1)] = mnistData->data[i * mnistData->n + j];
      }
    }

    auto *weightsOutput = swiftware::hpp::readCSV(dataFolder + "/model/weights_output.csv");
    auto *weightsHidden = swiftware::hpp::readCSV(dataFolder + "/model/weights_hidden.csv");
    auto *biasesHidden = swiftware::hpp::readCSV(dataFolder + "/model/biases_hidden.csv");
    auto *biasesOutput = swiftware::hpp::readCSV(dataFolder + "/model/biases_output.csv");
    std::cout << "Reading the csv done " << std::endl;
    swiftware::hpp::ScheduleParams scheduleParams(1024, -1);
    // Running the NN function
    auto *result = swiftware::hpp::dense_nn_gemm(features, weightsHidden, weightsOutput, biasesHidden, biasesOutput, scheduleParams);
    int correctPredictions = 0;
    int totalSamples = labels->m;
    //int totalSamples = 10; //this is to test the first 10 features of the gemV code only
    for (int i = 0; i < totalSamples; ++i) {
        if (result->data[i] == labels->data[i]) {  // Assuming both contain integer labels
            correctPredictions++;
        }
    }

    // Calculate the accuracy as a percentage
    double accuracy = (static_cast<double>(correctPredictions) / totalSamples) * 100.0;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

  }

  TEST(dense_nn_gemv, TestAll){
    std::string dataFolder = "../data/data"; //change this to accommodate folder struct
    auto *mnistData = swiftware::hpp::readCSV(dataFolder + "/mnist_train.csv", true);
    std::cout << "Reading the mnist data done " << std::endl;
    auto *labels = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
    auto *features = new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);
    //TODO: Extract labels and features from mnist dataset
    //m= numbers or rows, n = number of cols
    for (int i = 0; i < mnistData->m; ++i) {
      labels->data[i] = mnistData->data[i * mnistData->n];  // Copy label from first column
      //the rest of the cols contain the features
      for (int j = 1; j < mnistData->n; ++j) {
          features->data[i * features->n + (j - 1)] = mnistData->data[i * mnistData->n + j];
      }
    }

    auto *weightsOutput = swiftware::hpp::readCSV(dataFolder + "/model/weights_output.csv");
    auto *weightsHidden = swiftware::hpp::readCSV(dataFolder + "/model/weights_hidden.csv");
    auto *biasesHidden = swiftware::hpp::readCSV(dataFolder + "/model/biases_hidden.csv");
    auto *biasesOutput = swiftware::hpp::readCSV(dataFolder + "/model/biases_output.csv");
    std::cout << "Reading the csv done " << std::endl;
    swiftware::hpp::ScheduleParams scheduleParams(1024, -1);
    // Running the NN function
    auto *result = swiftware::hpp::dense_nn_gemv(features, weightsHidden, weightsOutput, biasesHidden, biasesOutput, scheduleParams);
    int correctPredictions = 0;
    // int totalSamples = labels->m;
    int totalSamples = 10; //this is to test the first 10 features of the gemV code only
    for (int i = 0; i < totalSamples; ++i) {
        if (result->data[i] == labels->data[i]) {  // Assuming both contain integer labels
            correctPredictions++;
        }
    }

    // Calculate the accuracy as a percentage
    double accuracy = (static_cast<double>(correctPredictions) / totalSamples) * 100.0;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

  }
}