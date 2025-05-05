// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include "sparse_nn.h"
#include <cmath>
#include <algorithm>

namespace swiftware::hpp {


  DenseMatrix *sparseNNSpmm(DenseMatrix *InData,
                            CSR *W1, CSR *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
    int batchSize = InData->m;
    int featDim = InData->n;
    int hiddenDim = W1->m;
    int outDim = W2->m;


    DenseMatrix *out1 = new DenseMatrix(batchSize, hiddenDim);
    DenseMatrix *out2 = new DenseMatrix(batchSize, outDim);
    DenseMatrix *pred = new DenseMatrix(batchSize, 1);
    auto predData = pred->data;
    auto outDataVec = out2->data;

    // TODO: Layer 1 Calculations: H = tanh(F * W1^T + B1)
    // TODO: Layer 2 Calculations: O = sigmoid(H * W2^T + B2)

    // argmax
    for (int i = 0; i < batchSize; i++) {
      predData[i] = static_cast<int >(std::distance(outDataVec.begin() + i * outDim,
                                                    std::max_element(outDataVec.begin() + i * outDim,
                                                                     outDataVec.begin() + (i + 1) * outDim)));
    }

    return pred;
  }

  //TODO: Copy your dense NN here
  DenseMatrix *denseNNGemm(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp){
    return nullptr;
  }

  //TODO: dense NN using GeMM skipping
  DenseMatrix *denseNNGemmSkipping(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp){
    return nullptr;
  }

}