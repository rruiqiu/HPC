// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include "dense_nn.h"
#include <cmath>
#include <algorithm>
#include "def.h"
#include "utils.h"
#include "gemm.h"
#include "gemv.h"
#include <iostream>
#include <cstdio>

namespace swiftware::hpp {

    DenseMatrix *dense_nn_gemm(DenseMatrix *InData,
                               DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
        int batchSize = InData->m;
        int featDim = InData->n;
        int hiddenDim = W1->m;
        int outDim = W2->m;


        DenseMatrix *out1 = new DenseMatrix(batchSize, hiddenDim);
        DenseMatrix *out2 = new DenseMatrix(batchSize, outDim);
        DenseMatrix *pred = new DenseMatrix(batchSize, 1);
        auto predData = pred->data;
        std::cout << "Processing input layer to hidden" << std::endl;
        auto *W1_t = swiftware::hpp::transpose(W1,hiddenDim,featDim);
        gemmVectorized(batchSize,hiddenDim,featDim,InData->data.data(), W1_t->data.data(),out1->data.data(),Sp);
        
        for (int i= 0; i<batchSize; i++){
            for (int j = 0; j < hiddenDim; j++){
                int indx = i * hiddenDim + j;
                //add bias vector to row of xW1^T
                out1->data[indx]+=B1->data[j];
                out1->data[indx] =  std::tanh(out1->data[indx]);
            }
        }

        std::cout << "Processing Hidden Layer to Output" << std::endl;
        auto *W2_t= swiftware::hpp::transpose(W2,outDim,hiddenDim);
        gemmVectorized(batchSize,outDim,hiddenDim,out1->data.data(), W2_t->data.data(),out2->data.data(),Sp);
        for (int i= 0; i<batchSize; i++){
            for (int j = 0; j < outDim; j++){
                int indx = i * outDim + j;
                out2->data[indx]+=B2->data[j];
                float neg = -(out2->data[indx]);
                float nexpz = std::exp(neg);
                out2->data[indx] =  1.0/(1.0 + nexpz);
            }
        }
        std::cout << "Finding argmax" << std::endl;
        
        auto outDataVec = out2->data;
        
        // argmax
        for (int i = 0; i < batchSize; i++) {
            //stores the index of the max element in each row
            predData[i] = static_cast<int >(std::distance(outDataVec.begin() + i * outDim,
                                                          std::max_element(outDataVec.begin() + i * outDim,
                                                                           outDataVec.begin() + (i + 1) * outDim)));                                                       
        }
        pred->data = predData;

        return pred;
    }

    DenseMatrix *dense_nn_gemv(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp){
        int batchSize = InData->m; 
        int featDim = InData->n;
        int hiddenDim = W1->m;
        int outDim = W2->m;
        DenseMatrix *pred = new DenseMatrix(10, 1);
        auto predData = pred->data;
        for (int i = 0; i < 10; i++){
            DenseMatrix *out1 = new DenseMatrix(hiddenDim, 1);
            DenseMatrix *out2 = new DenseMatrix(outDim, 1);
            DenseMatrix *Input = new DenseMatrix(featDim,1);
            for (int ii = 0; ii < featDim; ii++) {
                //set the input data to the ith row of InData
                Input->data[ii] = InData->data[i * featDim + ii];  
            }
            gemvVec(hiddenDim,featDim, W1->data.data(), Input->data.data(), out1->data.data(),Sp);
            
            for (int j= 0; j<hiddenDim; j++){
                //add bias vector to row of xW1
                out1->data[j]+=B1->data[j];
                out1->data[j] = std::tanh(out1->data[j]);
            
            }

            gemvVec(outDim,hiddenDim,W2->data.data(),out1->data.data(),out2->data.data(),Sp);

            for (int k= 0; k<outDim; k++){
                out2->data[k]+=B2->data[k];
                float neg = -(out2->data[k]);
                float nexpz = std::exp(neg);
                out2->data[k] =  1.0/(1.0 + nexpz);
            }
            auto result_data = out2->data;
            auto max_element_it = std::max_element(result_data.begin(), result_data.end());
            int max_index = std::distance(result_data.begin(), max_element_it);
            predData[i] = max_index;

            delete out1;
            delete out2;
            delete Input;
        }

        pred->data = predData;
        return pred;

    }
   
}
