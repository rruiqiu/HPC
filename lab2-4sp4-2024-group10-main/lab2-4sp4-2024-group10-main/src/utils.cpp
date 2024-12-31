// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include "utils.h"
#include <fstream>
#include <sstream>


namespace swiftware::hpp {


  CSR *dense2CSR(const DenseMatrix *A) {
      int m = A->m;
      int n  = A->n;

      auto *OutMat = new CSR(A->m, A->n);
      // TODO implement dense2CSR
      OutMat->rowPtr.resize(m + 1); //fix segfault error ( need to allocate space)
      OutMat->rowPtr[0] = 0;
      for(int i=0; i<m; i++){
        for (int j=0; j<n;j++){
          if (A->data[i * n + j] !=0.0){
            OutMat->data.push_back(A->data[i * A->n + j]);
            OutMat->colIdx.push_back(j);

          }
          
        }
        OutMat->rowPtr[i + 1] = OutMat->data.size();
      }
      return OutMat;
  }


  // Do not change the following function signatures
  DenseMatrix *readCSV(const std::string &filename, bool removeFirstRow) {
    std::ifstream file(filename);
    std::string line, word;
    // determine number of columns in file
    std::vector<std::string> lines;
    int cntr = 0;
    while (getline(file, line)) {
      lines.push_back(line);
    }
    if (removeFirstRow) {
      lines.erase(lines.begin());
    }
    std::vector<std::vector<float>> valuesPerLine(lines.size());
    for (int i = 0; i < lines.size(); i++) {
      std::stringstream lineStream(lines[i]);
      while (getline(lineStream, word, ',')) {
        valuesPerLine[i].push_back(std::stof(word));
      }
    }
    auto *OutMat = new DenseMatrix(valuesPerLine.size(), valuesPerLine[0].size());
    auto *data = OutMat->data.data();
    int ncol = OutMat->n;
    for (int i = 0; i < valuesPerLine.size(); i++) {
      size_t cols = valuesPerLine[i].size();
      for (uint j = 0; j < cols; j++) {
        data[i * ncol + j] = valuesPerLine[i][j];
      }
    }
    return OutMat;
  }

  DenseMatrix *samplingDense(const DenseMatrix *A, float samplingRate) {
    //transforms the dense matrix into a sparse matrix based on stride length. It 
    //will go through every element of matrix A, and based on the stude
    //it will either keep the oriingal value or replace it with 0
    //so if the smapling rate is 80, tha tmeans i want to keep 80 percent of the original elments
    auto *OutMat = new DenseMatrix(A->m, A->n);
    int stride = 1. / samplingRate;
    for (int i = 0; i < A->m; i++) {
      for (int j = 0; j < A->n; j++) {
        int idx = i * A->n + j;
        if (idx % stride)
          OutMat->data[idx] = 0;
        else
          OutMat->data[idx] = A->data[idx];
      }
    }
    return OutMat;
  }


}