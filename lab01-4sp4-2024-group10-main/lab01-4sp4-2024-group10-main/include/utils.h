// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#ifndef LAB1_DENSE_MATMUL_UTILS_H
#define LAB1_DENSE_MATMUL_UTILS_H

#include <string>
#include "def.h"

namespace swiftware::hpp {

    //TODO add necessary includes

    // Do not change the following function signatures
    /// \brief Read a CSV file and store it in a DenseMatrix
    /// \param filename Path to the CSV file
    /// \param OutMat Pointer to the DenseMatrix to store the data
    /// \param removeFirstRow Whether to remove the first row of the CSV file
    DenseMatrix * readCSV(const std::string &filename, bool removeFirstRow = false);
    DenseMatrix* transpose(DenseMatrix* matrix, int rows, int cols);
}

#endif //LAB1_DENSE_MATMUL_UTILS_H
