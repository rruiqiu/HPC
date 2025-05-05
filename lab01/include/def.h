// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#ifndef LAB1_DENSE_MATMUL_DEF_H
#define LAB1_DENSE_MATMUL_DEF_H

#include <vector>
#include <ctime>
#include <cstdlib>

namespace swiftware::hpp {

 struct ScheduleParams {
    int TileSize1;
    int TileSize2;
    ScheduleParams(int TileSize1, int TileSize2): TileSize1(TileSize1), TileSize2(TileSize2){}
 };

 // please do not change the following struct
 struct DenseMatrix{
    int m; //row
    int n; //col
    std::vector<float> data;
    DenseMatrix(int m, int n): m(m), n(n), data(m*n){} //constructro

    void set(int row, int col, float val) {
        if (row <=m && col <=n){
            data[row * n + col] = val;
        }
    }

    float get(int row, int col) const {
        return data[row * n + col];
    }

    int numRows() const {
        return m;
    }

    int numCols() const {
        return n;
    }
    
    float* getdata() {
        return data.data();
    }

    //populates matrix with random number within desired bounds
    void fillMatrix(float min, float max) {
        //seed
        srand (static_cast <unsigned> (time(0)));
        for (int i = 0; i < m * n; ++i) {
                // generate a random number between min and max
                data[i] = (float) min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX) * (max-min));
        }
        
    }


 };

}

#endif //LAB1_DENSE_MATMUL_DEF_H
