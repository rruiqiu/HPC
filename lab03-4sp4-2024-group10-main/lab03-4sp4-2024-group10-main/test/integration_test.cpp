// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include <gtest/gtest.h>
#include "integration.h"

namespace swiftware::hpp {
  TEST(IntegrationTest, SmallTest) {
    int numSteps = 20;
    auto chunkSize = 2;
    auto numThreads = 8;

    auto *params_seq = new swiftware::hpp::IntegrationParams(numSteps);
    auto *params_par = new swiftware::hpp::IntegrationParams(numSteps);
    swiftware::hpp::integrateSequential(params_seq, swiftware::hpp::ScheduleParams(1, 1, numThreads, chunkSize)); 
    //swiftware::hpp::integrateParallelWithAllOMP(params_par, swiftware::hpp::ScheduleParams(1, 1, numThreads, chunkSize)); 
    swiftware::hpp::integrateParallel(params_par, swiftware::hpp::ScheduleParams(1, 1, numThreads, chunkSize)); 
    EXPECT_FLOAT_EQ(params_seq->PI, params_par->PI);
    
  }
}