// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#include "integration.h"
#include <omp.h>
#include <iostream>
namespace swiftware::hpp {

  void integrateSequential(IntegrationParams *params, ScheduleParams sp) {
    //TODO implement the integration here
    for (int i = 0; i<params->NumSteps; i++){
      double x = (i + 0.5) * params->StepSize;
      params->Sum+=1.0 / (1.0 + x * x);
    }
    params->PI = 4.0 * params->StepSize * params->Sum;

  }

  void integrateParallel(IntegrationParams *params, ScheduleParams sp) {
    //TODO implement the integration here
    std::vector<double> thread_sum_vec(sp.NumThreads, 0.0);  // Array to store each thread's partial sum

    #pragma omp parallel num_threads(sp.NumThreads) 
    {
        int thread_id = omp_get_thread_num();
        double thread_sum = 0.0;

        // each integration is a diff thread ( this is done by incrementing by num threads)
        for (int i = thread_id; i < params->NumSteps; i += sp.NumThreads) {
            double x = (i + 0.5) * params->StepSize;
            thread_sum += 1.0 / (1.0 + x * x);
            #pragma omp critical
            {
                std::cout << "Thread ID: " << thread_id << std::endl;
            }
        }
        thread_sum_vec[thread_id] = thread_sum;
    }
    
    for (int i = 0; i < sp.NumThreads; i++) {
        params->Sum += thread_sum_vec[i];
    }

    params->PI = 4.0 * params->StepSize * params->Sum;

  }

  void integrateParallelWithAllOMP(IntegrationParams *params, ScheduleParams sp) {
    //TODO implement the integration here
    double sum = 0.0;
    #pragma omp parallel reduction(+: sum) num_threads(sp.NumThreads)
    {
      #pragma omp for schedule(static) 
      for (int i = 0; i<params->NumSteps; i++){
        float x = (i + 0.5) * params->StepSize;
        sum+=1.0 / (1.0 + x * x);
      }
    }
    params->PI = 4.0 * params->StepSize * sum;
  }
}