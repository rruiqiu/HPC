// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#ifndef LAB3_INTEGRATION_H
#define LAB3_INTEGRATION_H

#include "def.h"

namespace swiftware::hpp {


// please do not change below
struct IntegrationParams {
  int NumSteps; //number of steps in the integration (n)
  double Sum; //sum of the integration values
  double StepSize; //h: the interval width
  double PI; //calcultd value of pi
  IntegrationParams(int numSteps) : NumSteps(numSteps), Sum(0.0), StepSize(1.0 / numSteps), PI(0.0) {}
};

void integrateSequential(IntegrationParams *params, ScheduleParams sp);
void integrateParallel(IntegrationParams *params, ScheduleParams sp);
void integrateParallelWithAllOMP(IntegrationParams *params, ScheduleParams sp);

} // namespace swiftware::hpp

#endif // LAB3_INTEGRATION_H
