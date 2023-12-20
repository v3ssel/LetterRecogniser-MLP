#ifndef _MLPTESTMETRICS_H_
#define _MLPTESTMETRICS_H_

#include <chrono>

struct MLPTestMetrics {
    double accurancy_percent;
    double accurancy;
    double precision;
    double recall;
    double f_measure;
    std::chrono::milliseconds testing_time;
};

#endif //_MLPTESTMETRICS_H_
