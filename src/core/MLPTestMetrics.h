#ifndef _MLPTESTMETRICS_H_
#define _MLPTESTMETRICS_H_

#include <chrono>

namespace s21 {
struct MLPTestMetrics {
    double accurancy_percent = 0.0l;
    double accurancy = 0.0l;
    double precision = 0.0l;
    double recall = 0.0l;
    double f_measure = 0.0l;
    std::chrono::milliseconds testing_time;
};
}  // namespace s21

#endif  //_MLPTESTMETRICS_H_
