#ifndef _GRAPHNODE_H_
#define _GRAPHNODE_H_

#include <vector>

namespace s21 {
struct GraphNode {
    double value = 0.0l;
    double bias = 0.0l;
    std::vector<double> weights;
};
}  // namespace s21

#endif  // _GRAPHNODE_H_
