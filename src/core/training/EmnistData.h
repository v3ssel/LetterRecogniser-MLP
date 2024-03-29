#ifndef _EMNISTDATA_H_
#define _EMNISTDATA_H_

#include <vector>

namespace s21 {
struct EMNISTData {
    EMNISTData();

    std::vector<double> image;
    std::size_t result;
};
}  // namespace s21

#endif  // _EMNISTDATA_H_
