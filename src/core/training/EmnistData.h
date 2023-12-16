#ifndef _EMNISTDATA_H_
#define _EMNISTDATA_H_

#include <vector>

namespace s21 {
    struct EMNISTData {
        EMNISTData();
        EMNISTData(std::vector<double> i, std::size_t r);

        std::vector<double> image;
        std::size_t result;
    };
}

#endif // _EMNISTDATA_H_
