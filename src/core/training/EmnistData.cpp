#include "EmnistData.h"

namespace s21 {
    EMNISTData::EMNISTData() { result = -1; }

    EMNISTData::EMNISTData(std::vector<double> i, std::size_t r) 
        : image(i), result(r) {}
}