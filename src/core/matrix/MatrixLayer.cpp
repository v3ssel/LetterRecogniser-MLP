#include "MatrixLayer.h"

namespace s21 {
    MatrixLayer::MatrixLayer(size_t s) 
        : size(s) {}

    MatrixLayer::MatrixLayer(Matrix w, Matrix b, size_t s) 
        : weights(w), bias(b), size(s) {}
}