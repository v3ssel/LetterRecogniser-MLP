#ifndef _MATRIX_LAYER_H_
#define _MATRIX_LAYER_H_

#include <vector>

#include "libmatrix/matrix.h"

namespace s21 {
class MatrixLayer {
 public:
  MatrixLayer(size_t s);
  MatrixLayer(Matrix w, Matrix b, size_t s);

  Matrix weights;
  Matrix bias;
  Matrix values;

  size_t size;
};
}  // namespace s21

#endif  // _MATRIX_LAYER_H_