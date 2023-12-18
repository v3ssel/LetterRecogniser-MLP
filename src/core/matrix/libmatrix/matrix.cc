#include "matrix.h"

namespace s21 {
  Matrix::Matrix() : Matrix(0, 0) {}

  Matrix::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
    memoryAlloc();
  }

  
  Matrix::Matrix(const std::vector<double>& vec) : rows_(1), cols_(vec.size()) {
    memoryAlloc();
    std::copy(vec.begin(), vec.end(), matrix_[0]);
  }

  Matrix::Matrix(const Matrix& other)
      : rows_(other.rows_), cols_(other.cols_) {
    memoryAlloc();
    for (size_t i = 0; i < rows_; i++) {
      std::copy(other.matrix_[i], other.matrix_[i] + cols_, matrix_[i]);
    }
  }

  Matrix::Matrix(Matrix&& other)
      : rows_(other.rows_), cols_(other.cols_), matrix_(other.matrix_) {
    other.matrix_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
  }

  Matrix::~Matrix() {
    if (matrix_ != nullptr) {
      for (size_t i = 0; i < rows_; i++) {
        if (matrix_[i] != nullptr) delete[] matrix_[i];
      }
      delete[] matrix_;
      matrix_ = nullptr;
    }
  }

  bool Matrix::EqMatrix(const Matrix& other) const {
    bool ret = true;
    if (rows_ != other.rows_) ret = false;
    if (cols_ != other.cols_) ret = false;
    if (ret) {
      for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
          if (fabs(matrix_[i][j] - other.matrix_[i][j]) > 1e-7) {
            ret = false;
          }
        }
      }
    }
    return ret;
  }

  void Matrix::SumMatrix(const Matrix& other) {
    if (other.cols_ != cols_ || other.rows_ != rows_)
      throw std::out_of_range("SumMatrix: Cols or rows not equal");
    for (size_t i = 0; i < rows_; i++) {
      for (size_t j = 0; j < cols_; j++) {
        matrix_[i][j] = matrix_[i][j] + other.matrix_[i][j];
      }
    }
  }

  void Matrix::SubMatrix(const Matrix& other) {
    if (other.cols_ != cols_ || other.rows_ != rows_)
      throw std::out_of_range("SubMatrix: Cols or rows not equal");
    for (size_t i = 0; i < rows_; i++) {
      for (size_t j = 0; j < cols_; j++) {
        matrix_[i][j] = matrix_[i][j] - other.matrix_[i][j];
      }
    }
  }

  void Matrix::MulNumber(const double num) {
    for (size_t i = 0; i < rows_; i++) {
      for (size_t j = 0; j < cols_; j++) {
        matrix_[i][j] *= num;
      }
    }
  }

  void Matrix::MulMatrix(const Matrix& other) {
    if (this->cols_ != other.rows_)
      throw std::out_of_range("MulMatrix: Object cols not equal to other rows");
    Matrix res(rows_, other.cols_);
    for (size_t i = 0; i < rows_; i++) {
      for (size_t j = 0; j < other.cols_; j++) {
        for (size_t l = 0; l < cols_; l++) {
          res.matrix_[i][j] += matrix_[i][l] * other.matrix_[l][j];
        }
      }
    }
    *this = res;
  }

  Matrix Matrix::Transpose() {
    Matrix res(cols_, rows_);
    for (size_t i = 0; i < rows_; i++) {
      for (size_t j = 0; j < cols_; j++) {
        res.matrix_[j][i] = matrix_[i][j];
      }
    }
    return res;
  }

  Matrix Matrix::CalcComplements() {
    if (rows_ != cols_) throw std::out_of_range("CalcComplements: Matrix not a square");
    size_t n = rows_;
    Matrix tmp(rows_ - 1, cols_ - 1);
    Matrix res(n, n);
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) {
        double det = 0;
        tmp.rows_ = n - 1;
        tmp.cols_ = n - 1;
        for (size_t l = 0, t = 0, r = 0; l < n; l++) {
          if (l == i) {
            r = 1;
          } else {
            t = 0;
          }
          for (size_t x = 0; x < n; x++) {
            if (x == j) t = 1;
            if (l != i && x != j) tmp.matrix_[l - r][x - t] = matrix_[l][x];
          }
          det = tmp.Determinant();
          res.matrix_[i][j] = det * pow(-1, i + j);
        }
      }
    }
    return res;
  }

  double Matrix::Determinant() {
    if (cols_ != rows_) throw std::out_of_range("Determinant: Matrix not a square");
    if (rows_ == 1) return matrix_[0][0];
    return DeterminantPlus(rows_);
  }

  double Matrix::DeterminantPlus(size_t n) {
    double det = 0;
    if (n == 2) {
      det = matrix_[0][0] * matrix_[1][1] - matrix_[1][0] * matrix_[0][1];
    } else {
      Matrix tmp(rows_, cols_);
      for (size_t l = 0; l < n; l++) {
        size_t i2 = 0;
        for (size_t i = 1; i < n; i++) {
          size_t j2 = 0;
          for (size_t j = 0; j < n; j++) {
            if (l != j) {
              tmp.matrix_[i2][j2] = matrix_[i][j];
              j2++;
            }
          }
          i2++;
        }
        det += pow(-1, l) * matrix_[0][l] * tmp.DeterminantPlus(n - 1);
      }
    }
    return det;
  }

  Matrix Matrix::InverseMatrix() {
    double det = Determinant();
    if (det == 0) throw std::out_of_range("InverseMatrix: Determinant is 0");
    Matrix minor = CalcComplements();
    Matrix transpose = minor.Transpose();
    Matrix res = transpose * (1.0 / det);
    return res;
  }

  Matrix Matrix::operator+(const Matrix& rhs) const {
    Matrix res(*this);
    res.SumMatrix(rhs);
    return res;
  }

  Matrix Matrix::operator-(const Matrix& rhs) const {
    Matrix res(*this);
    res.SubMatrix(rhs);
    return res;
  }

  Matrix Matrix::operator*(const Matrix& rhs) const {
    Matrix res(*this);
    res.MulMatrix(rhs);
    return res;
  }

  Matrix Matrix::operator*(const double rhs) const {
    Matrix res(*this);
    res.MulNumber(rhs);
    return res;
  }

  Matrix& Matrix::operator+=(const Matrix& rhs) {
    this->SumMatrix(rhs);
    return *this;
  }

  Matrix& Matrix::operator-=(const Matrix& rhs) {
    this->SubMatrix(rhs);
    return *this;
  }

  Matrix& Matrix::operator*=(const Matrix& rhs) {
    this->MulMatrix(rhs);
    return *this;
  }

  Matrix& Matrix::operator*=(const double rhs) {
    this->MulNumber(rhs);
    return *this;
  }

  bool Matrix::operator==(const Matrix& rhs) const {
    return this->EqMatrix(rhs);
  }

  Matrix Matrix::operator=(const Matrix& rhs) {
    this->~Matrix();
    rows_ = rhs.rows_;
    cols_ = rhs.cols_;
    memoryAlloc();
    for (size_t i = 0; i < rows_; i++) {
      std::copy(rhs.matrix_[i], rhs.matrix_[i] + cols_, matrix_[i]);
    }
    return *this;
  }

  Matrix Matrix::operator=(Matrix &&rhs) {
    this->~Matrix();
    rows_ = rhs.rows_;
    cols_ = rhs.cols_;
    matrix_ = rhs.matrix_;

    rhs.rows_ = 0;
    rhs.cols_ = 0;
    rhs.matrix_ = nullptr;
    
    return *this;
  }

  double& Matrix::operator()(const size_t row, const size_t col) {
    return matrix_[row][col];
  }

  double Matrix::operator()(const size_t row, const size_t col) const {
      return matrix_[row][col];
  }

  size_t Matrix::getRows() const { return rows_; }

  size_t Matrix::getCols() const { return cols_; }

  void Matrix::setRows(size_t rows) {
    if (rows < 1) {
      throw std::out_of_range("setRows: rows cannot be less 1");
    }
    Matrix upd(rows, cols_);
    size_t r = (upd.rows_ < rows_) ? upd.rows_ : rows_;
    for (size_t i = 0; i < r; i++) {
      std::copy(matrix_[i], matrix_[i] + upd.cols_, upd.matrix_[i]);
    }
    *this = upd;
  }

  void Matrix::setCols(size_t cols) {
    if (cols < 1) throw std::out_of_range("setCols: Cols cannot be less than 1");
    Matrix upd(rows_, cols);
    for (size_t i = 0; i < rows_; i++) {
      std::copy(matrix_[i], matrix_[i] + upd.cols_, upd.matrix_[i]);
    }
    *this = upd;
  }

  std::vector<double> Matrix::ToVector() {
    std::vector<double> res;
    res.reserve(cols_);
    std::copy(matrix_[0], matrix_[0] + cols_, std::back_inserter(res));

    return res;
  }

  void Matrix::memoryAlloc() {
    matrix_ = new double*[rows_];
    for (size_t i = 0; i < rows_; i++) {
      matrix_[i] = new double[cols_]();
      std::fill(matrix_[i], matrix_[i] + cols_, 0);
    }
  }

  void Matrix::Print() {
    for(size_t i = 0; i < rows_; i++) {
      for(size_t j = 0; j < cols_; j++) {
        std::cout << matrix_[i][j] << " ";
      }
      std::cout << "\n";
    }
  }

  Matrix Matrix::GenerateRandom(size_t rows, size_t cols)
  {
      std::mt19937 gen(std::random_device{}());
      std::normal_distribution<long double> dist(0.0l, 1.0l);

      Matrix m(rows, cols);
      for (size_t i = 0; i < rows; i++)
      {
          for (size_t j = 0; j < cols; j++)
          {
              m(i, j) = dist(gen);
          }
      }

      return m;
  }
} // namespace s21
