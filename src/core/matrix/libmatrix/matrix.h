#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <array>
#include <cmath>
#include <iostream>

namespace s21 {
  class Matrix {
   public:
    Matrix();
    Matrix(size_t, size_t);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other);
    ~Matrix();

    bool EqMatrix(const Matrix& other) const;

    void SumMatrix(const Matrix& other);
    void SubMatrix(const Matrix& other);
    void MulNumber(const double num);
    void MulMatrix(const Matrix& other);

    Matrix Transpose();
    Matrix CalcComplements();
    double Determinant();
    Matrix InverseMatrix();

    Matrix operator+(const Matrix& rhs);
    Matrix operator-(const Matrix& rhs);
    Matrix operator*(const Matrix& rhs);
    Matrix operator*(const double rhs);

    Matrix& operator+=(const Matrix& rhs);
    Matrix& operator-=(const Matrix& rhs);
    Matrix& operator*=(const Matrix& rhs);
    Matrix& operator*=(const double rhs);

    bool operator==(const Matrix& rhs) const;
    Matrix operator=(const Matrix& rhs);
    double& operator()(const size_t row, const size_t col);

    size_t getRows();
    size_t getCols();

    void setRows(size_t);
    void setCols(size_t);

    void fillMatrix(double, double);

   private:
    void memoryAlloc();
    double DeterminantPlus(size_t);
    size_t rows_, cols_;
    double** matrix_;
  };
} // namespace s21

#endif  //  S21_MATRIX_H
