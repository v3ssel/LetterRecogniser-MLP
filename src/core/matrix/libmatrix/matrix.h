#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <array>
#include <cmath>
#include <random>
#include <iostream>
#include <functional>

namespace s21 {
  class Matrix {
   public:
    Matrix();
    Matrix(size_t, size_t);
    Matrix(const std::vector<double>& vec);
    Matrix(const Matrix &other);
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

    Matrix operator+(const Matrix& rhs) const;
    Matrix operator-(const Matrix& rhs) const;
    Matrix operator*(const Matrix& rhs) const;
    Matrix operator*(const double rhs) const;

    Matrix& operator+=(const Matrix& rhs);
    Matrix& operator-=(const Matrix& rhs);
    Matrix& operator*=(const Matrix& rhs);
    Matrix& operator*=(const double rhs);

    bool operator==(const Matrix& rhs) const;
    Matrix operator=(const Matrix& rhs);
    Matrix operator=(Matrix&& rhs);

    double& operator()(const size_t row, const size_t col);
    double operator()(const size_t row, const size_t col) const;

    size_t getRows() const;
    size_t getCols() const;

    void setRows(size_t);
    void setCols(size_t);

    std::vector<double> ToVector();

    void Print();
    static Matrix GenerateRandom(size_t rows, size_t cols);

   private:
    void memoryAlloc();
    double DeterminantPlus(size_t);
    size_t rows_, cols_;
    double** matrix_;
  };
} // namespace s21

#endif  //  S21_MATRIX_H
