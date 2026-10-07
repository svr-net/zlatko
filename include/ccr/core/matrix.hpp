#pragma once

#include <cstddef>
#include <vector>

namespace ccr {

/// Dense row-major matrix.
///
/// Throughout the library simulated quantities are stored as matrices with one
/// row per Monte Carlo path and one column per simulation date, i.e. value(p, j)
/// is the quantity on path p at grid time t_j.
class Matrix {
 public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols, double value = 0.0)
      : rows_(rows), cols_(cols), data_(rows * cols, value) {}

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
  bool empty() const { return data_.empty(); }

  double& operator()(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
  double operator()(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

  double* row(std::size_t r) { return data_.data() + r * cols_; }
  const double* row(std::size_t r) const { return data_.data() + r * cols_; }

  /// Copy of column c (for a path matrix: all paths at one simulation date).
  std::vector<double> column(std::size_t c) const;
  void setColumn(std::size_t c, const std::vector<double>& values);

  Matrix& operator+=(const Matrix& other);
  Matrix& operator-=(const Matrix& other);
  Matrix& operator*=(double scalar);

  static Matrix identity(std::size_t n);

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<double> data_;
};

Matrix operator+(Matrix lhs, const Matrix& rhs);
Matrix operator-(Matrix lhs, const Matrix& rhs);
Matrix operator*(Matrix lhs, double scalar);

/// Lower-triangular Cholesky factor L with A = L L^T.
/// Throws std::invalid_argument if A is not symmetric positive definite.
Matrix cholesky(const Matrix& a);

/// Solves A x = b by Gaussian elimination with partial pivoting.
std::vector<double> solveLinearSystem(Matrix a, std::vector<double> b);

/// Returns A^T.
Matrix transpose(const Matrix& a);

}  // namespace ccr
