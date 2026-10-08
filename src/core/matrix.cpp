#include "ccr/core/matrix.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ccr {

std::vector<double> Matrix::column(std::size_t c) const {
  std::vector<double> out(rows_);
  for (std::size_t r = 0; r < rows_; ++r) out[r] = (*this)(r, c);
  return out;
}

void Matrix::setColumn(std::size_t c, const std::vector<double>& values) {
  if (values.size() != rows_) throw std::invalid_argument("Matrix::setColumn: size mismatch");
  for (std::size_t r = 0; r < rows_; ++r) (*this)(r, c) = values[r];
}

Matrix& Matrix::operator+=(const Matrix& other) {
  if (rows_ != other.rows_ || cols_ != other.cols_)
    throw std::invalid_argument("Matrix::operator+=: dimension mismatch");
  for (std::size_t i = 0; i < data_.size(); ++i) data_[i] += other.data_[i];
  return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
  if (rows_ != other.rows_ || cols_ != other.cols_)
    throw std::invalid_argument("Matrix::operator-=: dimension mismatch");
  for (std::size_t i = 0; i < data_.size(); ++i) data_[i] -= other.data_[i];
  return *this;
}

Matrix& Matrix::operator*=(double scalar) {
  for (double& v : data_) v *= scalar;
  return *this;
}

Matrix Matrix::identity(std::size_t n) {
  Matrix m(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.0;
  return m;
}

Matrix operator+(Matrix lhs, const Matrix& rhs) { return lhs += rhs; }
Matrix operator-(Matrix lhs, const Matrix& rhs) { return lhs -= rhs; }
Matrix operator*(Matrix lhs, double scalar) { return lhs *= scalar; }

Matrix cholesky(const Matrix& a) {
  const std::size_t n = a.rows();
  if (a.cols() != n) throw std::invalid_argument("cholesky: matrix must be square");
  Matrix l(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      if (std::fabs(a(i, j) - a(j, i)) > 1e-10)
        throw std::invalid_argument("cholesky: matrix must be symmetric");
      double sum = a(i, j);
      for (std::size_t k = 0; k < j; ++k) sum -= l(i, k) * l(j, k);
      if (i == j) {
        if (sum <= 0.0) throw std::invalid_argument("cholesky: matrix is not positive definite");
        l(i, i) = std::sqrt(sum);
      } else {
        l(i, j) = sum / l(j, j);
      }
    }
  }
  return l;
}

std::vector<double> solveLinearSystem(Matrix a, std::vector<double> b) {
  const std::size_t n = a.rows();
  if (a.cols() != n || b.size() != n) throw std::invalid_argument("solveLinearSystem: dimension mismatch");
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    for (std::size_t r = col + 1; r < n; ++r)
      if (std::fabs(a(r, col)) > std::fabs(a(pivot, col))) pivot = r;
    if (std::fabs(a(pivot, col)) < 1e-300) throw std::runtime_error("solveLinearSystem: singular matrix");
    if (pivot != col) {
      for (std::size_t c = 0; c < n; ++c) std::swap(a(col, c), a(pivot, c));
      std::swap(b[col], b[pivot]);
    }
    for (std::size_t r = col + 1; r < n; ++r) {
      const double factor = a(r, col) / a(col, col);
      if (factor == 0.0) continue;
      for (std::size_t c = col; c < n; ++c) a(r, c) -= factor * a(col, c);
      b[r] -= factor * b[col];
    }
  }
  std::vector<double> x(n);
  for (std::size_t i = n; i-- > 0;) {
    double sum = b[i];
    for (std::size_t c = i + 1; c < n; ++c) sum -= a(i, c) * x[c];
    x[i] = sum / a(i, i);
  }
  return x;
}

Matrix transpose(const Matrix& a) {
  Matrix t(a.cols(), a.rows());
  for (std::size_t r = 0; r < a.rows(); ++r)
    for (std::size_t c = 0; c < a.cols(); ++c) t(c, r) = a(r, c);
  return t;
}

}  // namespace ccr
