#pragma once

#include <lapacke.h>

#include <array>
#include <exception>
#include <iosfwd>
#include <vector>

namespace Linalg {

class Vector {
 public:
  Vector(int size) : m_data(size), m_SIZE(size) {}
  Vector() : m_data(0), m_SIZE(0) {}

  int size() { return m_SIZE; };
  double &operator()(int n) {
    if (0 <= n && n < m_SIZE) {
      return m_data[n];
    } else {
      throw std::runtime_error("Index out of range");
    }
  }
  double operator()(int n) const {
    if (0 <= n && n < m_SIZE) {
      return m_data[n];
    } else {
      throw std::runtime_error("Index out of range");
    }
  }
  friend std::ostream &operator<<(std::ostream &os, const Vector &frac);

 private:
  std::vector<double> m_data;
  int m_SIZE;
};

struct LU_status {
  int status;
  std::vector<int> ipiv;
  operator int() const { return status; }
  LU_status(int size) : ipiv(size) {}
};

class Matrix {
 public:
  Matrix(int col, int row) : m_data(col * row), m_COL(col), m_ROW(row) {}
  Matrix() : m_data(0), m_COL(0), m_ROW(0) {}
  Matrix(const Matrix &mat) : m_COL(mat.m_COL), m_ROW(mat.m_ROW) {
    m_data = mat.m_data;
  }

  // {col, row}
  std::array<int, 2> shape() const { return {m_COL, m_ROW}; }

  LU_status lu() {
    LU_status status(m_ROW);
    status.status = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m_COL, m_ROW,
                                   m_data.data(), m_COL, status.ipiv.data());
    return status;
  }

  double det() {
    double det(1.);
    Matrix m = *this;
    if (m.m_ROW != m.m_COL) {
      throw std::runtime_error("Matrix is not squre");
    }
    auto status = m.lu();
    for (int i = 0; i < m.m_COL; ++i) {
      det *= m(i, i);
      if (status.ipiv.at(i) != i + 1) {
        det *= -1;
      }
    }
    return det;
  }

  friend std::ostream &operator<<(std::ostream &os, const Matrix &frac);
  double &operator()(int col, int row) {
    if (col < m_COL && row < m_ROW) {
      return m_data[col * m_ROW + row];
    } else {
      throw std::runtime_error("Index out of range");
    }
  }

  double operator()(int col, int row) const {
    if (col < m_COL && row < m_ROW) {
      return m_data[col * m_ROW + row];
    } else {
      throw std::runtime_error("Index out of range");
    }
  }

 private:
  std::vector<double> m_data;
  int m_COL, m_ROW;
  int m_precision{3};
};

}  // namespace Linalg
