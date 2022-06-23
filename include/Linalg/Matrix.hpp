#pragma once
#include <lapacke.h>

#include <array>
#include <exception>
#include <iosfwd>
#include <vector>

#include "Linalg/Vector.hpp"

namespace Linalg {
class Vector;

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
  Matrix(const Matrix &mat, int col, int row) : m_COL(col), m_ROW(row) {
    if (mat.m_ROW * mat.m_COL != col * row) {
      throw std::runtime_error("配列の大きさが違います");
    }
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
  friend Vector;

  // Vector に変換する
  Vector to_vec();

  friend std::ostream &operator<<(std::ostream &os, const Matrix &frac);
  double &operator()(int col, int row) {
    if (col < m_COL && row < m_ROW) {
      return m_data[row * m_COL + col];
    } else {
      throw std::runtime_error("Index out of range");
    }
  }

  double operator()(int col, int row) const {
    if (col < m_COL && row < m_ROW) {
      return m_data[row * m_COL + col];
    } else {
      throw std::runtime_error("Index out of range");
    }
  }

  double *data() { return m_data.data(); }

  void save(const char *filename, const char delimeter = ',');
  void set_precision(int precision) { m_precision = precision; }

 private:
  std::vector<double> m_data;
  int m_COL, m_ROW, m_precision{4};
};

}  // namespace Linalg
