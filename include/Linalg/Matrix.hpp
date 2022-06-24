#pragma once
#include <lapacke.h>

#include <array>
#include <concepts>
#include <iosfwd>
#include <vector>

#include "Linalg/Types.hpp"
#include "Linalg/Vector.hpp"

namespace Linalg {

template <FloatingPointType T>
class Vector;

struct LU_status {
  int status;
  std::vector<int> ipiv;
  operator int() const { return status; }
  LU_status(int size) : ipiv(size) {}
};

template <FloatingPointType T = double>
class Matrix {
 public:
  Matrix(int col, int row) : m_data(col * row), m_COL(col), m_ROW(row) {}
  Matrix() : m_data(0), m_COL(0), m_ROW(0) {}
  Matrix(const Matrix &mat) : m_COL(mat.m_COL), m_ROW(mat.m_ROW) {
    m_data = mat.m_data;
  }
  Matrix(const Matrix<T> &mat, int col, int row) : m_COL(col), m_ROW(row) {
    if (mat.m_ROW * mat.m_COL != col * row) {
      throw std::runtime_error("配列の大きさが違います");
    }
    m_data = mat.m_data;
  }
  Matrix(Matrix<T> &&mat, int col, int row) : m_COL(col), m_ROW(row) {
    if (mat.m_ROW * mat.m_COL != col * row) {
      throw std::runtime_error("配列の大きさが違います");
    }
    m_data = std::move(mat.m_data);
  }
  Matrix(const Vector<T> &vec, int col, int row) : m_COL(col), m_ROW(row) {
    if (vec.m_SIZE != col * row) {
      throw std::runtime_error("配列の大きさが違います");
    }
    m_data = vec.m_data;
  }

  // {col, row}
  std::array<int, 2> shape() const { return {m_COL, m_ROW}; }

  LU_status lu();

  T det() {
    T det(1.);
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
  friend Vector<T>;

  // Vector に変換する
  Vector<T> to_vec() { return Vector<T>(*this); }

  template <FloatingPointType U>
  friend std::ostream &operator<<(std::ostream &os, const Matrix<U> &mat);

  T &operator()(int col, int row) {
    if (1 <= col && col <= m_COL && 1 <= row && row <= m_ROW) {
      return m_data[(row - 1) * m_COL + (col - 1)];
    } else {
      throw std::runtime_error("Index out of range");
    }
  }
  T operator()(int col, int row) const {
    if (1 <= col && col <= m_COL && 1 <= row && row <= m_ROW) {
      return m_data[(row - 1) * m_COL + (col - 1)];
    } else {
      throw std::runtime_error("Index out of range");
    }
  }
  T *operator[](int i) { return &m_data.data()[i * m_COL]; }

  operator T *() { return m_data.data(); }

  void save(const char *filename, const char delimeter = ',',
            bool is_scientific = true);

  static void set_precision(int precision);
  static int get_precision();

 private:
  static int *const m_precision;

  std::vector<T> m_data;
  int m_COL, m_ROW;
};

}  // namespace Linalg
