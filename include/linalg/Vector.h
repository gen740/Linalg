#pragma once

#include <lapacke.h>

#include <array>
#include <exception>
#include <fstream>
#include <iostream>
#include <vector>

#include "linalg/Matrix.h"
#include "linalg/Types.h"

namespace Linalg {

template <FloatingPointType T>
class Matrix;

template <FloatingPointType T>
class Vector {
 public:
  Vector(int size) : m_data(size), m_SIZE(size) {}
  Vector() : m_data(0), m_SIZE(0) {}
  Vector(const Vector<T> &vec) : m_data(vec.m_data), m_SIZE(vec.m_SIZE) {}
  Vector(const Matrix<T> &mat) {
    m_data = mat.m_data;
    m_SIZE = mat.m_ROW * mat.m_COL;
  }
  Vector(Matrix<T> &&mat) {
    m_data = std::move(mat.m_data);
    m_SIZE = mat.m_ROW * mat.m_COL;
  }
  int size() { return m_SIZE; };

  void reshape(int size) {
    m_SIZE = size;
    m_data.resize(size);
  }

  // 1-index
  T &operator()(int n) {
    if (1 <= n && n <= m_SIZE) {
      return m_data[n - 1];
    } else {
      throw std::runtime_error("Index out of range");
    }
  }

  // 1-index
  T operator()(int n) const {
    if (0 <= n && n < m_SIZE) {
      return m_data[n];
    } else {
      throw std::runtime_error("Index out of range");
    }
  }

  operator T *() { return m_data.data(); }

  Matrix<T> to_mat(int col, int row) { return Vector<T>(*this); }

  template <FloatingPointType U>
  friend std::ostream &operator<<(std::ostream &os, const Vector<U> &vec);

  friend Matrix<T>;

  T *data() { return m_data.data(); }

 private:
  std::vector<T> m_data;
  int m_SIZE;
  static int m_precision;
};

}  // namespace Linalg
