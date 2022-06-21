#pragma once

#include <lapacke.h>

#include <array>
#include <exception>
#include <iosfwd>
#include <vector>

#include "Linalg/Matrix.hpp"

namespace Linalg {
class Matrix;

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

  Matrix to_mat(int col, int low);
  friend std::ostream &operator<<(std::ostream &os, const Vector &frac);
  friend Matrix;

  double *data() { return m_data.data(); }

 protected:
  std::vector<double> m_data;
  int m_SIZE;
};

}  // namespace Linalg
