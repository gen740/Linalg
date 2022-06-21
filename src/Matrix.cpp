#include "Linalg/Matrix.hpp"

#include <fstream>
#include <iomanip>

#include "Linalg/Vector.hpp"

namespace Linalg {

std::ostream &operator<<(std::ostream &os, const Matrix &mat) {
  os << "Matrix " << mat.m_COL << "×" << mat.m_ROW << "\n";
  // os << std::setprecision(mat.m_precision);
  for (int i = 0; i < mat.m_COL; ++i) {
    os << "\t";
    for (int j = 0; j < mat.m_ROW; ++j) {
      if (j == mat.m_ROW - 1) {
        if (i == mat.m_COL - 1) {
          os << mat(i, j);
          break;
        }
        os << mat(i, j) << "\n";
      } else {
        os << mat(i, j) << "\t";
      }
    }
  }
  return os;
}

Vector Matrix::to_vec() {
  Vector vec(m_COL * m_ROW);
  vec.m_data = m_data;
  return vec;
}

void Matrix::save(const char *filename, const char delimeter) {
  std::ofstream file(filename);
  for (int i = 0; i < m_COL; ++i) {
    file << "\t";
    for (int j = 0; j < m_ROW; ++j) {
      if (j == m_ROW - 1) {
        if (i == m_COL - 1) {
          file << (*this)(i, j);
          break;
        }
        file << (*this)(i, j) << '\n';
      } else {
        file << (*this)(i, j) << delimeter;
      }
    }
  }
  file << '\n';
}

}  // namespace Linalg
