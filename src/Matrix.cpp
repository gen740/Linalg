#include "Linalg/Matrix.hpp"

#include <iomanip>

namespace Linalg {

std::ostream &operator<<(std::ostream &os, const Vector &vec) {
  os << "Vector n=" << vec.m_SIZE << "\n";
  for (int i = 0; i < vec.m_SIZE; ++i) {
    os << "\t" << vec(i);
  }
  return os;
}

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

}  // namespace Linalg
