#include "Linalg/Vector.hpp"

#include <iomanip>

namespace Linalg {

std::ostream &operator<<(std::ostream &os, const Vector &vec) {
  os << "Vector n = " << vec.m_SIZE << "\n";
  for (int i = 0; i < vec.m_SIZE; ++i) {
    os << "\t" << vec(i);
  }
  return os;
}

Matrix Vector::to_mat(int col, int row) {
  if (col * row != m_SIZE) {
    throw std::runtime_error("配列のサイズが違います。");
  }
  Matrix res(col, row);
  res.m_data = m_data;
  return res;
}

}  // namespace Linalg
