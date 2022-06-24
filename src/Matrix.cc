#include "Linalg/Matrix.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace Linalg {

namespace {
int matrix_precision = 4;
}  // namespace

template <>
int *const Matrix<double>::m_precision = &matrix_precision;

template <>
int *const Matrix<float>::m_precision = &matrix_precision;

template <>
void Matrix<double>::set_precision(int precision) {
  *m_precision = precision;
}

template <>
void Matrix<float>::set_precision(int precision) {
  *m_precision = precision;
}

template <>
int Matrix<double>::get_precision() {
  return *m_precision;
}

template <>
int Matrix<float>::get_precision() {
  return *m_precision;
}

template <FloatingPointType U>
std::ostream &operator<<(std::ostream &os, const Matrix<U> &mat) {
  os << "Matrix " << mat.m_COL << "×" << mat.m_ROW << "\n";
  os << std::setprecision(*(mat.m_precision)) << std::scientific;
  for (int i = 1; i <= mat.m_COL; ++i) {
    os << "\t";
    for (int j = 1; j <= mat.m_ROW; ++j) {
      if (j == mat.m_ROW) {
        if (i == mat.m_COL) {
          os << mat(i, j);
          break;
        }
        os << mat(i, j) << "\n";
      } else {
        os << mat(i, j) << "\t";
      }
    }
  }
  os << std::defaultfloat;
  return os;
}

template <>
void Matrix<>::save(const char *filename, const char delimeter,
                    bool is_scientific) {
  std::ofstream file(filename);
  if (is_scientific) {
    file << std::scientific;
  }
  for (int i = 1; i <= m_COL; ++i) {
    for (int j = 1; j <= m_ROW; ++j) {
      if (j == m_ROW) {
        if (i == m_COL) {
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

template <>
LU_status Matrix<double>::lu() {
  LU_status status(m_ROW);
  status.status = LAPACKE_dgetrf(LAPACK_COL_MAJOR, m_COL, m_ROW, m_data.data(),
                                 m_COL, status.ipiv.data());
  return status;
}

template <>
LU_status Matrix<float>::lu() {
  LU_status status(m_ROW);
  status.status = LAPACKE_sgetrf(LAPACK_COL_MAJOR, m_COL, m_ROW, m_data.data(),
                                 m_COL, status.ipiv.data());
  return status;
}

template std::ostream &operator<<(std::ostream &os, const Matrix<double> &mat);
template std::ostream &operator<<(std::ostream &os, const Matrix<float> &mat);

template class Matrix<double>;
template class Matrix<float>;

}  // namespace Linalg
