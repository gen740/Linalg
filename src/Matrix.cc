#include "linalg/Matrix.h"

#include <cblas.h>
#include <lapacke.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

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

template <>
int Matrix<double>::svd(Vector<double> &s, Matrix<double> &u,
                        Matrix<double> &v) {
  Vector<double> superb(std::min(m_ROW, m_COL));
  u.reshape(m_COL, m_COL);
  v.reshape(m_ROW, m_ROW);
  s.reshape(std::max(m_COL, m_ROW));
  return LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', m_ROW, m_COL, *this, m_ROW,
                        s, u, m_ROW, v, m_COL, superb);
}

template <>
int Matrix<float>::svd(Vector<float> &s, Matrix<float> &u, Matrix<float> &v) {
  Vector<float> superb(std::min(m_ROW, m_COL));
  u.reshape(m_COL, m_COL);
  v.reshape(m_ROW, m_ROW);
  s.reshape(std::max(m_COL, m_ROW));
  return LAPACKE_sgesvd(LAPACK_COL_MAJOR, 'A', 'A', m_ROW, m_COL, *this, m_ROW,
                        s, u, m_ROW, v, m_COL, superb);
}

template <>
Matrix<double> Matrix<double>::operator*(Matrix<double> mat) {
  auto mat_shape = mat.shape();
  if (m_ROW != mat_shape[0]) {
    throw std::runtime_error("Cannot Multiply matrix");
  }
  // dgemm();
  Matrix<double> ret(m_COL, mat_shape[1]);
  cblas_dgemm(CBLAS_ORDER::CblasColMajor, CBLAS_TRANSPOSE::CblasNoTrans,
              CBLAS_TRANSPOSE::CblasNoTrans, m_COL, mat.shape()[1], m_ROW, 1,
              *this, m_COL, mat, mat_shape[0], 1, ret, m_COL);
  return ret;
}

template std::ostream &operator<<(std::ostream &os, const Matrix<double> &mat);
template std::ostream &operator<<(std::ostream &os, const Matrix<float> &mat);

template class Matrix<double>;
template class Matrix<float>;

}  // namespace Linalg
