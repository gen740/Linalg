#include "Linalg/Vector.hpp"

#include <iomanip>

namespace Linalg {

static int vector_precision = 4;

template <>
int Vector<double>::m_precision = vector_precision;

template <>
int Vector<float>::m_precision = vector_precision;

template <FloatingPointType U>
std::ostream &operator<<(std::ostream &os, const Vector<U> &vec) {
  os << "Vector n = " << vec.m_SIZE << "\n";
  os << std::setprecision(vec.m_precision) << std::scientific;
  for (int i = 0; i < vec.m_SIZE; ++i) {
    os << "\t" << vec(i);
  }
  os << std::defaultfloat;
  return os;
}

template std::ostream &operator<<(std::ostream &os, const Vector<double> &vec);
template std::ostream &operator<<(std::ostream &os, const Vector<float> &vec);

}  // namespace Linalg
