#include <gtest/gtest.h>
#include <linalg/Core.h>

using Linalg::Matrix;
using Linalg::Vector;

TEST(MatrixTest, Basic) {
  Linalg::Matrix<> m(4, 4);
  EXPECT_EQ(m.shape()[0], 4);
  EXPECT_EQ(m.shape()[1], 4);
}

TEST(MatrixTest, IndexAccess) {
  Matrix<> m(3, 3);
  m(1, 1) = 1;
  EXPECT_DOUBLE_EQ(m[0][0], 1);
  EXPECT_DOUBLE_EQ(m(1, 1), 1);
  m(3, 3) = 3;
  EXPECT_DOUBLE_EQ(m(3, 3), 3);
  m[1][1] = 4;
  EXPECT_DOUBLE_EQ(m(2, 2), 4);
}

// cout/save でどれだけの精度で出力するかが設定できる。
TEST(MatrixTest, ChangePrecision) {
  Linalg::Matrix<> m(2, 2);
  // デフォルトの精度は 4桁
  EXPECT_EQ(m.get_precision(), 4);
  // メンバーを介して精度を変えられる。
  m.set_precision(5);
  // 精度を変えた後は double / float の精度が変わる
  EXPECT_EQ(m.get_precision(), 5);
  EXPECT_EQ(Linalg::Matrix<double>::get_precision(), 5);
  EXPECT_EQ(Linalg::Matrix<float>::get_precision(), 5);
  Linalg::Matrix<> n(2, 2);
}

TEST(MatrixTest, Advance) {
  Linalg::Matrix<double> m(5, 5);
  EXPECT_EQ(m.to_vec().size(), 25);
  m(3, 3) = 5;
  EXPECT_FLOAT_EQ(m.to_vec()(13), 5);
  for (int i = 1; i <= 5; ++i) {
    m(i, i) = 1;
  }
  auto state = m.lu();
  EXPECT_EQ(state.status, 0);
  std::cout << m.det() << std::endl;
  EXPECT_FLOAT_EQ(m.det(), 1);
}

TEST(MatrixTest, Operation) {
  Matrix<double> m({
      {1, 2, 3},  //
      {4, 5, 6}   //
  });
  Matrix<double> l({{1, 2}, {3, 4}, {5, 6}});
  std::cout << m << std::endl;
  std::cout << l << std::endl;
  std::cout << m * l << std::endl;
}

TEST(MatrixTest, SVD) {
  Matrix<double> m({
      {1, 2, 3},  //
      {4, 5, 6},  //
      {7, 8, 10}  //
  });
  std::cout << m(2, 1) << std::endl;
  Vector<double> s;
  Matrix<double> u;
  Matrix<double> v;
  m.svd(s, u, v);
  // std::cout << s << std::endl;
  // std::cout << u << std::endl;
  // std::cout << v << std::endl;
}
