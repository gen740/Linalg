#include <gtest/gtest.h>

#include <Linalg/Core.hpp>

using Linalg::Matrix;

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
  m.to_vec();
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
