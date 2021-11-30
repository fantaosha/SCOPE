#pragma once
//#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_AVX2

#include <Eigen/Dense>
#include <type_traits>
#include <vector>

namespace scope {
using Scalar = double;

template <int rows, int cols>
using Matrix = Eigen::Matrix<Scalar, rows, cols>;
template <int size>
using Vector = Eigen::Matrix<Scalar, size, 1>;
template <int size>
using Diagonal = Eigen::Diagonal<Scalar, size>;

using MatrixX = Matrix<Eigen::Dynamic, Eigen::Dynamic>;
using VectorX = Vector<Eigen::Dynamic>;
using Matrix2X = Matrix<2, Eigen::Dynamic>;
using Matrix3X = Matrix<3, Eigen::Dynamic>;
using MatrixX2 = Matrix<Eigen::Dynamic, 2>;
using MatrixX3 = Matrix<Eigen::Dynamic, 3>;
using Matrix2 = Matrix<2, 2>;
using Matrix3 = Matrix<3, 3>;
using Matrix4 = Matrix<4, 4>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;
using Vector4 = Vector<4>;
using Matrix4X = Matrix<4, Eigen::Dynamic>;
using MatrixX4 = Matrix<Eigen::Dynamic, 4>;
using Matrix6X = Matrix<6, Eigen::Dynamic>;
using MatrixX6 = Matrix<Eigen::Dynamic, 6>;
using Matrix63 = Matrix<6, 3>;
using Matrix26 = Matrix<2, 6>;
using Matrix36 = Matrix<3, 6>;
using Matrix6 = Matrix<6, 6>;
using Vector6 = Matrix<6, 1>;

template <int rows, int cols>
using RowMajorMatrix = Eigen::Matrix<Scalar, rows, cols, Eigen::RowMajor>;

using RowMajorMatrixX = RowMajorMatrix<Eigen::Dynamic, Eigen::Dynamic>;
using RowMajorMatrix3X = RowMajorMatrix<3, Eigen::Dynamic>;
using RowMajorMatrix36 = RowMajorMatrix<3, 6>;

template <typename T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

enum class HumanModel { SMPL };
}  // namespace scope
