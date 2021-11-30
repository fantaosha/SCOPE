#pragma once

#include <Eigen/Dense>

#include <scope/math/macro.h>
#include <scope/math/operation.h>

namespace scope {
namespace math {
struct skew3 {
  template <math::OPS op, typename T1, typename T2>
  static Eigen::MatrixBase<T2>& set(Eigen::MatrixBase<T1> const& p,
                                    Eigen::MatrixBase<T2>& res) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 3)

    switch (op) {
      case OPS::EQL:
        res(0, 0) = 0;
        res(0, 1) = -p[2];
        res(0, 2) = p[1];
        res(1, 0) = p[2];
        res(1, 1) = 0;
        res(1, 2) = -p[0];
        res(2, 0) = -p[1];
        res(2, 1) = p[0];
        res(2, 2) = 0;

        break;

      case OPS::ADD:
        res(0, 0) += 0;
        res(0, 1) += -p[2];
        res(0, 2) += p[1];
        res(1, 0) += p[2];
        res(1, 1) += 0;
        res(1, 2) += -p[0];
        res(2, 0) += -p[1];
        res(2, 1) += p[0];
        res(2, 2) += 0;

        break;

      case OPS::SUB:
        res(0, 0) -= 0;
        res(0, 1) -= -p[2];
        res(0, 2) -= p[1];
        res(1, 0) -= p[2];
        res(1, 1) -= 0;
        res(1, 2) -= -p[0];
        res(2, 0) -= -p[1];
        res(2, 1) -= p[0];
        res(2, 2) -= 0;

        break;
    }

    return res;
  }

  template <math::OPS op, typename T1, typename T2, typename T3>
  static Eigen::MatrixBase<T3>& multL(Eigen::MatrixBase<T1> const& p,
                                      Eigen::MatrixBase<T2> const& other,
                                      Eigen::MatrixBase<T3>& res) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_ROWS(T2, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_ROWS(T3, 3)
    eigen_assert(other.cols() == res.cols());

    int const n = other.cols();

    switch (op) {
      case OPS::EQL:
#ifdef USE_OPENMP
#pragma omp parallel for default(shared) if (n > 100)
#endif
        for (int i = 0; i < n; i++) {
          res.col(i).noalias() = p.cross(other.col(i));
        }

        break;

      case OPS::ADD:
#ifdef USE_OPENMP
#pragma omp parallel for default(shared) if (n > 100)
#endif
        for (int i = 0; i < n; i++) {
          res.col(i).noalias() += p.cross(other.col(i));
        }

        break;

      case OPS::SUB:
#ifdef USE_OPENMP
#pragma omp parallel for default(shared) if (n > 100)
#endif
        for (int i = 0; i < n; i++) {
          res.col(i).noalias() -= p.cross(other.col(i));
        }

        break;

      default:
        assert(false && "Wrong op requesed value.");
        break;
    }

    return res;
  }

  template <math::OPS op, typename T1, typename T2, typename T3>
  static Eigen::MatrixBase<T3>& multR(Eigen::MatrixBase<T1> const& p,
                                      Eigen::MatrixBase<T2> const& other,
                                      Eigen::MatrixBase<T3>& res) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_COLS(T2, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_COLS(T3, 3)
    eigen_assert(other.rows() == res.rows());

    int const n = other.rows();

    switch (op) {
      case OPS::EQL:
#ifdef USE_OPENMP
#pragma omp parallel for default(shared) if (n > 100)
#endif
        for (int i = 0; i < n; i++) {
          res.row(i).noalias() = -p.cross(other.row(i));
        }

        break;

      case OPS::ADD:
#ifdef USE_OPENMP
#pragma omp parallel for default(shared) if (n > 100)
#endif
        for (int i = 0; i < n; i++) {
          res.row(i).noalias() -= p.cross(other.row(i));
        }

        break;

      case OPS::SUB:
#ifdef USE_OPENMP
#pragma omp parallel for default(shared) if (n > 100)
#endif
        for (int i = 0; i < n; i++) {
          res.row(i).noalias() += p.cross(other.row(i));
        }

        break;

      default:
        assert(false && "Wrong op requesed value.");
        break;
    }

    return res;
  }

  template <math::OPS op, typename T1, typename T2, typename T3>
  static Eigen::MatrixBase<T3>& mult2(Eigen::MatrixBase<T1> const& p1,
                                      Eigen::MatrixBase<T2> const& p2,
                                      Eigen::MatrixBase<T3>& res) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 3)
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T2, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T3, 3, 3)

    switch (op) {
      case OPS::EQL:
        res.noalias() = p2 * p1.transpose();
        res.diagonal().array() -= p2.dot(p1);

        break;

      case OPS::ADD:
        res.noalias() += p2 * p1.transpose();
        res.diagonal().array() -= p2.dot(p1);

        break;

      case OPS::SUB:
        res.noalias() -= p2 * p1.transpose();
        res.diagonal().array() += p2.dot(p1);

        break;
    }

    return res;
  }
};
}  // namespace math
}  // namespace scope
