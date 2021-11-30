#pragma once

#include <scope/base/Types.h>
#include <Eigen/Dense>

namespace scope {
namespace math {
struct SO3 {
  typedef Matrix3 LieGroup;
  typedef Vector3 LieAlgebra;

  template <typename T1, typename T2>
  static void exp(Eigen::MatrixBase<T1> const &w, Eigen::MatrixBase<T2> &res) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 3)

    Scalar const th = w.stableNorm();
    LieAlgebra s = w;

    if (th < 5e-2) {
      Scalar c, d, th2, th4, th6, d2, thd2;
      th2 = th * th;
      th4 = th2 * th2;
      th6 = th4 * th2;
      d = 1 + th2 / 12 + th4 / 120 + th6 / 1185.88235294117647058823529412;

      d2 = d * d;
      thd2 = th2 * d2;
      c = 2 / (4 + thd2);

      s *= c * d2;

      res.noalias() = s * w.transpose();
      res.diagonal().array() += 1 - c * thd2;

      s *= 2 / d;
    } else {
      // use exponential map

      Scalar sth, cth;

      fsincos(th, &sth, &cth);

      s /= th;

      res.noalias() = (1 - cth) * s * s.transpose();
      res.diagonal().array() += cth;

      s *= sth;
    }

    res(0, 1) -= s[2];
    res(1, 0) += s[2];
    res(0, 2) += s[1];
    res(2, 0) -= s[1];
    res(1, 2) -= s[0];
    res(2, 1) += s[0];
  }

  template <typename T1, typename T2>
  static void log(Eigen::MatrixBase<T1> const &R, Eigen::MatrixBase<T2> &res) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T1, 3, 3)
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T2, 3)

    Scalar const tr = R.trace() + 1;

    if (fabs(tr) < 1e-9) {
      // theta = pi and use eigenvalue decomposition

      if (R(0, 0) > R(1, 1)) {
        if (R(0, 0) > R(2, 2)) {
          res[0] = std::sqrt((R(0, 0) + 1));

          res[1] = R(1, 0) / res[0];
          res[2] = R(2, 0) / res[0];
        } else {
          res[2] = std::sqrt((R(2, 2) + 1));

          res[0] = R(0, 2) / res[2];
          res[1] = R(1, 2) / res[2];
        }
      } else {
        if (R(1, 1) > R(2, 2)) {
          res[1] = std::sqrt((R(1, 1) + 1));

          res[0] = R(0, 1) / res[1];
          res[2] = R(2, 1) / res[1];
        } else {
          res[2] = std::sqrt((R(2, 2) + 1));

          res[0] = R(0, 2) / res[2];
          res[1] = R(1, 2) / res[2];
        }
      }

      res *= 2.221441469079183;  // multiplied by  pi/sqrt(2)

    } else {
      res(0) = R(2, 1) - R(1, 2);
      res(1) = R(0, 2) - R(2, 0);
      res(2) = R(1, 0) - R(0, 1);

      if (tr > INVERSE_OF_RETRACT_TOL) {
        // use cayley map when th is small for fast approximation
        // res *= 2 / (1.0 + tr);
        res *= (3.0666666666666669 * tr - 5.8666666666666663 +
                6.4000000000000004 / tr) /
               (tr * tr);
      } else {
        // use exponential map
        Scalar const th = acos((tr - 2) / 2);
        res *= 0.5 * th / sin(th);
      }
    }
  }

  template <typename T1, typename T2>
  static void dexp(Eigen::MatrixBase<T1> const &w, Eigen::MatrixBase<T2> &res) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 3)

    Scalar const th = w.stableNorm();

    LieAlgebra s = w;

    if (th < 5e-2) {
      // use second-order taylor expansion when th is small for fast
      // approximation relative approx. error <= 1e-6 for DRETRACT_TOL = 0.1 rad

      Scalar th2 = th * th;
      Scalar c = 0.166666666666667 - th2 / 120;

      res.noalias() = c * w * w.transpose();
      res.diagonal().array() += 1 - c * th2;

      s *= 0.5 - th2 / 24.0;
    } else {
      // use exponential map
      Scalar sth, cth, th2, th3;

      fsincos(th, &sth, &cth);

      th2 = th * th;
      th3 = th2 * th;

      res.noalias() = (th - sth) / th3 * s *
                      s.transpose();  // (th-sth)/th^3 -> 1/6 as th -> 0
      res.diagonal().array() += sth / th;

      s *= (1 - cth) / th2;  // (1-cth)/th^2 -> 1/2 as th -> 0
    }

    res(0, 1) -= s[2];
    res(1, 0) += s[2];
    res(0, 2) += s[1];
    res(2, 0) -= s[1];
    res(1, 2) -= s[0];
    res(2, 1) += s[0];
  }

  template <typename T1, typename T2>
  static void dexpinv(Eigen::MatrixBase<T1> const &w,
                      Eigen::MatrixBase<T2> &res) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 3)

    Scalar const th = w.stableNorm();

    LieAlgebra s = w;

    if (th < 5e-2) {
      // use cayley map when th is small for fast approximation
      Scalar th2 = th * th;
      Scalar c = 0.333333333333333 + th2 / 180;

      s *= -0.5;

      res.noalias() = c * s * s.transpose();
      res.diagonal().array() += 1 - 0.25 * c * th2;
    } else {
      // use exponential map
      Scalar sth, cth;

      fsincos(th, &sth, &cth);

      sth *= th;
      cth = 2 * cth - 2;

      res.noalias() = (sth + cth) / (th * th * cth) * s * s.transpose();
      res.diagonal().array() -= sth / cth;

      s *= -0.5;
    }

    res(0, 1) -= s[2];
    res(1, 0) += s[2];
    res(0, 2) += s[1];
    res(2, 0) -= s[1];
    res(1, 2) -= s[0];
    res(2, 1) += s[0];
  }

  template <typename T1, typename T2>
  static void cay(Eigen::MatrixBase<T1> const &w, Eigen::MatrixBase<T2> &res) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 3)

    Scalar const th = w.stableNorm();
    LieAlgebra s = w;

    // using cayley map
    s *= 2 / (4 + th * th);

    res.noalias() = s * w.transpose();
    res.diagonal().array() += 1 - s.dot(w);

    s *= 2;

    res(0, 1) -= s[2];
    res(1, 0) += s[2];
    res(0, 2) += s[1];
    res(2, 0) -= s[1];
    res(1, 2) -= s[0];
    res(2, 1) += s[0];
  }

  template <typename T1, typename T2>
  static void cayinv(Eigen::MatrixBase<T1> const &R,
                     Eigen::MatrixBase<T2> &res) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T1, 3, 3)
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T2, 3)

    Scalar const tr = R.trace();

    res(0) = R(2, 1) - R(1, 2);
    res(1) = R(0, 2) - R(2, 0);
    res(2) = R(1, 0) - R(0, 1);

    res /= 2 / (1.0 + tr);
  }

  template <typename T1, typename T2>
  static void dcay(Eigen::MatrixBase<T1> const &w, Eigen::MatrixBase<T2> &res) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 3)

    Scalar const th = w.stableNorm();

    LieAlgebra s = w;

    Scalar c = 2 / (4 + th * th);

    s *= c;

    res.diagonal().array() = 2 * c;

    res(0, 1) = -s[2];
    res(1, 0) = s[2];
    res(0, 2) = s[1];
    res(2, 0) = -s[1];
    res(1, 2) = -s[0];
    res(2, 1) = s[0];
  }

  template <typename T1, typename T2>
  static void dcayinv(Eigen::MatrixBase<T1> const &w,
                      Eigen::MatrixBase<T2> &res) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T1, 3)
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T2, 3, 3)

    Scalar const th = w.stableNorm();

    LieAlgebra s = w;

    s *= -0.5;

    res.noalias() = s * s.transpose();
    res.diagonal().array() += 1 - 0.25 * th;

    res(0, 1) -= s[2];
    res(1, 0) += s[2];
    res(0, 2) += s[1];
    res(2, 0) -= s[1];
    res(1, 2) -= s[0];
    res(2, 1) += s[0];
  }

  template <typename T1, typename T2>
  static void Rot2XYZ(Eigen::MatrixBase<T1> const &R,
                      Eigen::MatrixBase<T2> &res) {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(T1, 3, 3)
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T2, 3)

    if (1 - fabs(R(0, 2)) >= 5e-11) {
      res[0] = atan2(-R(1, 2), R(2, 2));
      res[1] = asin(R(0, 2));
      res[2] = atan2(-R(0, 1), R(0, 0));
    } else {
      res[0] = atan2(R(1,0), R(1,1));
      res[1] = R(0, 2) > 0 ? 0.5 * M_PI : -0.5 * M_PI;
      res[2] = 0;
    }
  }

  static void fsincos(double x, double *sin, double *cos) {
    sincos(x, sin, cos);
  }

  static void fsincos(float x, float *sin, float *cos) { sincosf(x, sin, cos); }

  static void fsincos(long double x, long double *sin, long double *cos) {
    sincosl(x, sin, cos);
  }
};
}  // namespace math
}  // namespace scope
