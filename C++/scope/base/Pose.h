#pragma once

#include <scope/base/Types.h>
#include <scope/math/SO3.h>

namespace scope {
class Pose {
 public:
  Pose() : R(matrix.data()), t(matrix.data() + 9) {
    R.setIdentity();
    t.setZero();
  }

  Pose(const Pose& T)
      : matrix(T.matrix), R(matrix.data()), t(matrix.data() + 9) {}

  Pose(const Matrix3& R, const Vector3& t): Pose() {
    this->R = R;
    this->t = t;
  }

  Pose& operator=(const Pose& T) {
    matrix = T.matrix;

    return *this;
  }

  Pose(Pose&& T) : matrix(T.matrix), R(std::move(T.R)), t(std::move(T.t)) {}

  Pose& operator=(Pose&& T) {
    matrix = std::move(T.matrix);
    R = std::move(T.R);
    t = std::move(T.t);

    return *this;
  }

  static const Pose& Identity() {
    static Pose identity;

    return identity;
  }

  template <typename T>
  static int exp(Eigen::MatrixBase<T> const& xi, Pose& res) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(T, 6)
    const auto& w = xi.template head<3>();
    const auto& v = xi.template tail<3>();

    // TODO: Optimize the code
    const Scalar th = w.stableNorm();
    Vector3 s = w;

    if (th < 1e-2) {
      Scalar c, d, th2, th4, th6, d2, thd2;
      th2 = th * th;
      th4 = th2 * th2;
      th6 = th4 * th2;
      d = 1 + th2 / 12 + th4 / 120 + th6 / 1185.88235294117647058823529412;

      d2 = d * d;
      thd2 = th2 * d2;
      c = 2 / (4 + thd2);

      s *= c * d2;

      res.R.noalias() = s * w.transpose();
      res.R.diagonal().array() += 1 - c * thd2;

      s *= 2 / d;

      res.t = v;
      res.t += 0.1666666666666666666666666666667 * w.dot(v) * w;
      res.t += 0.5 * w.cross(v);
    } else {
      Scalar sth, cth;

      math::SO3::fsincos(th, &sth, &cth);

      s /= th;

      res.R.noalias() = (1 - cth) * s * s.transpose();
      res.R.diagonal().array() += cth;

      s *= sth;

      res.t = sth * v;
      res.t += (th - sth) * w.dot(v) * w;
      res.t += (1 - cth) * w.cross(v);
      res.t /= th;
    }

    res.R(0, 1) -= s[2];
    res.R(1, 0) += s[2];
    res.R(0, 2) += s[1];
    res.R(2, 0) -= s[1];
    res.R(1, 2) -= s[0];
    res.R(2, 1) += s[0];

    return 0;
  }

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix<Scalar, 3, 4> matrix;
  Eigen::Map<Matrix3> R;  // rotational part
  Eigen::Map<Vector3> t;  // translational part
};
}  // namespace scope
