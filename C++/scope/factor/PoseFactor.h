#pragma once

#include <scope/factor/Factor.h>

namespace scope {
class PoseFactor : public Factor {
public:
  struct Evaluation : public Factor::Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix3 errorR;
    Vector6 xi;
  };

  struct Linearization : public Factor::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix3 dexpinv;
    Matrix3 dexpinvR;
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_FACTOR_EVALUATION_NEW
  SCOAT_FACTOR_LINEARIZATION_NEW

  PoseFactor(int pose, const Matrix6 &sqrtCov, const Pose &mean,
             const std::string &name = "", int index = -1, bool active = true);

  virtual int evaluate(const AlignedVector<Pose> &poses,
                       const AlignedVector<VectorX> &shapes,
                       const AlignedVector<Matrix3> &joints,
                       const AlignedVector<VectorX> &params,
                       Factor::Evaluation &base_eval) const override;

  virtual int linearize(const AlignedVector<Pose> &poses,
                        const AlignedVector<VectorX> &shapes,
                        const AlignedVector<Matrix3> &joints,
                        const AlignedVector<VectorX> &params,
                        const Factor::Evaluation &base_eval,
                        Factor::Linearization &base_lin) const override;

protected:
  Matrix6 mSqrtCov;
  Pose mMean;
};
} // namespace scope
