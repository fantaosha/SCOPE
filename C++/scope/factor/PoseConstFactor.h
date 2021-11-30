#pragma once

#include <scope/factor/Factor.h>

namespace scope {
class PoseConstFactor : public Factor {
 public:
  struct Evaluation : public Factor::Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix3 errorR;
    Vector6 xi;

    Vector6 errorL;
    Vector6 errorU;

    Vector6 errorLInv;
    Vector6 errorUInv;

    Vector6 expL;
    Vector6 expU;
  };

  struct Linearization : public Factor::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix3 dexpinv;

    Vector6 Derror;
  };

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_FACTOR_EVALUATION_NEW
  SCOAT_FACTOR_LINEARIZATION_NEW

  PoseConstFactor(int pose, const Vector6 &weight, const Pose &mean,
                  const Vector6 &lbnd, const Vector6 &ubnd,
                  const std::string &name = "", int index = -1,
                  bool active = true);

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
  Vector6 mWeight;
  Pose mMean;

  Vector6 mLowerBnd;
  Vector6 mUpperBnd;

  const Scalar mScale = 1e-2;
  const Scalar mBnd = -1e-6;
};
}  // namespace scope
