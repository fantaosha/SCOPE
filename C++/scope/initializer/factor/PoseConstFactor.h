#pragma once

#include <scope/initializer/factor/Factor.h>

namespace scope {
namespace Initializer {
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

    Matrix6 jacobian;

    Matrix3 dexpinv;

    Vector6 Derror;
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_INITIAL_EVALUATION_NEW
  SCOAT_INITIAL_LINEARIZATION_NEW

  PoseConstFactor(int pose, const Vector6 &weight, const Pose &mean,
                  const Vector6 &lbnd, const Vector6 &ubnd,
                  const std::string &name = "", int index = -1);

  virtual int evaluate(const AlignedVector<Pose> &poses,
                       const AlignedVector<Matrix3> &joints,
                       Factor::Evaluation &base_eval) const override;

  virtual int linearize(const AlignedVector<Pose> &poses,
                        const AlignedVector<Matrix3> &joints,
                        const Factor::Evaluation &base_eval,
                        Factor::Linearization &base_lin) const override;

  int getPose() const { return mPose; }

protected:
  int mPose;

  Vector6 mWeight;
  Pose mMean;

  Vector6 mLowerBnd;
  Vector6 mUpperBnd;

  const Scalar mScale = 2.5e-2;
  const Scalar mBnd = -1e-6;
};
} // namespace Initializer
} // namespace scope
