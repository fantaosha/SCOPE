#pragma once

#include <scope/initializer/factor/Factor.h>

namespace scope {
namespace Initializer {
class PoseFactor : public Factor {
public:
  struct Evaluation : public Factor::Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix3 errorR;
    Vector6 xi;
  };

  struct Linearization : public Factor::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix6 jacobian;
    Matrix3 dexpinv;
    Matrix3 dexpinvR;
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_INITIAL_EVALUATION_NEW
  SCOAT_INITIAL_LINEARIZATION_NEW

  PoseFactor(int pose, const Matrix6 &sqrtCov, const Pose &mean,
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

  Matrix6 mSqrtCov;
  Pose mMean;
};
} // namespace Initializer
} // namespace scope
