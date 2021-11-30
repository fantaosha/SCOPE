#pragma once

#include <scope/initializer/factor/Factor.h>

namespace scope {
namespace Initializer {
class POFFactor : public Factor {
public:
  struct Evaluation : public Factor::Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Vector3 d; // direction

    Scalar squaredErrorNorm;
  };

  struct Linearization : public Factor::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix<1, 3> jacobian;

    Scalar sqrtDrho;
    Scalar alpha;
    Scalar scaledError;
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_INITIAL_EVALUATION_NEW
  SCOAT_INITIAL_LINEARIZATION_NEW

  POFFactor(int pose, const Vector3 &S, const Scalar &sigma, const Scalar &eps,
            const Vector3 &measurement, const Scalar &confidence = 1.0,
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

  Vector3 mMeasurement;

  Scalar mCon;
  Scalar mSqrtCon;
  Scalar mSigmaCon;

  Scalar mSigma;
  Scalar mGMThreshold;
  Scalar mMinAlpha;

  Vector3 mS;
};
} // namespace Initializer
} // namespace scope
