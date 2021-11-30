#pragma once

#include <scope/initializer/factor/Factor.h>

namespace scope {
namespace Initializer {
class PinholeCameraFactor : public Factor {
public:
  struct Evaluation : public Factor::Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Vector3 pCam;

    Vector2 point2D;
    Scalar squaredErrorNorm;
  };

  struct Linearization : public Factor::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix26 jacobian;

    Scalar Drho;
    Scalar sqrtDrho;
    Scalar alpha;

    Vector2 scaledError;

    Vector6 g;
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_INITIAL_EVALUATION_NEW
  SCOAT_INITIAL_LINEARIZATION_NEW

  PinholeCameraFactor(int pose, const Vector3 &point, const Scalar &sigma,
                      const Scalar &eps, const Vector2 &measurement,
                      const Scalar &confidence = 1.0,
                      const std::string &name = "", int index = -1);

  virtual int evaluate(const AlignedVector<Pose> &poses,
                       const AlignedVector<Matrix3> &joints,
                       Factor::Evaluation &base_eval) const override;

  virtual int linearize(const AlignedVector<Pose> &poses,
                        const AlignedVector<Matrix3> &joints,
                        const Factor::Evaluation &base_eval,
                        Factor::Linearization &base_lin) const override;

  int getPose() const { return mPose; }

  const Vector3 &point() const { return mPoint; }

  const Vector2 &measurement() const { return mMeasurement; }

protected:
  int mPose;

  Vector3 mPoint;
  Vector2 mMeasurement;

  Scalar mCon;
  Scalar mSqrtCon;
  Scalar mSigmaCon;

  Scalar mSigma;
  Scalar mGMThreshold;
  Scalar mMaxGMScale;
};
} // namespace Initializer
} // namespace scope
