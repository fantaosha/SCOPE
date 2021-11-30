#pragma once

#define ROBUST_DEPTH 1

#include <scope/factor/Factor.h>

namespace scope {
class DepthCameraFactor : public Factor {
 public:
  struct Evaluation : public Factor::Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Scalar squaredErrorNorm;
  };

  struct Linearization : public Factor::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Scalar Drho;
    Scalar sqrtDrho;
    Scalar alpha;

    Vector3 scaledError;

    Vector6 gPose;
  };

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_FACTOR_EVALUATION_NEW
  SCOAT_FACTOR_LINEARIZATION_NEW

  DepthCameraFactor(const std::vector<int> &poses,
                    const std::vector<int> &shapes,
                    const std::vector<int> &joints,
                    const std::vector<int> &params, const Scalar &sigma,
                    const Scalar &eps, const Vector3 &measurement,
                    const Scalar &confidence, const std::string &name = "",
                    int index = -1, bool active = true);

protected:
  virtual int preLinearizeRobustKernel(const Evaluation &eval,
                                       Linearization &lin) const;

 protected:
  Vector3 mMeasurement;

  Scalar mCon;
  Scalar mSqrtCon;
  Scalar mSigmaCon;

  Scalar mSigma;
  Scalar mGMThreshold;
  Scalar mMaxGMScale;
};
}  // namespace scope
