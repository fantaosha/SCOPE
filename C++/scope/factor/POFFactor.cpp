#include <scope/factor/POFFactor.h>

namespace scope {

POFFactor::POFFactor(const std::vector<int> &poses,
                     const std::vector<int> &shapes,
                     const std::vector<int> &joints,
                     const std::vector<int> &params, const Scalar &sigma,
                     const Scalar &eps, const Vector3 &measurement,
                     const Scalar &confidence, const std::string &name,
                     int index, bool active)
    : Factor(poses, shapes, joints, params, name, index, active),
      mMeasurement(measurement),
      mSigma(sigma),
      mCon(confidence),
      mSqrtCon(std::sqrt(confidence)),
      mSigmaCon(confidence * sigma) {
  assert(eps > 1e-5 && eps <= 1.0);
  assert(sigma > 0);
  assert(confidence >= 0);

  mMaxGMScale = 1 - eps;
  mGMThreshold = (1 - eps * eps) / (3 + eps * eps) * sigma;

  mMeasurement.stableNormalize();
}

int POFFactor::preLinearizeRobustKernel(const Evaluation &eval,
                                        Linearization &lin) const {
  // scaled jacobians and graidents related with GM kernel
  lin.sqrtDrho = mSigma / (mSigma + eval.squaredErrorNorm);
  lin.Drho = lin.sqrtDrho * lin.sqrtDrho;
  lin.alpha = eval.squaredErrorNorm > mGMThreshold
                  ? mMaxGMScale
                  : 1 - std::sqrt(1 - 4 * eval.squaredErrorNorm /
                                          (mSigma + eval.squaredErrorNorm));

  if (lin.alpha > 1e-5) {
    lin.scaledError = lin.alpha * eval.error / eval.squaredErrorNorm;
  } else {
    lin.scaledError.setZero();
  }

  lin.sqrtDrho *= mSqrtCon;
  lin.Drho *= mCon;

  return 0;
}

}  // namespace scope
