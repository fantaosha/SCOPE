#include <glog/logging.h>

#include <scope/initializer/factor/POFFactor.h>

namespace scope {
namespace Initializer {
POFFactor::POFFactor(int pose, const Vector3 &s, const Scalar &sigma,
                     const Scalar &eps, const Vector3 &measurement,
                     const Scalar &confidence, const std::string &name,
                     int index)
    : Factor(name, index), mPose(pose), mMeasurement(measurement),
      mSigma(sigma), mCon(confidence), mSqrtCon(std::sqrt(confidence)),
      mSigmaCon(confidence * sigma), mMinAlpha(eps), mS(s) {
  mGMThreshold = (1 - eps * eps) / (3 + eps * eps) * sigma;
  mMeasurement.stableNormalize();
  mS.stableNormalize();
}

int POFFactor::evaluate(const AlignedVector<Pose> &poses,
                        const AlignedVector<Matrix3> &joints,
                        Factor::Evaluation &base_eval) const {
  auto &eval = dynamic_cast<Evaluation &>(base_eval);
  eval.clear();

  const auto &i = mPose;

  assert(i >= 0 && i < poses.size());

  if (i >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  auto &pose = poses[i];

  eval.error.resize(1);

  eval.d.noalias() = pose.R * mS;
  eval.error[0] = 1 - mMeasurement.dot(eval.d);
  eval.squaredErrorNorm = eval.error.squaredNorm();

#if 0
  eval.f = mCon * eval.squaredErrorNorm;
#else
  eval.f = mSigmaCon * eval.squaredErrorNorm / (mSigma + eval.squaredErrorNorm);
#endif

  eval.status = Status::VALID;

  return 0;
}

int POFFactor::linearize(const AlignedVector<Pose> &poses,
                         const AlignedVector<Matrix3> &joints,
                         const Factor::Evaluation &base_eval,
                         Factor::Linearization &base_lin) const {
  auto &eval = dynamic_cast<const Evaluation &>(base_eval);
  auto &lin = dynamic_cast<Linearization &>(base_lin);

  lin.clear();

  const auto &i = mPose;

  assert(i >= 0 && i < poses.size());

  if (i >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  assert(eval.status == Status::VALID);

  if (eval.status != Status::VALID) {
    LOG(ERROR) << "The evaluation must be valid." << std::endl;

    exit(-1);
  }

  lin.jacobian.setZero();

  Eigen::Map<Vector3> JR(lin.jacobian.data());

  JR.noalias() = mMeasurement.cross(eval.d);

#if 0
  JR *= mSqrtCon;
  lin.scaledError = mSqrtCon * eval.error[0];
#else
  lin.sqrtDrho = mSigma / (mSigma + eval.squaredErrorNorm);
  lin.alpha = eval.squaredErrorNorm > mGMThreshold
                  ? mMinAlpha
                  : std::sqrt(1 - 4 * eval.squaredErrorNorm /
                                      (mSigma + eval.squaredErrorNorm));

  lin.sqrtDrho *= mSqrtCon;
  lin.scaledError = lin.sqrtDrho * eval.error[0] / lin.alpha;

  JR *= lin.sqrtDrho * lin.alpha;
#endif

  lin.status = Status::VALID;

  return 0;
}
} // namespace Initializer
} // namespace scope
