#include <glog/logging.h>

#include <scope/initializer/factor/DepthCameraFactor.h>

namespace scope {
namespace Initializer {
DepthCameraFactor::DepthCameraFactor(int pose, const Vector3 &point,
                                     const Scalar &sigma, const Scalar &eps,
                                     const Vector3 &measurement,
                                     const Scalar &confidence,
                                     const std::string &name, int index)
    : Factor(name, index),
      mPose(pose),
      mPoint(point),
      mMeasurement(measurement),
      mSigma(sigma),
      mCon(confidence),
      mSqrtCon(std::sqrt(confidence)),
      mSigmaCon(confidence * sigma) {
  assert(eps > 1e-5 && eps < 1.0);
  assert(sigma > 0);
  assert(confidence >= 0);

  mMaxGMScale = 1 - eps;
  mGMThreshold = (1 - eps * eps) / (3 + eps * eps) * sigma;
}

int DepthCameraFactor::evaluate(const AlignedVector<Pose> &poses,
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

  eval.point3D = pose.t;
  eval.point3D.noalias() += pose.R * mPoint;

  eval.error = eval.point3D - mMeasurement;
  eval.squaredErrorNorm = eval.error.squaredNorm();

#if 0
  eval.f = mCon * eval.squaredErrorNorm;
#else
  eval.f = mSigmaCon * eval.squaredErrorNorm / (mSigma + eval.squaredErrorNorm);
#endif

  eval.status = Status::VALID;

  return 0;
}

int DepthCameraFactor::linearize(const AlignedVector<Pose> &poses,
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

  auto &J = lin.jacobian;

  Eigen::Map<Matrix<3, 3>> JR(J.data());
  Eigen::Map<Matrix<3, 3>> Jt(J.data() + 9);

  JR.setZero();

  JR(0, 1) = eval.point3D[2];
  JR(0, 2) = -eval.point3D[1];
  JR(1, 0) = -eval.point3D[2];
  JR(1, 2) = eval.point3D[0];
  JR(2, 0) = eval.point3D[1];
  JR(2, 1) = -eval.point3D[0];

  Jt.setIdentity();
  Jt.setIdentity();

  auto &g = lin.g;

  g.noalias() = J.transpose() * eval.error;

#if 0
  J *= mSqrtCon;
  g *= mCon;
#else
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

  J.noalias() -= lin.scaledError * g.transpose();

  J *= lin.sqrtDrho;
  g *= lin.Drho;
#endif

  lin.status = Status::VALID;

  return 0;
}
}  // namespace Initializer
}  // namespace scope
