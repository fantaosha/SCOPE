#include <glog/logging.h>

#include <scope/initializer/factor/PinholeCameraFactor.h>

namespace scope {
namespace Initializer {
PinholeCameraFactor::PinholeCameraFactor(int pose, const Vector3 &point,
                                         const Scalar &sigma, const Scalar &eps,
                                         const Vector2 &measurement,
                                         const Scalar &confidence,
                                         const std::string &name, int index)
    : Factor(name, index), mPose(pose), mPoint(point),
      mMeasurement(measurement), mSigma(sigma), mCon(confidence),
      mSqrtCon(std::sqrt(confidence)), mSigmaCon(confidence * sigma) {
  assert(eps > 1e-5 && eps < 1.0);
  assert(sigma > 0);
  assert(confidence >= 0);

  mMaxGMScale = 1 - eps;
  mGMThreshold = (1 - eps * eps) / (3 + eps * eps) * sigma;
}

int PinholeCameraFactor::evaluate(const AlignedVector<Pose> &poses,
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

  eval.pCam = pose.t;
  eval.pCam.noalias() += pose.R * mPoint;

  eval.point2D = eval.pCam.head<2>() / eval.pCam[2];
  eval.error = eval.point2D - mMeasurement;
  eval.squaredErrorNorm = eval.error.squaredNorm();

#if 0
  eval.f = mCon * eval.squaredErrorNorm;
#else
  eval.f = mSigmaCon * eval.squaredErrorNorm / (mSigma + eval.squaredErrorNorm);
#endif

  eval.status = Status::VALID;

  return 0;
}

int PinholeCameraFactor::linearize(const AlignedVector<Pose> &poses,
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

  J.resize(2, 6);

  Eigen::Map<Matrix<2, 3>> JR(J.data());
  Eigen::Map<Matrix<2, 3>> Jt(J.data() + 6);

  Scalar zinv = 1.0 / eval.pCam[2];

  Jt(0, 0) = zinv;
  Jt(0, 1) = 0;
  Jt(0, 2) = -eval.point2D[0] * zinv;
  Jt(1, 0) = 0;
  Jt(1, 1) = zinv;
  Jt(1, 2) = -eval.point2D[1] * zinv;

  JR.row(0) = eval.pCam.cross(Jt.row(0));
  JR.row(1) = eval.pCam.cross(Jt.row(1));

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
} // namespace Initializer
} // namespace scope
