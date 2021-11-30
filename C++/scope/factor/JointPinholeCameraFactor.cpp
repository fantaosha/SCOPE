#include <glog/logging.h>

#include <scope/factor/JointPinholeCameraFactor.h>

namespace scope {
JointPinholeCameraFactor::JointPinholeCameraFactor(
    int pose, const Scalar &sigma, const Scalar &eps,
    const Vector2 &measurement, const Scalar &confidence,
    const std::string &name, int index, bool active)
    : PinholeCameraFactor({pose}, {}, {}, {}, sigma, eps, measurement,
                          confidence, name, index, active) {}

int JointPinholeCameraFactor::evaluate(const AlignedVector<Pose> &poses,
                                       const AlignedVector<VectorX> &shapes,
                                       const AlignedVector<Matrix3> &joints,
                                       const AlignedVector<VectorX> &params,
                                       Factor::Evaluation &base_eval) const {
  auto &eval = dynamic_cast<Evaluation &>(base_eval);
  eval.clear();

  const auto &i = mvPoses[0];
  assert(i >= 0 && i < poses.size());

  if (i < 0 || i >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  evaluate(poses[i], mMeasurement, eval);

  eval.status = Status::VALID;

  return 0;
}

int JointPinholeCameraFactor::linearize(const AlignedVector<Pose> &poses,
                                        const AlignedVector<VectorX> &shapes,
                                        const AlignedVector<Matrix3> &joints,
                                        const AlignedVector<VectorX> &params,
                                        const Factor::Evaluation &base_eval,
                                        Factor::Linearization &base_lin) const {
  auto &eval = dynamic_cast<const Evaluation &>(base_eval);
  auto &lin = dynamic_cast<Linearization &>(base_lin);

  lin.clear();

  assert(eval.status == Status::VALID);

  if (eval.status != Status::VALID) {
    LOG(ERROR) << "The evaluation must be valid." << std::endl;

    exit(-1);
  }

  const auto &i = mvPoses[0];
  assert(i >= 0 && i < poses.size());

  if (i < 0 || i >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  lin.jacobians[0].resize(1);

  linearize(poses[i], mMeasurement, eval, lin);

  lin.status = Status::VALID;

  return 0;
}

int JointPinholeCameraFactor::evaluate(const Pose &pose,
                                       const Vector2 &measurement,
                                       Evaluation &eval) const {
  const auto &t = pose.t;

  eval.point2D = t.head<2>() / t[2];
  eval.error = eval.point2D - measurement;
  eval.squaredErrorNorm = eval.error.squaredNorm();

#if not ROBUST_PINHOLE
  eval.f = mCon * eval.squaredErrorNorm;
#else
  eval.f = mSigmaCon * eval.squaredErrorNorm / (mSigma + eval.squaredErrorNorm);
#endif

  return 0;
}

int JointPinholeCameraFactor::linearize(const Pose &pose,
                                        const Vector2 &measurement,
                                        const Evaluation &eval,
                                        Linearization &lin) const {
  assert(lin.jacobians[0].size() == 1);

  auto &JPose = lin.jacobians[0][0];

  JPose.resize(2, 6);

  Eigen::Map<Matrix<2, 3>> JR(JPose.data());
  Eigen::Map<Matrix<2, 3>> Jt(JPose.data() + 6);

  Scalar zinv = (Scalar)1.0 / pose.t[2];

  Jt(0, 0) = zinv;
  Jt(0, 1) = 0;
  Jt(0, 2) = -eval.point2D[0] * zinv;
  Jt(1, 0) = 0;
  Jt(1, 1) = zinv;
  Jt(1, 2) = -eval.point2D[1] * zinv;

  const auto &t = pose.t;

  JR.row(0) = t.cross(Jt.row(0));
  JR.row(1) = t.cross(Jt.row(1));

  auto &gPose = lin.gPose;
  gPose.noalias() = JPose.transpose() * eval.error;

#if not ROBUST_PINHOLE
  JPose *= mSqrtCon;
  gPose *= mCon;
#else
  preLinearizeRobustKernel(eval, lin);

  JPose.noalias() -= lin.scaledError * gPose.transpose();

  JPose *= lin.sqrtDrho;
  gPose *= lin.Drho;
#endif

  return 0;
}
} // namespace scope
