#include <glog/logging.h>

#include <scope/factor/JointDepthCameraFactor.h>

namespace scope {
JointDepthCameraFactor::JointDepthCameraFactor(int pose, const Scalar &sigma,
                                               const Scalar &eps,
                                               const Vector3 &measurement,
                                               const Scalar &confidence,
                                               const std::string &name,
                                               int index, bool active)
    : DepthCameraFactor({pose}, {}, {}, {}, sigma, eps, measurement, confidence,
                        name, index, active) {}

int JointDepthCameraFactor::evaluate(const AlignedVector<Pose> &poses,
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

int JointDepthCameraFactor::linearize(const AlignedVector<Pose> &poses,
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

int JointDepthCameraFactor::evaluate(const Pose &pose,
                                     const Vector3 &measurement,
                                     Evaluation &eval) const {
  eval.error = pose.t - measurement;
  eval.squaredErrorNorm = eval.error.squaredNorm();

#if not ROBUST_DEPTH
  eval.f = mCon * eval.squaredErrorNorm;
#else
  eval.f = mSigmaCon * eval.squaredErrorNorm / (mSigma + eval.squaredErrorNorm);
#endif

  return 0;
}

int JointDepthCameraFactor::linearize(const Pose &pose,
                                      const Vector3 &measurement,
                                      const Evaluation &eval,
                                      Linearization &lin) const {
  assert(lin.jacobians[0].size() == 1);

  auto &JPose = lin.jacobians[0][0];

  JPose.resize(3, 6);

  Eigen::Map<Matrix3> JR(JPose.data());
  Eigen::Map<Matrix3> Jt(JPose.data() + 9);

  const auto &t = pose.t;

  JR.setZero();

  JR(0, 1) = t[2];
  JR(0, 2) = -t[1];
  JR(1, 0) = -t[2];
  JR(1, 2) = t[0];
  JR(2, 0) = t[1];
  JR(2, 1) = -t[0];

  Jt.setIdentity();

  auto &gPose = lin.gPose;
  gPose.noalias() = JPose.transpose() * eval.error;

#if not ROBUST_DEPTH
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
}  // namespace scope
