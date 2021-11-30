#include <glog/logging.h>

#include <scope/factor/FullJointExtraPinholeCameraFactor.h>

namespace scope {
FullJointExtraPinholeCameraFactor::FullJointExtraPinholeCameraFactor(
    int pose, int camParam, const Scalar &sigma, const Scalar &eps,
    const Pose &extraCamPose, const Vector2 &measurement,
    const Scalar &confidence, const std::string &name, int index, bool active)
    : JointExtraPinholeCameraFactor(pose, sigma, eps, extraCamPose, measurement,
                                    confidence, name, index, active) {
  mvParams = {camParam};
}

int FullJointExtraPinholeCameraFactor::evaluate(
    const AlignedVector<Pose> &poses, const AlignedVector<VectorX> &shapes,
    const AlignedVector<Matrix3> &joints, const AlignedVector<VectorX> &params,
    Factor::Evaluation &base_eval) const {
  auto &eval = dynamic_cast<Evaluation &>(base_eval);
  eval.clear();

  const auto &pose = mvPoses[0];
  assert(pose >= 0 && pose < poses.size());

  if (pose < 0 || pose >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &camParam = mvParams[0];
  assert(camParam >= 0 && camParam < params.size());

  if (camParam < 0 || camParam > params.size()) {
    LOG(ERROR) << "The parameter must be valid." << std::endl;

    exit(-1);
  }

  evaluate(poses[pose], params[camParam], mMeasurement, eval);

  eval.status = Status::VALID;

  return 0;
}

int FullJointExtraPinholeCameraFactor::linearize(
    const AlignedVector<Pose> &poses, const AlignedVector<VectorX> &shapes,
    const AlignedVector<Matrix3> &joints, const AlignedVector<VectorX> &params,
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

  const auto &pose = mvPoses[0];
  assert(pose >= 0 && pose < poses.size());

  if (pose < 0 || pose >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &camParam = mvParams[0];
  assert(camParam >= 0 && camParam < params.size());

  if (camParam < 0 || camParam > params.size()) {
    LOG(ERROR) << "The parameter must be valid." << std::endl;

    exit(-1);
  }

  lin.jacobians[0].resize(1);
  lin.jacobians[3].resize(1);

  linearize(poses[pose], params[camParam], mMeasurement, eval, lin);

  lin.status = Status::VALID;

  return 0;
}

int FullJointExtraPinholeCameraFactor::evaluate(const Pose &pose,
                                                const VectorX &camParam,
                                                const Vector2 &measurement,
                                                Evaluation &eval) const {
  assert(camParam.size() == 4);

  eval.camMeasurement = camParam.tail<2>();
  eval.camMeasurement[0] += camParam[0] * measurement[0];
  eval.camMeasurement[1] += camParam[1] * measurement[1];

  JointExtraPinholeCameraFactor::evaluate(pose, eval.camMeasurement, eval);

  return 0;
}

int FullJointExtraPinholeCameraFactor::linearize(const Pose &pose,
                                                 const VectorX &camParam,
                                                 const Vector2 &measurement,
                                                 const Evaluation &eval,
                                                 Linearization &lin) const {
  assert(camParam.size() == 4);

  assert(lin.jacobians[0].size() == 1);
  assert(lin.jacobians[3].size() == 1);

  JointExtraPinholeCameraFactor::linearize(pose, eval.camMeasurement, eval,
                                           lin);

  auto &JCam = lin.jacobians[3][0];

  JCam.resize(2, 4);

  JCam(0, 0) = -measurement[0];
  JCam(0, 1) = 0;
  JCam(0, 2) = -1;
  JCam(0, 3) = 0;

  JCam(1, 0) = 0;
  JCam(1, 1) = -measurement[1];
  JCam(1, 2) = 0;
  JCam(1, 3) = -1;

  auto &gCam = lin.gCam;
  gCam.noalias() = JCam.transpose() * eval.error;

#if not ROBUST_PINHOLE
  JCam *= mSqrtCon;
  gCam *= mCon;
#else
  JCam.noalias() -= lin.scaledError * gCam.transpose();

  JCam *= lin.sqrtDrho;
  gCam *= lin.Drho;
#endif

  return 0;
}
} // namespace scope
