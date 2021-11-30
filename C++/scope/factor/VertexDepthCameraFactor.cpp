#include <glog/logging.h>

#include <scope/factor/VertexDepthCameraFactor.h>

namespace scope {
VertexDepthCameraFactor::VertexDepthCameraFactor(
    int pose, int vParam, const Matrix3X &vDirs, const Vector3 &v,
    const Scalar &sigma, const Scalar &eps, const Vector3 &measurement,
    const Scalar &confidence, const std::string &name, int index, bool active)
    : DepthCameraFactor({pose}, {}, {}, {vParam}, sigma, eps, measurement,
                        confidence, name, index, active),
      mvDirs(vDirs),
      mv(v) {}

int VertexDepthCameraFactor::evaluate(const AlignedVector<Pose> &poses,
                                      const AlignedVector<VectorX> &shapes,
                                      const AlignedVector<Matrix3> &joints,
                                      const AlignedVector<VectorX> &params,
                                      Factor::Evaluation &base_eval) const {
  auto &eval = dynamic_cast<Evaluation &>(base_eval);
  eval.clear();

  const auto &pose = mvPoses[0];
  assert(pose >= 0 && pose < poses.size());

  if (pose < 0 || pose >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &vertexParam = mvParams[0];
  assert(vertexParam >= 0 && vertexParam < params.size());

  if (vertexParam < 0 || vertexParam > params.size()) {
    LOG(ERROR) << "The parameter must be valid." << std::endl;

    exit(-1);
  }

  evaluate(poses[pose], params[vertexParam], mMeasurement, eval);

  eval.status = Status::VALID;

  return 0;
}

int VertexDepthCameraFactor::linearize(const AlignedVector<Pose> &poses,
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

  const auto &pose = mvPoses[0];
  assert(pose >= 0 && pose < poses.size());

  if (pose < 0 || pose >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &vertexParam = mvParams[0];
  assert(vertexParam >= 0 && vertexParam < params.size());

  if (vertexParam < 0 || vertexParam > params.size()) {
    LOG(ERROR) << "The parameter must be valid." << std::endl;

    exit(-1);
  }

  lin.jacobians[0].resize(1);
  lin.jacobians[3].resize(1);

  linearize(poses[pose], params[vertexParam], mMeasurement, eval, lin);

  lin.status = Status::VALID;

  return 0;
}

int VertexDepthCameraFactor::evaluate(const Pose &pose,
                                      const VectorX &vertexParam,
                                      const Vector3 &measurement,
                                      Evaluation &eval) const {
  eval.vertex = mv;
  eval.vertex.noalias() += mvDirs * vertexParam;

  eval.point3D = pose.t;
  eval.point3D.noalias() += pose.R * eval.vertex;

  eval.error = eval.point3D - measurement;
  eval.squaredErrorNorm = eval.error.squaredNorm();

#if not ROBUST_DEPTH
  eval.f = mCon * eval.squaredErrorNorm;
#else
  eval.f = mSigmaCon * eval.squaredErrorNorm / (mSigma + eval.squaredErrorNorm);
#endif

  return 0;
}

int VertexDepthCameraFactor::linearize(const Pose &pose,
                                       const VectorX &vertexParam,
                                       const Vector3 &measurement,
                                       const Evaluation &eval,
                                       Linearization &lin) const {
  assert(lin.jacobians[0].size() == 1);
  assert(lin.jacobians[3].size() >= 1);

  auto &JPose = lin.jacobians[0][0];

  JPose.resize(3, 6);

  Eigen::Map<Matrix<3, 3>> JR(JPose.data());
  Eigen::Map<Matrix<3, 3>> Jt(JPose.data() + 9);

  JR.setZero();

  JR(0, 1) = eval.point3D[2];
  JR(0, 2) = -eval.point3D[1];
  JR(1, 0) = -eval.point3D[2];
  JR(1, 2) = eval.point3D[0];
  JR(2, 0) = eval.point3D[1];
  JR(2, 1) = -eval.point3D[0];

  Jt.setIdentity();

  auto &JVertex = lin.jacobians[3][0];

  JVertex.noalias() = pose.R * mvDirs;

  auto &gPose = lin.gPose;
  auto &gVertex = lin.gVertex;

  gPose.noalias() = JPose.transpose() * eval.error;
  gVertex.noalias() = JVertex.transpose() * eval.error;

#if not ROBUST_DEPTH
  JPose *= mSqrtCon;
  JVertex *= mSqrtCon;

  gPose *= mCon;
  gVertex *= mCon;
#else
  preLinearizeRobustKernel(eval, lin);

  JPose.noalias() -= lin.scaledError * gPose.transpose();
  JVertex.noalias() -= lin.scaledError * gVertex.transpose();

  JPose *= lin.sqrtDrho;
  JVertex *= lin.sqrtDrho;

  gPose *= lin.Drho;
  gVertex *= lin.Drho;
#endif

  return 0;
}
}  // namespace scope
