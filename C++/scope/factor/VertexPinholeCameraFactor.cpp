#include <glog/logging.h>

#include <scope/factor/VertexPinholeCameraFactor.h>

namespace scope {
VertexPinholeCameraFactor::VertexPinholeCameraFactor(
    int pose, int vParam, const Matrix3X &vDirs, const Vector3 &v,
    const Scalar &sigma, const Scalar &eps, const Vector2 &measurement,
    const Scalar &confidence, const std::string &name, int index, bool active)
    : PinholeCameraFactor({pose}, {}, {}, {vParam}, sigma, eps, measurement,
                          confidence, name, index, active),
      mvDirs(vDirs), mv(v) {}

int VertexPinholeCameraFactor::evaluate(const AlignedVector<Pose> &poses,
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

int VertexPinholeCameraFactor::linearize(
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

int VertexPinholeCameraFactor::evaluate(const Pose &pose,
                                        const VectorX &vertexParam,
                                        const Vector2 &measurement,
                                        Evaluation &eval) const {
  eval.vertex = mv;
  eval.vertex.noalias() += mvDirs * vertexParam;

  eval.pCam = pose.t;
  eval.pCam.noalias() += pose.R * eval.vertex;

  eval.point2D = eval.pCam.head<2>() / eval.pCam[2];
  eval.error = eval.point2D - measurement;
  eval.squaredErrorNorm = eval.error.squaredNorm();

#if not ROBUST_PINHOLE
  eval.f = mCon * eval.squaredErrorNorm;
#else
  eval.f = mSigmaCon * eval.squaredErrorNorm / (mSigma + eval.squaredErrorNorm);
#endif

  return 0;
}

int VertexPinholeCameraFactor::linearize(const Pose &pose,
                                         const VectorX &vertexParam,
                                         const Vector2 &measurement,
                                         const Evaluation &eval,
                                         Linearization &lin) const {
  assert(lin.jacobians[0].size() == 1);
  assert(lin.jacobians[3].size() >= 1);

  auto &JPose = lin.jacobians[0][0];

  JPose.resize(2, 6);

  Eigen::Map<Matrix<2, 3>> JR(JPose.data());
  Eigen::Map<Matrix<2, 3>> Jt(JPose.data() + 6);

  Scalar zinv = 1.0 / eval.pCam[2];

  Jt(0, 0) = zinv;
  Jt(0, 1) = 0;
  Jt(0, 2) = -eval.point2D[0] * zinv;
  Jt(1, 0) = 0;
  Jt(1, 1) = zinv;
  Jt(1, 2) = -eval.point2D[1] * zinv;

  JR.row(0) = eval.pCam.cross(Jt.row(0));
  JR.row(1) = eval.pCam.cross(Jt.row(1));

  auto &JVertex = lin.jacobians[3][0];

  lin.D.noalias() = Jt * pose.R;
  JVertex.noalias() = lin.D * mvDirs;

  auto &gPose = lin.gPose;
  auto &gVertex = lin.gVertex;

  gPose.noalias() = JPose.transpose() * eval.error;
  gVertex.noalias() = JVertex.transpose() * eval.error;

#if not ROBUST_PINHOLE
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
} // namespace scope
