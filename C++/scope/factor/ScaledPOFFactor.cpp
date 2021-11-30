#include <glog/logging.h>

#include <scope/factor/ScaledPOFFactor.h>

namespace scope {
ScaledPOFFactor::ScaledPOFFactor(
    int pose, int vParam, const std::array<Matrix3X, 2> &VDirs,
    const std::array<Vector3, 2> &V, const Scalar &sigma, const Scalar &eps,
    const Vector3 &measurement, const Scalar &confidence,
    const std::string &name, int index, bool active)
    : POFFactor({pose}, {}, {}, {vParam}, sigma, eps, measurement, confidence,
                name, index, active) {
  mVDirs = VDirs[1] - VDirs[0];
  mV = V[1] - V[0];
}

int ScaledPOFFactor::evaluate(const AlignedVector<Pose> &poses,
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

  const auto &vertexParam = mvParams[0];
  assert(vertexParam >= 0 && vertexParam < params.size());

  if (vertexParam < 0 || vertexParam > params.size()) {
    LOG(ERROR) << "The parameter must be valid." << std::endl;

    exit(-1);
  }

  evaluate(poses[i], params[vertexParam], mMeasurement, eval);

  eval.status = Status::VALID;

  return 0;
}

int ScaledPOFFactor::linearize(const AlignedVector<Pose> &poses,
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

  const auto &vertexParam = mvParams[0];
  assert(vertexParam >= 0 && vertexParam < params.size());

  if (vertexParam < 0 || vertexParam > params.size()) {
    LOG(ERROR) << "The parameter must be valid." << std::endl;

    exit(-1);
  }

  lin.jacobians[0].resize(1);
  lin.jacobians[3].resize(1);

  linearize(poses[i], params[vertexParam], mMeasurement, eval, lin);

  lin.status = Status::VALID;

  return 0;
}

int ScaledPOFFactor::evaluate(const Pose &pose, const VectorX &vertexParam,
                              const Vector3 &measurement,
                              Evaluation &eval) const {
  eval.error.resize(3);

  eval.vertex = mV;
  eval.vertex.noalias() += mVDirs * vertexParam;

  eval.D.noalias() = pose.R * eval.vertex;

  eval.DNorm = eval.D.stableNorm();

  eval.d.setZero();

  if (eval.DNorm > 1e-10) {
    eval.d = eval.D / eval.DNorm;
  }

  eval.error = eval.d - mMeasurement;
  eval.squaredErrorNorm = eval.error.squaredNorm();

#if not ROBUST_POF
  eval.f = mCon * eval.squaredErrorNorm;
#else
  eval.f = mSigmaCon * eval.squaredErrorNorm / (mSigma + eval.squaredErrorNorm);
#endif

  return 0;
}

int ScaledPOFFactor::linearize(const Pose &pose, const VectorX &vertexParam,
                               const Vector3 &measurement,
                               const Evaluation &eval,
                               Linearization &lin) const {
  assert(lin.jacobians[0].size() == 1);
  assert(lin.jacobians[3].size() == 1);

  lin.DF.noalias() = eval.d * eval.d.transpose();
  lin.DF.diagonal().array() -= 1;
  lin.DF /= -eval.DNorm;

  lin.jacobians[0][0].setZero(3, 6);
  Eigen::Map<Matrix3> JR(lin.jacobians[0][0].data());

  JR.row(0).noalias() = eval.D.cross(lin.DF.row(0));
  JR.row(1).noalias() = eval.D.cross(lin.DF.row(1));
  JR.row(2).noalias() = eval.D.cross(lin.DF.row(2));

  auto &JVertex = lin.jacobians[3][0];

  lin.DFR.noalias() = lin.DF * pose.R;
  JVertex.noalias() = lin.DFR * mVDirs;

  lin.gPose.setZero();

  Eigen::Map<Vector3> gR(lin.gPose.data());
  auto& gVertex = lin.gVertex;

  gR.noalias() = JR.transpose() * eval.error;
  gVertex.noalias() = JVertex.transpose() * eval.error;

#if not ROBUST_POF
  JR *= mSqrtCon;
  JVertex *= mSqrtCon;

  gR *= mCon;
  gVertex *= mCon;
#else
  preLinearizeRobustKernel(eval, lin);

  JR.noalias() -= lin.scaledError * gR.transpose();
  JVertex.noalias() -= lin.scaledError * gVertex.transpose();

  JR *= lin.sqrtDrho;
  JVertex *= lin.sqrtDrho;

  gR *= lin.Drho;
  gVertex *= lin.Drho;
#endif

  return 0;
}
}  // namespace scope
