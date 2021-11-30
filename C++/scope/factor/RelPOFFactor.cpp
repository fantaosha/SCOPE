#include <glog/logging.h>

#include <scope/factor/RelPOFFactor.h>

namespace scope {
RelPOFFactor::RelPOFFactor(int pose, int vParam,
                           const std::vector<int> &parents,
                           const std::array<Matrix3X, 2> &VDirs,
                           const std::array<Vector3, 2> &V, const Scalar &sigma,
                           const Scalar &eps, const Vector3 &measurement,
                           const Scalar &confidence, const std::string &name,
                           int index, bool active)
    : POFFactor({pose}, {}, {pose - 1}, {vParam}, sigma, eps, measurement,
                confidence, name, index, active) {
  assert(pose > 0 && pose < parents.size());
  assert(VDirs[0].cols() == VDirs[1].cols());

  if (pose <= 0 || pose >= parents.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    mvPoses[0] = -1;
    mvParams[0] = -1;
  }

  if (VDirs[0].cols() != VDirs[1].cols()) {
    LOG(ERROR) << "Inconsistent shape directions." << std::endl;

    mvPoses[0] = -1;
    mvParams[0] = -1;
  }

  mParent = parents[pose];

  mVDirs.resize(6, VDirs[0].cols());
  mVDirs.topRows<3>() = VDirs[1];
  mVDirs.bottomRows<3>() = VDirs[0];

  mV.head<3>() = V[1];
  mV.tail<3>() = V[0];
}

int RelPOFFactor::evaluate(const AlignedVector<Pose> &poses,
                           const AlignedVector<VectorX> &shapes,
                           const AlignedVector<Matrix3> &joints,
                           const AlignedVector<VectorX> &params,
                           Factor::Evaluation &base_eval) const {
  auto &eval = dynamic_cast<Evaluation &>(base_eval);
  eval.clear();

  const auto &i = mvPoses[0];
  const auto &j = mParent;

  assert(i >= 1 && i < poses.size());
  assert(j >= 0 && j < poses.size());

  if (i < 1 || i >= poses.size() || j < 0 || j >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &vertexParam = mvParams[0];
  assert(vertexParam >= 0 && vertexParam < params.size());

  if (vertexParam < 0 || vertexParam > params.size()) {
    LOG(ERROR) << "The parameter must be valid." << std::endl;

    exit(-1);
  }

  evaluate(poses[i], poses[j], params[vertexParam], mMeasurement, eval);

  eval.status = Status::VALID;

  return 0;
}

int RelPOFFactor::linearize(const AlignedVector<Pose> &poses,
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
  const auto &j = mParent;

  assert(i >= 1 && i < poses.size());
  assert(j >= 0 && j < poses.size());

  if (i < 1 || i >= poses.size() || j < 0 || j >= poses.size()) {
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
  lin.jacobians[2].resize(1);
  lin.jacobians[3].resize(1);

  linearize(poses[i], poses[j], params[vertexParam], mMeasurement, eval, lin);

  lin.status = Status::VALID;

  return 0;
}

int RelPOFFactor::evaluate(const Pose &pose, const Pose &parent,
                           const VectorX &vertexParam,
                           const Vector3 &measurement, Evaluation &eval) const {
  eval.error.resize(1);

  eval.vertex = mV;
  eval.vertex.noalias() += mVDirs * vertexParam;

  eval.D.noalias() = pose.R * eval.vertex.head<3>();
  eval.D.noalias() -= parent.R * eval.vertex.tail<3>();

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

int RelPOFFactor::linearize(const Pose &pose, const Pose &parent,
                            const VectorX &vertexParam,
                            const Vector3 &measurement, const Evaluation &eval,
                            Linearization &lin) const {
  assert(lin.jacobians[0].size() == 1);
  assert(lin.jacobians[2].size() == 1);
  assert(lin.jacobians[3].size() == 1);

  lin.DF.noalias() = eval.d * eval.d.transpose();
  lin.DF.diagonal().array() -= 1;
  lin.DF /= -eval.DNorm;

  lin.jacobians[0][0].setZero(3, 6);
  Eigen::Map<Matrix3> JR(lin.jacobians[0][0].data());

  JR.row(0).noalias() = eval.D.cross(lin.DF.col(0));
  JR.row(1).noalias() = eval.D.cross(lin.DF.col(1));
  JR.row(2).noalias() = eval.D.cross(lin.DF.col(2));

  lin.DFR.middleCols(0, 3).noalias() = lin.DF * pose.R;
  lin.DFR.middleCols(3, 3).noalias() = -lin.DF * parent.R;

  lin.jacobians[2][0].setZero(3, 3);
  Eigen::Map<Matrix3> JJoint(lin.jacobians[2][0].data());

  JJoint.row(0).noalias() =
      -eval.vertex.tail<3>().cross(lin.DFR.block<1, 3>(0, 3));
  JJoint.row(1).noalias() =
      -eval.vertex.tail<3>().cross(lin.DFR.block<1, 3>(1, 3));
  JJoint.row(2).noalias() =
      -eval.vertex.tail<3>().cross(lin.DFR.block<1, 3>(2, 3));

  auto &JVertex = lin.jacobians[3][0];
  JVertex.noalias() = lin.DFR * mVDirs;

  lin.gPose.setZero();

  Eigen::Map<Vector3> gR(lin.gPose.data());
  auto& gJoint = lin.gJoint;
  auto& gVertex = lin.gVertex;

  gR.noalias() = JR.transpose() * eval.error;
  gJoint.noalias() = JJoint.transpose() * eval.error;
  gVertex.noalias() = JVertex.transpose() * eval.error;

#if not ROBUST_POF
  JR *= mSqrtCon;
  JJoint *= mSqrtCon;
  JVertex *= mSqrtCon;

  gR *= mCon;
  gJoint *= mCon;
  gVertex *= mCon;
#else
  preLinearizeRobustKernel(eval, lin);

  JR.noalias() -= lin.scaledError * gR.transpose();
  JJoint.noalias() -= lin.scaledError * gJoint.transpose();
  JVertex.noalias() -= lin.scaledError * gVertex.transpose();

  JR *= lin.sqrtDrho;
  JJoint *= lin.sqrtDrho;
  JVertex *= lin.sqrtDrho;

  gR *= lin.Drho;
  gJoint *= lin.Drho;
  gVertex *= lin.Drho;
#endif

  return 0;
}
}  // namespace scope
