#include <glog/logging.h>

#include <scope/factor/UnitPOFFactor.h>

namespace scope {
UnitPOFFactor::UnitPOFFactor(int pose, const Vector3 &s, const Scalar &sigma,
                             const Scalar &eps, const Vector3 &measurement,
                             const Scalar &confidence, const std::string &name,
                             int index, bool active)
    : POFFactor({pose}, {}, {}, {}, sigma, eps, measurement, confidence, name,
                index, active),
      mS(s) {
  mS.stableNormalize();
}

int UnitPOFFactor::evaluate(const AlignedVector<Pose> &poses,
                            const AlignedVector<VectorX> &shapes,
                            const AlignedVector<Matrix3> &joints,
                            const AlignedVector<VectorX> &params,
                            Factor::Evaluation &base_eval) const {
  auto &eval = dynamic_cast<Evaluation &>(base_eval);
  eval.clear();

  const auto &i = mvPoses[0];
  assert(i >= 0 && i < poses.size());

  if (i >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  auto &pose = poses[i];
  eval.error.resize(3);

  eval.d.noalias() = pose.R * mS;
  eval.error = eval.d - mMeasurement;
  eval.squaredErrorNorm = eval.error.squaredNorm();

#if not ROBUST_POF
  eval.f = mCon * eval.squaredErrorNorm;
#else
  eval.f = mSigmaCon * eval.squaredErrorNorm / (mSigma + eval.squaredErrorNorm);
#endif

  eval.status = Status::VALID;

  return 0;
}

int UnitPOFFactor::linearize(const AlignedVector<Pose> &poses,
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
    LOG(ERROR) << "The joint must be valid." << std::endl;

    exit(-1);
  }

  lin.jacobians[0].resize(1);
  lin.jacobians[0][0].setZero(3, 6);

  Eigen::Map<Matrix3> JR(lin.jacobians[0][0].data());

  JR(0, 1) = eval.d[2];
  JR(0, 2) = -eval.d[1];
  JR(1, 0) = -eval.d[2];
  JR(1, 2) = eval.d[0];
  JR(2, 0) = eval.d[1];
  JR(2, 1) = -eval.d[0];

  lin.gPose.setZero();

  Eigen::Map<Vector3> gR(lin.gPose.data());

  gR.noalias() = JR.transpose() * eval.error;

#if not ROBUST_POF
  JR *= mSqrtCon;
  gR *= mCon;
#else
  preLinearizeRobustKernel(eval, lin);

  JR.noalias() -= lin.scaledError * gR.transpose();

  JR *= lin.sqrtDrho;
  gR *= lin.Drho;
#endif

  lin.status = Status::VALID;

  return 0;
}
}  // namespace scope
