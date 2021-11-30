#include <glog/logging.h>

#include <scope/factor/JointFactor.h>
#include <scope/math/SO3.h>

namespace scope {
JointFactor::JointFactor(int joint, const Matrix3 &sqrtCov, const Matrix3 &mean,
                         const std::string &name, int index, bool active)
    : Factor({}, {}, {joint}, {}, name, index, active),
      mSqrtCov(sqrtCov),
      mMean(mean) {
  assert((mean.col(0).cross(mean.col(1)) - mean.col(2)).norm() < 1e-5 &&
         "mean must be a rotational matrix");
}

int JointFactor::evaluate(const AlignedVector<Pose> &poses,
                          const AlignedVector<VectorX> &shapes,
                          const AlignedVector<Matrix3> &joints,
                          const AlignedVector<VectorX> &params,
                          Factor::Evaluation &base_eval) const {
  auto &eval = dynamic_cast<Evaluation &>(base_eval);
  eval.clear();

  const auto &i = mvJoints[0];
  assert(i >= 0 && i < joints.size());

  if (i >= joints.size()) {
    LOG(ERROR) << "The joint must be valid." << std::endl;

    exit(-1);
  }

  const auto &R = joints[i];

  eval.errorR.noalias() = mMean.transpose() * R;

  math::SO3::log(eval.errorR, eval.xi);
  eval.error.noalias() = mSqrtCov * eval.xi;

  eval.f = eval.error.squaredNorm();

  eval.status = Status::VALID;

  return 0;
}

int JointFactor::linearize(const AlignedVector<Pose> &poses,
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

  const auto &i = mvJoints[0];
  assert(i >= 0 && i < joints.size());

  if (i < 0 || i >= joints.size()) {
    LOG(ERROR) << "The joint must be valid." << std::endl;

    exit(-1);
  }

  lin.jacobians[2].resize(1);
  lin.jacobians[2][0].resize(3, 3);
  Eigen::Map<Matrix3> J(lin.jacobians[2][0].data());

  math::SO3::dexpinv(eval.xi, lin.dexpinv);
  lin.dexpinvR.noalias() = lin.dexpinv * mMean.transpose();

  J.noalias() = mSqrtCov * lin.dexpinvR;

  lin.status = Status::VALID;

  return 0;
}
}  // namespace scope
