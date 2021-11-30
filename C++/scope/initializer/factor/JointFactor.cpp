#include <glog/logging.h>

#include <scope/initializer/factor/JointFactor.h>
#include <scope/math/SO3.h>

namespace scope {
namespace Initializer {
JointFactor::JointFactor(int joint, const Matrix3 &sqrtCov, const Matrix3 &mean,
                         const std::string &name, int index)
    : Factor(name, index), mJoint(joint), mSqrtCov(sqrtCov), mMean(mean) {
  assert((mMean.col(0).cross(mMean.col(1)) - mMean.col(2)).norm() < 1e-5 &&
         "mean.R must be a rotational matrix");
}

int JointFactor::evaluate(const AlignedVector<Pose> &poses,
                          const AlignedVector<Matrix3> &joints,
                          Factor::Evaluation &base_eval) const {
  auto &eval = dynamic_cast<Evaluation &>(base_eval);
  eval.clear();

  const auto &i = mJoint;

  assert(i >= 0 && i < joints.size());

  if (i >= joints.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

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
                           const AlignedVector<Matrix3> &joints,
                           const Factor::Evaluation &base_eval,
                           Factor::Linearization &base_lin) const {
  auto &eval = dynamic_cast<const Evaluation &>(base_eval);
  auto &lin = dynamic_cast<Linearization &>(base_lin);

  lin.clear();

  const auto &i = mJoint;

  assert(i >= 0 && i < joints.size());

  if (i >= joints.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  assert(eval.status == Status::VALID);

  if (eval.status != Status::VALID) {
    LOG(ERROR) << "The evaluation must be valid." << std::endl;

    exit(-1);
  }

  auto &J = lin.jacobian;

  math::SO3::dexpinv(eval.xi, lin.dexpinv);
  lin.dexpinvR.noalias() = lin.dexpinv * mMean.transpose();

  J.noalias() = mSqrtCov * lin.dexpinvR;

  lin.status = Status::VALID;

  return 0;
}
}  // namespace Initializer
}  // namespace scope
