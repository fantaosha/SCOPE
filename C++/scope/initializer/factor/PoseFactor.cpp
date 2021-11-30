#include <glog/logging.h>

#include <scope/initializer/factor/PoseFactor.h>
#include <scope/math/SO3.h>

namespace scope {
namespace Initializer {
PoseFactor::PoseFactor(int pose, const Matrix6 &sqrtCov, const Pose &mean,
                       const std::string &name, int index)
    : Factor(name, index), mPose(pose), mSqrtCov(sqrtCov), mMean(mean) {
  assert((mean.R.col(0).cross(mean.R.col(1)) - mean.R.col(2)).norm() < 1e-5 &&
         "mean.R must be a rotational matrix");
}

int PoseFactor::evaluate(const AlignedVector<Pose> &poses,
                         const AlignedVector<Matrix3> &joints,
                         Factor::Evaluation &base_eval) const {
  auto &eval = dynamic_cast<Evaluation &>(base_eval);
  eval.clear();

  const auto &i = mPose;

  assert(i >= 0 && i < poses.size());

  if (i >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  auto &pose = poses[i];

  eval.errorR.noalias() = mMean.R.transpose() * pose.R;

  eval.error.resize(6);

  Eigen::Map<Vector3> omega(eval.xi.data());
  math::SO3::log(eval.errorR, omega);
  eval.xi.tail<3>() = pose.t - mMean.t;
  eval.error.noalias() = mSqrtCov * eval.xi;

  eval.f = eval.error.squaredNorm();

  eval.status = Status::VALID;

  return 0;
}

int PoseFactor::linearize(const AlignedVector<Pose> &poses,
                          const AlignedVector<Matrix3> &joints,
                          const Factor::Evaluation &base_eval,
                          Factor::Linearization &base_lin) const {
  auto &eval = dynamic_cast<const Evaluation &>(base_eval);
  auto &lin = dynamic_cast<Linearization &>(base_lin);

  lin.clear();

  const auto &i = mPose;

  assert(i >= 0 && i < poses.size());

  if (i >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  assert(eval.status == Status::VALID);

  if (eval.status != Status::VALID) {
    LOG(ERROR) << "The evaluation must be valid." << std::endl;

    exit(-1);
  }

  auto &pose = poses[i];

  auto &J = lin.jacobian;

  math::SO3::dexpinv(eval.xi.head<3>(), lin.dexpinv);
  lin.dexpinvR.noalias() = lin.dexpinv * mMean.R.transpose();

  J.topRightCorner<3, 3>().setZero();

  const auto &t = pose.t;

  Eigen::Map<Matrix63> JR(J.data());
  Eigen::Map<Matrix63> Jt(J.data() + 18);

  Jt = mSqrtCov.rightCols<3>();

  JR.row(0).noalias() = t.cross(Jt.row(0));
  JR.row(1).noalias() = t.cross(Jt.row(1));
  JR.row(2).noalias() = t.cross(Jt.row(2));
  JR.row(3).noalias() = t.cross(Jt.row(3));
  JR.row(4).noalias() = t.cross(Jt.row(4));
  JR.row(5).noalias() = t.cross(Jt.row(5));
  JR.noalias() += mSqrtCov.leftCols<3>() * lin.dexpinvR;

  lin.status = Status::VALID;

  return 0;
}
} // namespace Initializer
} // namespace scope
