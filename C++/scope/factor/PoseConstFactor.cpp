#include <glog/logging.h>

#include <scope/factor/PoseConstFactor.h>
#include <scope/math/SO3.h>

namespace scope {
PoseConstFactor::PoseConstFactor(int pose, const Vector6 &weight,
                                 const Pose &mean, const Vector6 &lbnd,
                                 const Vector6 &ubnd, const std::string &name,
                                 int index, bool active)
    : Factor({pose}, {}, {}, {}, name, index, active),
      mLowerBnd(lbnd),
      mUpperBnd(ubnd),
      mWeight(weight),
      mMean(mean) {
  assert((mean.R.col(0).cross(mean.R.col(1)) - mean.R.col(2)).norm() < 1e-5 &&
         "mean.R must be a rotational matrix");
  assert((lbnd.cwiseMin(ubnd) - lbnd).norm() < 1e-7);
  assert((ubnd.cwiseMax(lbnd) - ubnd).norm() < 1e-7);
}

int PoseConstFactor::evaluate(const AlignedVector<Pose> &poses,
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

  eval.errorR.noalias() = mMean.R.transpose() * pose.R;

  eval.error.resize(6);

  Eigen::Map<Vector3> omega(eval.xi.data());
  math::SO3::log(eval.errorR, omega);
  eval.xi.tail<3>() = pose.t - mMean.t;

  eval.errorL = eval.xi - mLowerBnd;
  eval.errorL = eval.errorL.cwiseMin(mBnd);

  eval.errorU = mUpperBnd - eval.xi;
  eval.errorU = eval.errorU.cwiseMin(mBnd);

  eval.errorLInv = mScale * eval.errorL.cwiseInverse();
  eval.errorUInv = mScale * eval.errorU.cwiseInverse();

  eval.expL = eval.errorLInv.array().exp();
  eval.expU = eval.errorUInv.array().exp();

  eval.error = eval.errorL.cwiseProduct(eval.expL);
  eval.error += eval.errorU.cwiseProduct(eval.expU);

  eval.error.noalias() = mWeight.asDiagonal() * eval.error;

  eval.f = eval.error.squaredNorm();

  eval.status = Status::VALID;

  return 0;
}

int PoseConstFactor::linearize(const AlignedVector<Pose> &poses,
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

  auto &J = lin.jacobians[0][0];
  J.resize(6, 6);

  math::SO3::dexpinv(eval.xi.head<3>(), lin.dexpinv);

  lin.Derror = eval.expL;
  lin.Derror -= eval.expL.cwiseProduct(eval.errorLInv);
  lin.Derror -= eval.expU;
  lin.Derror += eval.expU.cwiseProduct(eval.errorUInv);

  J.topRightCorner<3, 3>().setZero();

  const auto &t = poses[i].t;

  Eigen::Map<Matrix63> JR(J.data());
  Eigen::Map<Matrix63> Jt(J.data() + 18);

  JR.topRows<3>().noalias() = lin.dexpinv * mMean.R.transpose();
  JR(3, 0) = 0;
  JR(3, 1) = t[2];
  JR(3, 2) = -t[1];
  JR(4, 0) = -t[2];
  JR(4, 1) = 0;
  JR(4, 2) = t[0];
  JR(5, 0) = t[1];
  JR(5, 1) = -t[0];
  JR(5, 2) = 0;
  JR.noalias() = lin.Derror.asDiagonal() * JR;
  JR.noalias() = mWeight.asDiagonal() * JR;

  Jt.topRows<3>().setZero();
  Jt.bottomRows<3>() =
      mWeight.tail<3>().cwiseProduct(lin.Derror.tail<3>()).asDiagonal();

  lin.status = Status::VALID;

  return 0;
}
}  // namespace scope
