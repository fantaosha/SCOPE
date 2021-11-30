#include <glog/logging.h>

#include <scope/factor/JointConstFactor.h>
#include <scope/math/SO3.h>

namespace scope {
JointConstFactor::JointConstFactor(int joint, const Vector3& weight, const Matrix3 &mean,
                                   const Vector3 &lbnd, const Vector3 &ubnd,
                                   const std::string &name, int index,
                                   bool active)
    : Factor({}, {}, {joint}, {}, name, index, active),
      mWeight(weight),
      mMean(mean),
      mLowerBnd(lbnd),
      mUpperBnd(ubnd) {
  assert((mMean.col(0).cross(mMean.col(1)) - mMean.col(2)).norm() < 1e-5 &&
         "mean.R must be a rotational matrix");
  assert((lbnd.cwiseMin(ubnd) - lbnd).norm() < 1e-7);
  assert((ubnd.cwiseMax(lbnd) - ubnd).norm() < 1e-7);
}

int JointConstFactor::evaluate(const AlignedVector<Pose> &poses,
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

int JointConstFactor::linearize(const AlignedVector<Pose> &poses,
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

  math::SO3::dexpinv(eval.xi, lin.DF);
  J.noalias() = lin.DF * mMean.transpose();

  lin.Derror = eval.expL;
  lin.Derror -= eval.expL.cwiseProduct(eval.errorLInv);
  lin.Derror -= eval.expU;
  lin.Derror += eval.expU.cwiseProduct(eval.errorUInv);

  J.noalias() = lin.Derror.asDiagonal() * J;
  J.noalias() = mWeight.asDiagonal() * J;

  lin.status = Status::VALID;

  return 0;
}
}  // namespace scope
