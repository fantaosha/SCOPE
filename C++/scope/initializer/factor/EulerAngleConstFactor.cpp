#include <glog/logging.h>

#include <scope/initializer/factor/EulerAngleConstFactor.h>
#include <scope/math/SO3.h>

namespace scope {
namespace Initializer {
EulerAngleConstFactor::EulerAngleConstFactor(int joint, const Vector3 &weight,
                                             const Matrix3 &mean,
                                             const Vector3 &lbnd,
                                             const Vector3 &ubnd,
                                             const std::string &name, int index)
    : JointConstFactor(joint, weight, mean, lbnd, ubnd, name, index){
}

int EulerAngleConstFactor::evaluate(const AlignedVector<Pose> &poses,
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

  math::SO3::Rot2XYZ(eval.errorR, eval.xi);

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

int EulerAngleConstFactor::linearize(const AlignedVector<Pose> &poses,
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

  lin.DF.setIdentity();
  lin.DF(1, 1) = cos(eval.xi[0]);
  lin.DF(2, 1) = sin(eval.xi[0]);
  lin.DF.col(2) = eval.errorR.col(2);

  lin.H.noalias() = lin.DF.transpose() * lin.DF;
  lin.H.diagonal().array() += 1e-4;

  lin.D.noalias() = lin.H.inverse() * lin.DF.transpose();

  J.noalias() = lin.D * mMean.transpose();

  lin.Derror = eval.expL;
  lin.Derror -= eval.expL.cwiseProduct(eval.errorLInv);
  lin.Derror -= eval.expU;
  lin.Derror += eval.expU.cwiseProduct(eval.errorUInv);

  J.noalias() = lin.Derror.asDiagonal() * J;
  J.noalias() = mWeight.asDiagonal() * J;

  lin.status = Status::VALID;

  return 0;
}
}  // namespace Initializer
}  // namespace scope
