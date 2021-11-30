#pragma once

#include <scope/factor/Factor.h>

namespace scope {
class JointConstFactor : public Factor {
 public:
  struct Evaluation : public Factor::Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix3 errorR;
    Vector3 xi;

    Vector3 errorL;
    Vector3 errorU;

    Vector3 errorLInv;
    Vector3 errorUInv;

    Vector3 expL;
    Vector3 expU;
  };

  struct Linearization : public Factor::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix3 DF;

    Vector3 Derror;
  };

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_FACTOR_EVALUATION_NEW
  SCOAT_FACTOR_LINEARIZATION_NEW

  JointConstFactor(int joint, const Vector3 &weight, const Matrix3 &mean,
                   const Vector3 &lbnd, const Vector3 &ubnd,
                   const std::string &name = "", int index = -1,
                   bool active = true);

  virtual int evaluate(const AlignedVector<Pose> &poses,
                       const AlignedVector<VectorX> &shapes,
                       const AlignedVector<Matrix3> &joints,
                       const AlignedVector<VectorX> &params,
                       Factor::Evaluation &base_eval) const override;

  virtual int linearize(const AlignedVector<Pose> &poses,
                        const AlignedVector<VectorX> &shapes,
                        const AlignedVector<Matrix3> &joints,
                        const AlignedVector<VectorX> &params,
                        const Factor::Evaluation &base_eval,
                        Factor::Linearization &base_lin) const override;

 protected:
  Vector3 mWeight;
  Matrix3 mMean;

  Vector3 mLowerBnd;
  Vector3 mUpperBnd;

  const Scalar mScale = 2.5e-2;
  const Scalar mBnd = -1e-6;
};
}  // namespace scope
