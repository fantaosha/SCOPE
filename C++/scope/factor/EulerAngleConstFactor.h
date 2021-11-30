#pragma once

#include <scope/factor/JointConstFactor.h>

namespace scope {
class EulerAngleConstFactor : public JointConstFactor {
 public:
  struct Linearization : public JointConstFactor::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix3 H;
    Matrix3 D;
  };

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_FACTOR_EVALUATION_NEW
  SCOAT_FACTOR_LINEARIZATION_NEW

  EulerAngleConstFactor(int joint, const Vector3 &weight, const Matrix3 &mean,
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
};
}  // namespace scope
