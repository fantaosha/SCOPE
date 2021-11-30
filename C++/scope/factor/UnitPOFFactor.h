#pragma once

#include <scope/factor/POFFactor.h>

namespace scope {
class UnitPOFFactor : public POFFactor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_FACTOR_EVALUATION_NEW
  SCOAT_FACTOR_LINEARIZATION_NEW

  UnitPOFFactor(int pose, const Vector3 &s, const Scalar &sigma,
                const Scalar &eps, const Vector3 &measurement,
                const Scalar &confidence = 1.0, const std::string &name = "",
                int index = -1, bool active = true);

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
  Vector3 mS; // unit directional vector
};
} // namespace scope
