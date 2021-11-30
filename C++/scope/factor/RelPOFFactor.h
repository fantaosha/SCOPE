#pragma once

#include <scope/factor/POFFactor.h>

namespace scope {
class RelPOFFactor : public POFFactor {
 public:
  struct Evaluation : public POFFactor::Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // vertex positions
    Vector6 vertex;
    Vector3 D;
    Scalar DNorm;
  };

  struct Linearization : public POFFactor::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix3 DF;
    Matrix36 DFR;  // u = [DF*R -Df*Rpar]

    Vector3 gJoint;
    VectorX gVertex;
  };

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_FACTOR_EVALUATION_NEW
  SCOAT_FACTOR_LINEARIZATION_NEW

  RelPOFFactor(int pose, int vParam, const std::vector<int> &parents,
               const std::array<Matrix3X, 2> &VDirs,
               const std::array<Vector3, 2> &V, const Scalar &sigma,
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
  int mParent;

  Matrix6X mVDirs;
  Vector6 mV;

 protected:
  virtual int evaluate(const Pose &pose, const Pose &parent,
                       const VectorX &vertexParam, const Vector3 &measurement,
                       Evaluation &eval) const;

  virtual int linearize(const Pose &pose, const Pose &parent,
                        const VectorX &vertexParam, const Vector3 &measurement,
                        const Evaluation &eval, Linearization &lin) const;
};
}  // namespace scope
