#pragma once

#include <scope/factor/ExtraPinholeCamera.h>
#include <scope/factor/JointPinholeCameraFactor.h>

namespace scope {
class JointExtraPinholeCameraFactor : public JointPinholeCameraFactor,
                                      public ExtraPinholeCamera {
public:
  struct Evaluation : public JointPinholeCameraFactor::Evaluation,
                      public ExtraPinholeCamera::Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct Linearization : public JointPinholeCameraFactor::Linearization,
                         public ExtraPinholeCamera::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_FACTOR_EVALUATION_NEW
  SCOAT_FACTOR_LINEARIZATION_NEW

  JointExtraPinholeCameraFactor(int pose, const Scalar &sigma,
                                const Scalar &eps, const Pose &extraCamPose,
                                const Vector2 &measurement,
                                const Scalar &confidence = 1.0,
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
  virtual int evaluate(const Pose &pose, const Vector2 &measurement,
                       Evaluation &eval) const;

  virtual int linearize(const Pose &pose, const Vector2 &measurement,
                        const Evaluation &eval, Linearization &lin) const;
};
} // namespace scope
