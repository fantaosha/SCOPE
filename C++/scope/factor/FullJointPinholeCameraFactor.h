#pragma once

#include <memory>

#include <scope/factor/JointPinholeCameraFactor.h>

namespace scope {
class FullJointPinholeCameraFactor : public JointPinholeCameraFactor {
public:
  struct Evaluation : public JointPinholeCameraFactor::Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Vector2 camMeasurement;
  };

  struct Linearization : public JointPinholeCameraFactor::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Vector4 gCam;
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_FACTOR_EVALUATION_NEW
  SCOAT_FACTOR_LINEARIZATION_NEW

  FullJointPinholeCameraFactor(int pose, int camParam, const Scalar &sigma,
                               const Scalar &eps, const Vector2 &measurement,
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
  // camParam: the inverse of the camera intrinsic parameters
  virtual int evaluate(const Pose &pose, const VectorX &camParam,
                       const Vector2 &measurement, Evaluation &eval) const;

  virtual int linearize(const Pose &pose, const VectorX &camParam,
                        const Vector2 &measurement, const Evaluation &eval,
                        Linearization &lin) const;
};

} // namespace scope
