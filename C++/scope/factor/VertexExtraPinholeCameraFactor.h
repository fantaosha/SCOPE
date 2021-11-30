#pragma once

#include <scope/factor/ExtraPinholeCamera.h>
#include <scope/factor/VertexPinholeCameraFactor.h>

namespace scope {
class VertexExtraPinholeCameraFactor : public VertexPinholeCameraFactor,
                                       public ExtraPinholeCamera {
public:
  struct Evaluation : public VertexPinholeCameraFactor::Evaluation,
                      public ExtraPinholeCamera::Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct Linearization : public VertexPinholeCameraFactor::Linearization,
                         public ExtraPinholeCamera::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Vector3 gPoseR;
    Vector3 gJoint;
    VectorX gParam;
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_FACTOR_EVALUATION_NEW
  SCOAT_FACTOR_LINEARIZATION_NEW

  VertexExtraPinholeCameraFactor(int pose, int vParam, const Matrix3X &vDirs,
                                 const Vector3 &v, const Scalar &sigma,
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
  virtual int evaluate(const Pose &pose, const VectorX &vertexParam,
                       const Vector2 &measurement, Evaluation &eval) const;

  virtual int linearize(const Pose &pose, const VectorX &vertexParam,
                        const Vector2 &measurement, const Evaluation &eval,
                        Linearization &lin) const;
};
} // namespace scope
