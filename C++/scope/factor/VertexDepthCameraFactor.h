#pragma once

#include <scope/factor/DepthCameraFactor.h>

namespace scope {
class VertexDepthCameraFactor : public DepthCameraFactor {
public:
  struct Evaluation : public DepthCameraFactor::Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // vertex position in the body frame
    Vector3 vertex;
    // vertex position in the camera frame
    Vector3 point3D;
  };

  struct Linearization : public DepthCameraFactor::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VectorX gVertex;
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_FACTOR_EVALUATION_NEW
  SCOAT_FACTOR_LINEARIZATION_NEW

  VertexDepthCameraFactor(int pose, int vParam, const Matrix3X &vDirs,
                            const Vector3 &v, const Scalar &sigma,
                            const Scalar &eps, const Vector3 &measurement,
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
  Matrix3X mvDirs;
  Vector3 mv;

protected:
  virtual int evaluate(const Pose &pose, const VectorX &vertexParam,
                       const Vector3 &measurement, Evaluation &eval) const;

  virtual int linearize(const Pose &pose, const VectorX &vertexParam,
                        const Vector3 &measurement, const Evaluation &eval,
                        Linearization &lin) const;
};
}
