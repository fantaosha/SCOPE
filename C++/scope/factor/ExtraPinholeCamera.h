#pragma once

#include <scope/base/Pose.h>
#include <scope/base/Types.h>

namespace scope {
class ExtraPinholeCamera {
public:
  struct Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // position in the extra camera frame
    Vector3 pExtraCam;
  };

  struct Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix<2, 3> JPoint;
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ExtraPinholeCamera(const Pose &T);

  const Pose &getExtraCameraPose() const;

protected:
  // rigid body transformation from the main camera frame to the extra camera
  // frame
  Pose mExtraCamPose;
};
} // namespace scope
