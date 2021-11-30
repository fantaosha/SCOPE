#include <scope/factor/ExtraPinholeCamera.h>

namespace scope {
ExtraPinholeCamera::ExtraPinholeCamera(const Pose &T) : mExtraCamPose(T) {}

const Pose &ExtraPinholeCamera::getExtraCameraPose() const {
  return mExtraCamPose;
}

} // namespace scope
