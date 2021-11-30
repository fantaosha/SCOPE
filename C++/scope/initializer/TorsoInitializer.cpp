#include <glog/logging.h>

#include <memory>
#include <scope/initializer/TorsoInitializer.h>

namespace scope {
namespace Initializer {
TorsoInitializer::TorsoInitializer(const Options &options)
    : Initializer(0, options) {}

int TorsoInitializer::updateGaussNewton() const {
  mH = mvMxx[0];
  mh = mvmx[0];

  return 0;
}

int TorsoInitializer::update(Scalar stepsize) const {
  assert(stepsize > 0);

  mDRootPoseChange = mhGN * stepsize;
  Pose::exp(mDRootPoseChange, mRootPoseChange);

  mvPoses[1][0].R.noalias() = mRootPoseChange.R * mvPoses[0][0].R;
  mvPoses[1][0].t = mRootPoseChange.t;
  mvPoses[1][0].t.noalias() += mRootPoseChange.R * mvPoses[0][0].t;

  return 0;
}

} // namespace Initializer
} // namespace scope
