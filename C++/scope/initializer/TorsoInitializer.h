#pragma once

#include <memory>
#include <vector>

#include <scope/initializer/Initializer.h>

namespace scope {
namespace Initializer {

class TorsoInitializer : public Initializer {
protected:
  // intermediates used for update
  mutable Vector6 mDRootPoseChange;
  mutable Pose mRootPoseChange;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TorsoInitializer(const Options &options);

  TorsoInitializer(const TorsoInitializer &) = delete;

  TorsoInitializer &operator=(const TorsoInitializer &) = delete;

protected:
  virtual int FKintree(int n) const override { return 0; }
  virtual int DFKintree() const override { return 0; }

  virtual int updateGaussNewton() const override;
  virtual int update(Scalar stepsize) const override;
};

} // namespace Initializer
} // namespace scope
