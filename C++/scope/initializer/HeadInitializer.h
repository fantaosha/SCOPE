#pragma once

#include <memory>
#include <vector>

#include <scope/initializer/Initializer.h>

namespace scope {
namespace Initializer {

class HeadInitializer : public Initializer {
protected:
  // intermediates used for update
  mutable Vector6 mDRootPoseChange;
  mutable Pose mRootPoseChange;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  HeadInitializer(const Options &options);

  HeadInitializer(const HeadInitializer &) = delete;

  HeadInitializer &operator=(const HeadInitializer &) = delete;

protected:
  virtual int FKintree(int n) const override { return 0; }
  virtual int DFKintree() const override { return 0; }

  virtual int updateGaussNewton() const override;
  virtual int update(Scalar stepsize) const override;
};

} // namespace Initializer
} // namespace scope
