#pragma once

#include <memory>
#include <vector>

#include <scope/initializer/Initializer.h>

namespace scope {
namespace Initializer {
class ExtTorsoInitializer : public Initializer {
protected:
  Vector3 mRelJointLocation;

  mutable Vector6 mDRootPoseChange;
  mutable Pose mRootPoseChange;

  mutable Scalar mSpineAngle[2];

  mutable Vector6 mB;

  mutable Vector<6> mS;
  mutable Vector<6> ms;
  mutable Scalar mA;
  mutable Scalar ma;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ExtTorsoInitializer(const Options &options, const Vector3 &RelJointLocation);

  ExtTorsoInitializer(const ExtTorsoInitializer &) = delete;

  ExtTorsoInitializer &operator=(const ExtTorsoInitializer &) = delete;

  virtual int initialize(const Pose &T0,
                         const AlignedVector<Matrix3> &joints) const override;

  int initialize(const Pose& T0, const Scalar& SpineAngle) const;

  Scalar getSpineAngle() const { return mSpineAngle[0]; };

public:
  virtual int FKintree(int n) const override;
  virtual int DFKintree() const override;

  virtual int updateGaussNewton() const override;
  virtual int solveGaussNewton() const override;

  virtual int update(Scalar stepsize) const override;
  virtual int accept() const override;

protected:
  const Scalar SpineLowerBnd = 0;
  const Scalar SpineUpperBnd = 0.3 * M_PI;
};

} // namespace Initializer
} // namespace scope
