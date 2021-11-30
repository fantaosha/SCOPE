#pragma once

#include <memory>
#include <vector>

#include <scope/initializer/Initializer.h>

namespace scope {
namespace Initializer {
class ArmInitializer : public Initializer {
public:
  enum class Arm { Left, Right };

protected:
  std::array<Vector3, 2> mRelJointLocations;

  mutable Vector3 mDJointChange;
  mutable Matrix3 mJointChange;

  mutable Scalar mElbowJoint[2];

  mutable Matrix63 mB0;
  mutable Vector6 mB1;

  mutable Vector<3> mS;
  mutable Vector<3> ms;
  mutable Scalar mA;
  mutable Scalar ma;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ArmInitializer(const Options &options,
                 const std::array<Vector3, 2> &RelJointLocations, Arm LR);

  ArmInitializer(const ArmInitializer &) = delete;

  ArmInitializer &operator=(const ArmInitializer &) = delete;

  virtual int initialize(const Pose &T0,
                         const AlignedVector<Matrix3> &joints) const override;

public:
  virtual int FKintree(int n) const override;
  virtual int DFKintree() const override;

  virtual int updateGaussNewton() const override;
  virtual int solveGaussNewton() const override;

  virtual int update(Scalar stepsize) const override;
  virtual int accept() const override;

protected:
  Arm mArm;

  const Scalar mElbowLowerBnd;
  const Scalar mElbowUpperBnd;
};

} // namespace Initializer
} // namespace scope
