#pragma once

#include <memory>
#include <vector>

#include <scope/initializer/Initializer.h>

namespace scope {
namespace Initializer {

class LegInitializer : public Initializer {
public:
  std::array<Vector3, 2> mRelJointLocations;

  mutable Vector6 mDRootPoseChange;
  mutable Pose mRootPoseChange;

  mutable Vector3 mDJointChange;
  mutable Matrix3 mJointChange;

  mutable Scalar mKneeJoint[2];

  mutable Matrix63 mB0;
  mutable Vector6 mB1;

  mutable Vector<9> mS;
  mutable Vector<9> ms;
  mutable Scalar mA;
  mutable Scalar ma;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LegInitializer(const Options &options,
                 const std::array<Vector3, 2> &RelJointLocations);

  LegInitializer(const LegInitializer &) = delete;

  LegInitializer &operator=(const LegInitializer &) = delete;

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
  const Scalar KneeLowerBnd = 0;
  const Scalar KneeUpperBnd = 0.85 * M_PI;
};

} // namespace Initializer
} // namespace scope
