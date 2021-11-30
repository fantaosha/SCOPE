#pragma once

#include <memory>
#include <vector>

#include <scope/initializer/Initializer.h>

namespace scope {
namespace Initializer {

class ArmRefiner : public Initializer {
public:
  std::array<Vector3, 2> mRelJointLocations;

  mutable Vector6 mDJointChange;
  mutable std::array<Matrix3, 2> mvJointChange;

  mutable std::array<Matrix63, 2> mvB;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ArmRefiner(const Options &options,
             const std::array<Vector3, 2> &RelJointLocations);

  ArmRefiner(const ArmRefiner &) = delete;

  ArmRefiner &operator=(const ArmRefiner &) = delete;

public:
  virtual int FKintree(int n) const override;
  virtual int DFKintree() const override;

  virtual int updateGaussNewton() const override;
  virtual int update(Scalar stepsize) const override;
};

} // namespace Initializer
} // namespace scope
