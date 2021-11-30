#include <glog/logging.h>

#include <memory>
#include <scope/initializer/ArmRefiner.h>

namespace scope {
namespace Initializer {
ArmRefiner::ArmRefiner(const Options &options,
                       const std::array<Vector3, 2> &RelJointLocations)
    : Initializer(2, options), mRelJointLocations(RelJointLocations) {}

int ArmRefiner::FKintree(int n) const {
  auto &poses = mvPoses[n];
  auto &joints = mvJoints[n];

  poses[1].R.noalias() = poses[0].R * joints[0];
  poses[1].t = poses[0].t;
  poses[1].t.noalias() += poses[0].R * mRelJointLocations[0];

  poses[2].R.noalias() = poses[1].R * joints[1];
  poses[2].t = poses[1].t;
  poses[2].t.noalias() += poses[1].R * mRelJointLocations[1];

  return 0;
}

int ArmRefiner::DFKintree() const {
  const auto &poses = mvPoses[0];

  mvB[0].topRows<3>() = poses[0].R;
  mvB[0].block<3, 1>(3, 0) = poses[1].t.cross(poses[0].R.col(0));
  mvB[0].block<3, 1>(3, 1) = poses[1].t.cross(poses[0].R.col(1));
  mvB[0].block<3, 1>(3, 2) = poses[1].t.cross(poses[0].R.col(2));

  mvB[1].topRows<3>() = poses[1].R;
  mvB[1].block<3, 1>(3, 0) = poses[2].t.cross(poses[1].R.col(0));
  mvB[1].block<3, 1>(3, 1) = poses[2].t.cross(poses[1].R.col(1));
  mvB[1].block<3, 1>(3, 2) = poses[2].t.cross(poses[1].R.col(2));

  return 0;
}

int ArmRefiner::updateGaussNewton() const {
  Matrix6 temp1;
  Matrix63 temp2;

  temp1 = mvMxx[1] + mvMxx[2];
  temp2.noalias() = temp1 * mvB[0];

  mH.resize(6, 6);
  mH.topLeftCorner<3, 3>().noalias() = mvB[0].transpose() * temp2;

  temp2.noalias() = mvMxx[2] * mvB[1];

  mH.topRightCorner<3, 3>().noalias() = mvB[0].transpose() * temp2;
  mH.bottomLeftCorner<3, 3>().noalias() = mH.topRightCorner<3, 3>().transpose();

  mH.bottomRightCorner<3, 3>().noalias() = mvB[1].transpose() * temp2;

  mH.topLeftCorner<3, 3>().noalias() += mvMuu[0];
  mH.bottomRightCorner<3, 3>().noalias() += mvMuu[1];

  Vector6 temp3;
  temp3 = mvmx[1] + mvmx[2];

  mh.resize(6);
  mh.head<3>() = mvmu[0];
  mh.tail<3>() = mvmu[1];

  mh.head<3>().noalias() += mvB[0].transpose() * temp3;
  mh.tail<3>().noalias() += mvB[1].transpose() * mvmx[2];

  return 0;
}

int ArmRefiner::update(Scalar stepsize) const {
  assert(stepsize > 0);

  mDJointChange = mhGN * stepsize;

  math::SO3::exp(mDJointChange.head<3>(), mvJointChange[0]);
  math::SO3::exp(mDJointChange.tail<3>(), mvJointChange[1]);

  mvJoints[1][0].noalias() = mvJointChange[0] * mvJoints[0][0];
  mvJoints[1][1].noalias() = mvJointChange[1] * mvJoints[0][1];

  FKintree(1);

  return 0;
}

} // namespace Initializer
} // namespace scope
