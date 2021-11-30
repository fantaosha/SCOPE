#include <glog/logging.h>

#include <memory>
#include <scope/initializer/ArmInitializer.h>

namespace scope {
namespace Initializer {
ArmInitializer::ArmInitializer(const Options &options,
                               const std::array<Vector3, 2> &RelJointLocations,
                               Arm LR)
    : Initializer(2, options), mRelJointLocations(RelJointLocations), mArm(LR),
      mElbowLowerBnd(LR == Arm::Left ? -0.85 * M_PI : 0),
      mElbowUpperBnd(LR == Arm::Left ? 0 : 0.85 * M_PI) {}

int ArmInitializer::initialize(const Pose &T0,
                               const AlignedVector<Matrix3> &joints) const {
  assert(joints.size() == mNumJoints);

  if (joints.size() != mNumJoints) {
    LOG(ERROR) << "There should be " << mNumJoints << " joint states."
               << std::endl;

    exit(-1);
  }

  Vector3 xi;
  Matrix3 R;

  // Scale eblow joint to feasbile range
  math::SO3::log(joints[1], xi);
  xi[1] = std::max(xi[1], mElbowLowerBnd);
  xi[1] = std::min(xi[1], mElbowUpperBnd);
  xi[0] = xi[2] = 0;
  math::SO3::exp(xi, R);

  mElbowJoint[0] = mElbowJoint[1] = xi[1];

  Initializer::initialize(T0, {joints[0], R});

  return 0;
}

int ArmInitializer::FKintree(int n) const {
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

int ArmInitializer::DFKintree() const {
  const auto &poses = mvPoses[0];

  mB0.topRows<3>() = poses[0].R;
  mB0.block<3, 1>(3, 0).noalias() = poses[1].t.cross(poses[0].R.col(0));
  mB0.block<3, 1>(3, 1).noalias() = poses[1].t.cross(poses[0].R.col(1));
  mB0.block<3, 1>(3, 2).noalias() = poses[1].t.cross(poses[0].R.col(2));

  mB1.head<3>() = poses[1].R.col(1);
  mB1.tail<3>().noalias() = poses[2].t.cross(poses[1].R.col(1));

  return 0;
}

int ArmInitializer::updateGaussNewton() const {
  mH.resize(4, 4);

  Vector6 temp1;
  Matrix6 temp2;
  Matrix63 temp3;

  temp1.noalias() = mvMxx[2] * mB1;
  mH.topRightCorner<3, 1>().noalias() = mB0.transpose() * temp1;
  mH(3, 3) = mB1.dot(temp1);
  mH(3, 3) += mvMuu[1](1, 1);

  temp2 = mvMxx[1] + mvMxx[2];
  temp3.noalias() = temp2 * mB0;
  mH.topLeftCorner<3, 3>().noalias() = mB0.transpose() * temp3;
  mH.topLeftCorner<3, 3>().noalias() += mvMuu[0];

  mH.bottomLeftCorner<1, 3>().noalias() = mH.topRightCorner<3, 1>().transpose();

  mh.resize(4);

  mh[3] = mB1.dot(mvmx[2]);
  mh[3] += mvmu[1][1];

  temp1 = mvmx[1] + mvmx[2];
  mh.head<3>().noalias() = mB0.transpose() * temp1;
  mh.head<3>() += mvmu[0];

  return 0;
}

int ArmInitializer::solveGaussNewton() const {
  mLambda = mH.diagonal() * (mDLambda - 1);
  mLambda.array() += mOptions.delta;
  mH.diagonal() += mLambda;

  mHchol.compute(mH.topLeftCorner<3, 3>());

  mS.noalias() = -mHchol.solve(mH.block<3, 1>(0, 3));
  ms.noalias() = -mHchol.solve(mh.head<3>());

  mA = mH(3, 3) + mH.block<3, 1>(0, 3).dot(mS);
  ma = mh[3] + mH.block<3, 1>(0, 3).dot(ms);

  mhGN.resize(4);

  // Guarantee mKneeJoint in [0, 0.78 * M_PI]
  mhGN[3] = -ma / mA;
  mhGN[3] = std::max(mhGN[3], mElbowLowerBnd - mElbowJoint[0]);
  mhGN[3] = std::min(mhGN[3], mElbowUpperBnd - mElbowJoint[0]);
  mhGN.head<3>() = ms;
  mhGN.head<3>().noalias() += mS * mhGN[3];

  mSquaredError = mh;
  mSquaredError += 0.5 * mH * mhGN;

  mE = mhGN.dot(mSquaredError);
  mSquaredError = mhGN.cwiseAbs2();
  mE -= 0.5 * mLambda.dot(mSquaredError);

  return 0;
}

int ArmInitializer::update(Scalar stepsize) const {
  assert(stepsize > 0);

  mDJointChange = mhGN.head<3>() * stepsize;
  math::SO3::exp(mDJointChange, mJointChange);
  mvJoints[1][0].noalias() = mJointChange * mvJoints[0][0];

  mElbowJoint[1] = mElbowJoint[0] + stepsize * mhGN[3];

  mvJoints[1][1].setIdentity();
  mvJoints[1][1](0, 0) = std::cos(mElbowJoint[1]);
  mvJoints[1][1](2, 2) = mvJoints[1][1](0, 0);
  mvJoints[1][1](0, 2) = std::sin(mElbowJoint[1]);
  mvJoints[1][1](2, 0) = -mvJoints[1][1](0, 2);

  FKintree(1);

  return 0;
}

int ArmInitializer::accept() const {
  mElbowJoint[0] = mElbowJoint[1];

  return Initializer::accept();
}
} // namespace Initializer
} // namespace scope
