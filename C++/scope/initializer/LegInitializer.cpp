#include <glog/logging.h>

#include <memory>
#include <scope/initializer/LegInitializer.h>

namespace scope {
namespace Initializer {
LegInitializer::LegInitializer(const Options &options,
                               const std::array<Vector3, 2> &RelJointLocations)
    : Initializer(2, options), mRelJointLocations(RelJointLocations) {
  if (mRelJointLocations.size() != mNumJoints) {
    LOG(ERROR) << "Inconsistent relative joint locations." << std::endl;
    exit(-1);
  }
}

int LegInitializer::initialize(const Pose &T0,
                               const AlignedVector<Matrix3> &joints) const {
  assert(joints.size() == mNumJoints);

  if (joints.size() != mNumJoints) {
    LOG(ERROR) << "There should be " << mNumJoints << " joint states."
               << std::endl;

    exit(-1);
  }

  Vector3 xi;
  Matrix3 R;

  // Scale knee joint to feasbile range
  math::SO3::log(joints[1], xi);
  xi[0] = std::max(xi[0], KneeLowerBnd);
  xi[0] = std::min(xi[0], KneeUpperBnd);
  xi.tail<2>().setZero();
  math::SO3::exp(xi, R);

  mKneeJoint[0] = mKneeJoint[1] = xi[0];

  Initializer::initialize(T0, {joints[0], R});

  return 0;
}

int LegInitializer::FKintree(int n) const {
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

int LegInitializer::DFKintree() const {
  const auto &poses = mvPoses[0];

  mB0.topRows<3>() = poses[0].R;
  mB0.block<3, 1>(3, 0).noalias() = poses[1].t.cross(poses[0].R.col(0));
  mB0.block<3, 1>(3, 1).noalias() = poses[1].t.cross(poses[0].R.col(1));
  mB0.block<3, 1>(3, 2).noalias() = poses[1].t.cross(poses[0].R.col(2));

  mB1.head<3>() = poses[1].R.col(0);
  mB1.tail<3>().noalias() = poses[2].t.cross(poses[1].R.col(0));

  return 0;
}

int LegInitializer::updateGaussNewton() const {
  mH.setZero(10, 10);

  mH.topLeftCorner<6, 6>() = mvMxx[2];
  mH.block<6, 1>(0, 9).noalias() = mH.topLeftCorner<6, 6>() * mB1;
  mH.block<3, 1>(6, 9).noalias() = mB0.transpose() * mH.block<6, 1>(0, 9);
  mH(9, 9) = mB1.dot(mH.block<6, 1>(0, 9));
  mH(9, 9) += mvMuu[1](0, 0);

  mH.topLeftCorner<6, 6>() += mvMxx[1];
  mH.block<6, 3>(0, 6).noalias() = mH.topLeftCorner<6, 6>() * mB0;
  mH.block<3, 3>(6, 6).noalias() = mB0.transpose() * (mH.block<6, 3>(0, 6));
  mH.block<3, 3>(6, 6) += mvMuu[0];

  mH.topLeftCorner<6, 6>() += mvMxx[0];

  mH.block<3, 6>(6, 0) = mH.block<6, 3>(0, 6).transpose();
  mH.block<1, 9>(9, 0) = mH.block<9, 1>(0, 9).transpose();

  mh.resize(10);

  mh.head<6>() = mvmx[2];
  mh[9] = mB1.dot(mh.head<6>());
  mh[9] += mvmu[1][0];

  mh.head<6>() += mvmx[1];
  mh.segment<3>(6) = mB0.transpose() * mh.head<6>();
  mh.segment<3>(6) += mvmu[0];

  mh.head<6>() += mvmx[0];

  return 0;
}

int LegInitializer::solveGaussNewton() const {
  mLambda = mH.diagonal() * (mDLambda - 1);
  mLambda.array() += mOptions.delta;
  mH.diagonal() += mLambda;

  mHchol.compute(mH.topLeftCorner<9, 9>());

  mS.noalias() = -mHchol.solve(mH.block<9, 1>(0, 9));
  ms.noalias() = -mHchol.solve(mh.head<9>());

  mA = mH(9, 9) + mH.block<9, 1>(0, 9).dot(mS);
  ma = mh[9] + mH.block<9, 1>(0, 9).dot(ms);

  mhGN.resize(10);

  // Guarantee mKneeJoint in [0, 0.78 * M_PI]
  mhGN[9] = -ma / mA;
  mhGN[9] = std::max(mhGN[9], KneeLowerBnd - mKneeJoint[0]);
  mhGN[9] = std::min(mhGN[9], KneeUpperBnd - mKneeJoint[0]);
  mhGN.head<9>() = ms;
  mhGN.head<9>().noalias() += mS * mhGN[9];

  mSquaredError = mh;
  mSquaredError += 0.5 * mH * mhGN;

  mE = mhGN.dot(mSquaredError);
  mSquaredError = mhGN.cwiseAbs2();
  mE -= 0.5 * mLambda.dot(mSquaredError);

  return 0;
}

int LegInitializer::update(Scalar stepsize) const {
  assert(stepsize > 0);

  mDRootPoseChange = mhGN.head<6>() * stepsize;
  Pose::exp(mDRootPoseChange, mRootPoseChange);

  mvPoses[1][0].R.noalias() = mRootPoseChange.R * mvPoses[0][0].R;
  mvPoses[1][0].t = mRootPoseChange.t;
  mvPoses[1][0].t.noalias() += mRootPoseChange.R * mvPoses[0][0].t;

  mDJointChange = mhGN.segment<3>(6) * stepsize;
  math::SO3::exp(mDJointChange, mJointChange);
  mvJoints[1][0].noalias() = mJointChange * mvJoints[0][0];

  mKneeJoint[1] = mKneeJoint[0] + stepsize * mhGN[9];

  mvJoints[1][1].setIdentity();
  mvJoints[1][1](1, 1) = std::cos(mKneeJoint[1]);
  mvJoints[1][1](2, 2) = mvJoints[1][1](1, 1);
  mvJoints[1][1](2, 1) = std::sin(mKneeJoint[1]);
  mvJoints[1][1](1, 2) = -mvJoints[1][1](2, 1);

  FKintree(1);

  return 0;
}

int LegInitializer::accept() const {
  mKneeJoint[0] = mKneeJoint[1];

  return Initializer::accept();
}
} // namespace Initializer
} // namespace scope
