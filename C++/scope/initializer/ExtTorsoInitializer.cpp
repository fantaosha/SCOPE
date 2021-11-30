#include <glog/logging.h>

#include <memory>
#include <scope/initializer/ExtTorsoInitializer.h>

namespace scope {
namespace Initializer {
ExtTorsoInitializer::ExtTorsoInitializer(const Options &options, const Vector3 &RelJointLocation)
    : Initializer(1, options), mRelJointLocation({RelJointLocation}) {}

int ExtTorsoInitializer::initialize(const Pose &T0,
                               const AlignedVector<Matrix3> &joints) const {
  assert(joints.size() == mNumJoints);

  if (joints.size() != mNumJoints) {
    LOG(ERROR) << "There should be " << mNumJoints << " joint states."
               << std::endl;

    exit(-1);
  }

  Vector3 xi;
  Matrix3 R;

  // Scale spine joint to feasbile range
  math::SO3::log(joints[0], xi);
  xi[0] = std::max(xi[0], SpineLowerBnd);
  xi[0] = std::min(xi[0], SpineUpperBnd);
  xi[1] = xi[2] = 0;
  math::SO3::exp(xi, R);

  mSpineAngle[0] = mSpineAngle[1] = xi[0];

  Initializer::initialize(T0, {R});

  return 0;
}

int ExtTorsoInitializer::initialize(const Pose &T0, const Scalar& SpineAngle) const {
  // Scale spine joint to feasbile range
  mSpineAngle[0] = SpineAngle;
  mSpineAngle[0] = std::max(mSpineAngle[0], SpineLowerBnd);
  mSpineAngle[0] = std::min(mSpineAngle[0], SpineUpperBnd);

  mSpineAngle[1] = mSpineAngle[0];

  Matrix3 R = Matrix3::Identity();
  R(1, 1) = cos(mSpineAngle[0]);
  R(2, 1) = sin(mSpineAngle[0]);
  R(2, 2) = R(1, 1);
  R(1, 2) = -R(2, 1);

  Initializer::initialize(T0, {R});

  return 0;
}

int ExtTorsoInitializer::FKintree(int n) const {
  auto &poses = mvPoses[n];
  auto &joints = mvJoints[n];

  poses[1].R.noalias() = poses[0].R * joints[0];
  poses[1].t = poses[0].t;
  poses[1].t.noalias() += poses[0].R * mRelJointLocation;

  return 0;
}

int ExtTorsoInitializer::DFKintree() const {
  const auto &poses = mvPoses[0];

  mB.head<3>() = poses[0].R.col(0);
  mB.tail<3>().noalias() = poses[1].t.cross(poses[0].R.col(0));

  return 0;
}

int ExtTorsoInitializer::updateGaussNewton() const {
  mH.resize(7, 7);

  mH.block<6, 1>(0, 6).noalias() = mvMxx[1] * mB;
  mH(6, 6) = mB.dot(mH.block<6, 1>(0, 6));
  mH(6, 6) += mvMuu[0](0, 0);

  mH.topLeftCorner<6, 6>() = mvMxx[0] + mvMxx[1];

  mH.block<1, 6>(6, 0).noalias() = mH.block<6, 1>(0, 6).transpose();

  mh.resize(7);

  mh[6] = mB.dot(mvmx[1]);
  mh[6] += mvmu[0][0];

  mh.head<6>() = mvmx[0] + mvmx[1];

  return 0;
}

int ExtTorsoInitializer::solveGaussNewton() const {
  mLambda = mH.diagonal() * (mDLambda - 1);
  mLambda.array() += mOptions.delta;
  mH.diagonal() += mLambda;

  mHchol.compute(mH.topLeftCorner<6, 6>());

  mS.noalias() = -mHchol.solve(mH.block<6, 1>(0, 6));
  ms.noalias() = -mHchol.solve(mh.head<6>());

  mA = mH(6, 6) + mH.block<6, 1>(0, 6).dot(mS);
  ma = mh[6] + mH.block<6, 1>(0, 6).dot(ms);

  mhGN.resize(7);

  // Guarantee mKneeJoint in [0, 0.78 * M_PI]
  mhGN[6] = -ma / mA;
  mhGN[6] = std::max(mhGN[6], SpineLowerBnd - mSpineAngle[0]);
  mhGN[6] = std::min(mhGN[6], SpineUpperBnd - mSpineAngle[0]);
  mhGN.head<6>() = ms;
  mhGN.head<6>().noalias() += mS * mhGN[6];

  mSquaredError = mh;
  mSquaredError += 0.5 * mH * mhGN;

  mE = mhGN.dot(mSquaredError);
  mSquaredError = mhGN.cwiseAbs2();
  mE -= 0.5 * mLambda.dot(mSquaredError);

  return 0;
}

int ExtTorsoInitializer::update(Scalar stepsize) const {
  assert(stepsize > 0);

  mDRootPoseChange = mhGN.head<6>() * stepsize;
  Pose::exp(mDRootPoseChange, mRootPoseChange);

  mvPoses[1][0].R.noalias() = mRootPoseChange.R * mvPoses[0][0].R;
  mvPoses[1][0].t = mRootPoseChange.t;
  mvPoses[1][0].t.noalias() += mRootPoseChange.R * mvPoses[0][0].t;

  mSpineAngle[1] = mSpineAngle[0] + stepsize * mhGN[6];

  mvJoints[1][0].setIdentity();
  mvJoints[1][0](1, 1) = std::cos(mSpineAngle[1]);
  mvJoints[1][0](2, 2) = mvJoints[1][0](1, 1);
  mvJoints[1][0](2, 1) = std::sin(mSpineAngle[1]);
  mvJoints[1][0](1, 2) = -mvJoints[1][0](2, 1);

  FKintree(1);

  return 0;
}

int ExtTorsoInitializer::accept() const {
  mSpineAngle[0] = mSpineAngle[1];

  return Initializer::accept();
}
} // namespace Initializer
} // namespace scope
