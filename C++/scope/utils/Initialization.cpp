#include <scope/utils/Initialization.h>

#include <memory>

namespace scope {
namespace Initialization {
namespace SMPL {
int initialize(const MatrixX &ShapeSqrtCov, Scalar FocalLength,
               // KeyPoint2D setup
               const Matrix3X &Measurements2D, Scalar KeyPoint2DGMSigma,
               Scalar KeyPoint2DGMEps,
               const VectorX &KeyPoint2DConfidenceThreshold,
               // POF setup
               const Matrix3X &POFMeasurements, Scalar POFGMSigma,
               Scalar POFGMEps,
               // KeyPoint3D setup
               const Matrix3X &Measurements3D, Scalar KeyPoint3DGMSigma,
               Scalar KeyPoint3DGMEps,
               // joint limit
               const AlignedVector<AlignedVector<Matrix6>> &JointLimitWeight,
               const AlignedVector<AlignedVector<Vector6>> &JointLimitBias,
               const AlignedVector<VectorX> &JointLimitPRelua,
               const AlignedVector<VectorX> &JointLimitPRelub,
               const Vector<23> &JointLimitScale,
               // initial estimates
               Pose &TorsoPose,      // torso pose
               Matrix3 &SpineJoint,  // spine joint
               AlignedVector<Matrix3> &LeftArmJoint,
               AlignedVector<Matrix3> &RightArmJoint,
               AlignedVector<Matrix3> &LeftLegJoint,
               AlignedVector<Matrix3> &RightLegJoint, VectorX &betas) {
  //------------------------------------------------------------------------------
  // Optimizer weights
  //------------------------------------------------------------------------------
  Initializer::Options options;
  options.rel_lin_func_decrease_tol = 4e-3;
  options.delta = 2;
  options.weights[Initializer::FactorIndex::PinholeCamera] =
      FocalLength * FocalLength;
  options.weights[Initializer::FactorIndex::POF] = 100000;
  options.weights[Initializer::FactorIndex::DepthCamera] = 1000;
  options.weights[Initializer::FactorIndex::JointLimit] = 1000;
  options.weights[Initializer::FactorIndex::JointConst] = 1000;
  options.weights[Initializer::FactorIndex::Pose] = 100;

  //------------------------------------------------------------------------------
  // Create skeleton
  //------------------------------------------------------------------------------
  VectorX KeyPoints;
  createInitialSMPLSkeleton(Measurements3D, KeyPoints);

  //------------------------------------------------------------------------------
  // Torso Initializer
  //------------------------------------------------------------------------------
  const auto &TorsoIndex = InitialInfo::SMPL::TorsoIndex[1];
  const auto &TorsoDepthCamSInfo =
      scope::InitialInfo::SMPL::TorsoDepthCameraFactorInfo[1];

  solvePose(TorsoDepthCamSInfo, KeyPoints.segment<3>(3 * TorsoIndex[0][1]),
            KeyPoints, Measurements3D, TorsoPose);

  Vector3 t;
  solveReducedEPnP(KeyPointInfo::SMPL::KeyPoint3Dto2D, Measurements3D,
                   Measurements2D, KeyPoint2DConfidenceThreshold, t);
  TorsoPose.t = t;

  Scalar SpineAngle = acos(POFMeasurements.col(6).dot(POFMeasurements.col(7)));

  // create factors
  const auto &ExtTorsoIndex = InitialInfo::SMPL::ExtTorsoIndex[1];
  const auto &ExtTorsoPinholeCamInfo =
      InitialInfo::SMPL::ExtTorsoPinholeCameraFactorInfo[1];
  const auto &ExtTorsoPOFInfo = InitialInfo::SMPL::ExtTorsoPOFFactorInfo[1];
  const auto &ExtTorsoDepthCamInfo =
      InitialInfo::SMPL::ExtTorsoDepthCameraFactorInfo[1];

  // Pinhole camera factors
  std::vector<std::shared_ptr<Initializer::PinholeCameraFactor>>
      ExtTorsoPinholeCamFactors;
  createInitialPinholeCameraFactors(
      ExtTorsoPinholeCamInfo, ExtTorsoIndex, KeyPoints, KeyPoint2DGMSigma,
      KeyPoint2DGMEps, Measurements2D, KeyPoint2DConfidenceThreshold,
      ExtTorsoPinholeCamFactors);

  // POF factors
  std::vector<std::shared_ptr<Initializer::POFFactor>> ExtTorsoPOFFactors;
  createInitialPOFFactors(ExtTorsoPOFInfo, POFGMSigma, POFGMEps,
                          POFMeasurements, ExtTorsoPOFFactors);

  // Pose factors
  Matrix6 ExtTorsoPoseSqrtCov = Matrix6::Zero();
  ExtTorsoPoseSqrtCov.diagonal().tail<3>().array() = 5;
  std::shared_ptr<Initializer::PoseFactor> ExtTorsoPoseFactor =
      std::make_shared<Initializer::PoseFactor>(0, ExtTorsoPoseSqrtCov,
                                                TorsoPose);

  // add factors
  Initializer::ExtTorsoInitializer ExtTorsoInitializer(
      options, KeyPoints.segment<3>(3 * ExtTorsoIndex[1][1]));

  for (auto &factor : ExtTorsoPinholeCamFactors) {
    ExtTorsoInitializer.addPinholeCameraFactor(factor);
  }

  for (auto &factor : ExtTorsoPOFFactors) {
    ExtTorsoInitializer.addPOFFactor(factor);
  }

  ExtTorsoInitializer.addPoseFactor(ExtTorsoPoseFactor);

  ExtTorsoInitializer.initialize(TorsoPose, SpineAngle);
  ExtTorsoInitializer.solve();

  TorsoPose = ExtTorsoInitializer.getPoses()[0];
  SpineAngle = ExtTorsoInitializer.getSpineAngle();

  // Torso depth camera factors
  std::vector<std::shared_ptr<Initializer::DepthCameraFactor>>
      ExtTorsoDepthCamFactors;
  createInitialDepthCameraFactors(
      ExtTorsoDepthCamInfo, ExtTorsoIndex, KeyPoints, KeyPoint3DGMSigma,
      KeyPoint3DGMEps, Measurements3D, TorsoPose.t, ExtTorsoDepthCamFactors);

  for (auto &factor : ExtTorsoDepthCamFactors) {
    ExtTorsoInitializer.addDepthCameraFactor(factor);
  }

  ExtTorsoInitializer.initialize(TorsoPose, SpineAngle);
  ExtTorsoInitializer.solve();

  TorsoPose = ExtTorsoInitializer.getPoses()[0];
  const auto &ChestPose = ExtTorsoInitializer.getPoses()[1];

  SpineJoint = ExtTorsoInitializer.getJoints()[0];

  //------------------------------------------------------------------------------
  // Left Leg Initializer
  //------------------------------------------------------------------------------
  LeftLegJoint.resize(2);
  Matrix3 LeftLegMat = Matrix3::Identity();
  LeftLegMat.col(1) -= 4000 * TorsoPose.R.transpose() * POFMeasurements.col(1);
  projectToSO3(LeftLegMat, LeftLegJoint[0]);

  Scalar LeftKneeAngle =
      acos(POFMeasurements.col(1).dot(POFMeasurements.col(2)));
  LeftLegJoint[1].setIdentity();
  LeftLegJoint[1](1, 1) = cos(LeftKneeAngle);
  LeftLegJoint[1](2, 1) = sin(LeftKneeAngle);
  LeftLegJoint[1](2, 2) = LeftLegJoint[1](1, 1);
  LeftLegJoint[1](1, 2) = -LeftLegJoint[1](2, 1);

  // create factors
  const auto &LeftLegIndex = InitialInfo::SMPL::LeftLegIndex[1];
  const auto &LeftLegKintree = InitialInfo::SMPL::LeftLegKinematicsTree;
  const auto &LeftLegPinholeCamInfo =
      InitialInfo::SMPL::LeftLegPinholeCameraFactorInfo[1];
  const auto &LeftLegPOFInfo = InitialInfo::SMPL::LeftLegPOFFactorInfo[1];
  const auto &LeftLegDepthCamInfo =
      InitialInfo::SMPL::LeftLegDepthCameraFactorInfo[1];
  const auto &LeftLegEulerAngleConstInfo =
      InitialInfo::SMPL::LeftLegEulerAngleConstInfo;

  // pinhole camera factors
  std::vector<std::shared_ptr<Initializer::PinholeCameraFactor>>
      LeftLegPinholeCamFactors;
  createInitialPinholeCameraFactors(
      LeftLegPinholeCamInfo, LeftLegIndex, KeyPoints, KeyPoint2DGMSigma,
      KeyPoint2DGMEps, Measurements2D, KeyPoint2DConfidenceThreshold,
      LeftLegPinholeCamFactors);

  // POF factors
  std::vector<std::shared_ptr<Initializer::POFFactor>> LeftLegPOFFactors;
  createInitialPOFFactors(LeftLegPOFInfo, POFGMSigma, POFGMEps, POFMeasurements,
                          LeftLegPOFFactors);

  // depth camera factors
  std::vector<std::shared_ptr<Initializer::DepthCameraFactor>>
      LeftLegDepthCamFactors;
  createInitialDepthCameraFactors(
      LeftLegDepthCamInfo, LeftLegIndex, KeyPoints, KeyPoint3DGMSigma,
      KeyPoint3DGMEps, Measurements3D, TorsoPose.t, LeftLegDepthCamFactors);

  // pose factors
  Matrix6 LeftLegPoseSqrtCov = Matrix6::Zero();
  LeftLegPoseSqrtCov.diagonal().array() = 20;
  std::shared_ptr<Initializer::PoseFactor> LeftLegPoseFactor =
      std::make_shared<Initializer::PoseFactor>(0, LeftLegPoseSqrtCov,
                                                TorsoPose);

  // joint constraint factors
  std::vector<std::shared_ptr<Initializer::EulerAngleConstFactor>>
      LeftLegEulerAngleConstFactors;

  createInitialEulerAngleConstFactors(LeftLegEulerAngleConstInfo,
                                      LeftLegEulerAngleConstFactors);

  // left leg initializer
  std::array<Vector3, 2> LeftLegRelJointLocations;

  for (int i = 0; i < LeftLegKintree.size(); i++) {
    LeftLegRelJointLocations[i] =
        KeyPoints.segment<3>(3 * LeftLegIndex[LeftLegKintree[i][1]][1]) -
        KeyPoints.segment<3>(3 * LeftLegIndex[LeftLegKintree[i][0]][1]);
  }

  Initializer::LegInitializer LeftLegInitializer(options,
                                                 LeftLegRelJointLocations);

  for (auto &factor : LeftLegPinholeCamFactors) {
    LeftLegInitializer.addPinholeCameraFactor(factor);
  }

  for (auto &factor : LeftLegPOFFactors) {
    LeftLegInitializer.addPOFFactor(factor);
  }

  for (auto &factor : LeftLegDepthCamFactors) {
    LeftLegInitializer.addDepthCameraFactor(factor);
  }

  LeftLegInitializer.addPoseFactor(LeftLegPoseFactor);

  for (auto &factor : LeftLegEulerAngleConstFactors) {
    LeftLegInitializer.addJointConstFactor(factor);
  }

  LeftLegInitializer.updateFactorWeights(options.weights);
  LeftLegInitializer.initialize(TorsoPose, LeftLegJoint);
  LeftLegInitializer.solve();

  LeftLegJoint = LeftLegInitializer.getJoints();

  //------------------------------------------------------------------------------
  // Right Leg Initializer
  //------------------------------------------------------------------------------
  RightLegJoint.resize(2);
  Matrix3 RightLegMat = Matrix3::Identity();
  RightLegMat.col(1) -= 4000 * TorsoPose.R.transpose() * POFMeasurements.col(4);
  projectToSO3(RightLegMat, RightLegJoint[0]);

  Scalar RightKneeAngle =
      acos(POFMeasurements.col(4).dot(POFMeasurements.col(5)));
  RightLegJoint[1].setIdentity();
  RightLegJoint[1](1, 1) = cos(RightKneeAngle);
  RightLegJoint[1](2, 1) = sin(RightKneeAngle);
  RightLegJoint[1](2, 2) = RightLegJoint[1](1, 1);
  RightLegJoint[1](1, 2) = -RightLegJoint[1](2, 1);

  // create factors
  const auto &RightLegIndex = InitialInfo::SMPL::RightLegIndex[1];
  const auto &RightLegKintree = InitialInfo::SMPL::RightLegKinematicsTree;
  const auto &RightLegPinholeCamInfo =
      InitialInfo::SMPL::RightLegPinholeCameraFactorInfo[1];
  const auto &RightLegPOFInfo = InitialInfo::SMPL::RightLegPOFFactorInfo[1];
  const auto &RightLegDepthCamInfo =
      InitialInfo::SMPL::RightLegDepthCameraFactorInfo[1];
  const auto &RightLegEulerAngleConstInfo =
      InitialInfo::SMPL::RightLegEulerAngleConstInfo;

  // pinhole camera factors
  std::vector<std::shared_ptr<Initializer::PinholeCameraFactor>>
      RightLegPinholeCamFactors;
  createInitialPinholeCameraFactors(
      RightLegPinholeCamInfo, RightLegIndex, KeyPoints, KeyPoint2DGMSigma,
      KeyPoint2DGMEps, Measurements2D, KeyPoint2DConfidenceThreshold,
      RightLegPinholeCamFactors);

  // POF factors
  std::vector<std::shared_ptr<Initializer::POFFactor>> RightLegPOFFactors;
  createInitialPOFFactors(RightLegPOFInfo, POFGMSigma, POFGMEps,
                          POFMeasurements, RightLegPOFFactors);

  // depth camera factors
  std::vector<std::shared_ptr<Initializer::DepthCameraFactor>>
      RightLegDepthCamFactors;
  createInitialDepthCameraFactors(
      RightLegDepthCamInfo, RightLegIndex, KeyPoints, KeyPoint3DGMSigma,
      KeyPoint3DGMEps, Measurements3D, TorsoPose.t, RightLegDepthCamFactors);

  // pose factors
  Matrix6 RightLegPoseSqrtCov = Matrix6::Zero();
  RightLegPoseSqrtCov.diagonal().array() = 20;
  std::shared_ptr<Initializer::PoseFactor> RightLegPoseFactor =
      std::make_shared<Initializer::PoseFactor>(0, RightLegPoseSqrtCov,
                                                TorsoPose);

  // joint constraint factors
  std::vector<std::shared_ptr<Initializer::EulerAngleConstFactor>>
      RightLegEulerAngleConstFactors;

  createInitialEulerAngleConstFactors(RightLegEulerAngleConstInfo,
                                      RightLegEulerAngleConstFactors);

  // right leg initializer
  std::array<Vector3, 2> RightLegRelJointLocations;

  for (int i = 0; i < RightLegKintree.size(); i++) {
    RightLegRelJointLocations[i] =
        KeyPoints.segment<3>(3 * RightLegIndex[RightLegKintree[i][1]][1]) -
        KeyPoints.segment<3>(3 * RightLegIndex[RightLegKintree[i][0]][1]);
  }

  Initializer::LegInitializer RightLegInitializer(options,
                                                  RightLegRelJointLocations);

  for (auto &factor : RightLegPinholeCamFactors) {
    RightLegInitializer.addPinholeCameraFactor(factor);
  }

  for (auto &factor : RightLegPOFFactors) {
    RightLegInitializer.addPOFFactor(factor);
  }

  for (auto &factor : RightLegDepthCamFactors) {
    RightLegInitializer.addDepthCameraFactor(factor);
  }

  RightLegInitializer.addPoseFactor(RightLegPoseFactor);

  for (auto &factor : RightLegEulerAngleConstFactors) {
    RightLegInitializer.addJointConstFactor(factor);
  }

  RightLegInitializer.updateFactorWeights(options.weights);
  RightLegInitializer.initialize(TorsoPose, RightLegJoint);
  RightLegInitializer.solve();

  RightLegJoint = RightLegInitializer.getJoints();

  //------------------------------------------------------------------------------
  // Left Arm Initializer
  //------------------------------------------------------------------------------
  LeftArmJoint.resize(2);
  Matrix3 LeftArmMat = Matrix3::Identity();
  LeftArmMat.col(0) += 4000 * ChestPose.R.transpose() * POFMeasurements.col(11);
  projectToSO3(LeftArmMat, LeftArmJoint[0]);

  Scalar LeftElbowAngle =
      -acos(POFMeasurements.col(11).dot(POFMeasurements.col(12)));
  LeftArmJoint[1].setIdentity();
  LeftArmJoint[1](0, 0) = cos(LeftElbowAngle);
  LeftArmJoint[1](0, 2) = sin(LeftElbowAngle);
  LeftArmJoint[1](2, 2) = LeftArmJoint[1](0, 0);
  LeftArmJoint[1](2, 0) = -LeftArmJoint[1](0, 2);

  // create factors
  const auto &LeftArmIndex = InitialInfo::SMPL::LeftArmIndex[1];
  const auto &LeftArmKintree = InitialInfo::SMPL::LeftArmKinematicsTree;
  const auto &LeftArmPinholeCamInfo =
      InitialInfo::SMPL::LeftArmPinholeCameraFactorInfo[1];
  const auto &LeftArmPOFInfo = InitialInfo::SMPL::LeftArmPOFFactorInfo[1];
  const auto &LeftArmDepthCamInfo =
      InitialInfo::SMPL::LeftArmDepthCameraFactorInfo[1];
  const auto &LeftArmEulerAngleConstInfo =
      InitialInfo::SMPL::LeftArmEulerAngleConstInfo;

  // pinhole camera factors
  std::vector<std::shared_ptr<Initializer::PinholeCameraFactor>>
      LeftArmPinholeCamFactors;
  createInitialPinholeCameraFactors(
      LeftArmPinholeCamInfo, LeftArmIndex, KeyPoints, KeyPoint2DGMSigma,
      KeyPoint2DGMEps, Measurements2D, KeyPoint2DConfidenceThreshold,
      LeftArmPinholeCamFactors);

  // POF factors
  std::vector<std::shared_ptr<Initializer::POFFactor>> LeftArmPOFFactors;
  createInitialPOFFactors(LeftArmPOFInfo, POFGMSigma, POFGMEps, POFMeasurements,
                          LeftArmPOFFactors);

  // depth camera factors
  std::vector<std::shared_ptr<Initializer::DepthCameraFactor>>
      LeftArmDepthCamFactors;
  createInitialDepthCameraFactors(
      LeftArmDepthCamInfo, LeftArmIndex, KeyPoints, KeyPoint3DGMSigma,
      KeyPoint3DGMEps, Measurements3D, TorsoPose.t, LeftArmDepthCamFactors);

  // joint constraint factors
  std::vector<std::shared_ptr<Initializer::EulerAngleConstFactor>>
      LeftArmEulerAngleConstFactors;

  createInitialEulerAngleConstFactors(LeftArmEulerAngleConstInfo,
                                      LeftArmEulerAngleConstFactors);

  // left arm joint limit factors
  std::vector<std::shared_ptr<Initializer::JointLimitFactor>>
      LeftArmJointLimitFactors;

  createInitialJointLimitFactors(LeftArmIndex, JointLimitWeight, JointLimitBias,
                                 JointLimitPRelua, JointLimitPRelub,
                                 JointLimitScale, LeftArmJointLimitFactors);

  // left arm initializer
  std::array<Vector3, 2> LeftArmRelJointLocations;

  for (int i = 0; i < LeftArmKintree.size(); i++) {
    LeftArmRelJointLocations[i] =
        KeyPoints.segment<3>(3 * LeftArmIndex[LeftArmKintree[i][1]][1]) -
        KeyPoints.segment<3>(3 * LeftArmIndex[LeftArmKintree[i][0]][1]);
  }

  Initializer::ArmInitializer LeftArmInitializer(
      options, LeftArmRelJointLocations,
      Initializer::ArmInitializer::Arm::Left);

  for (auto &factor : LeftArmPinholeCamFactors) {
    LeftArmInitializer.addPinholeCameraFactor(factor);
  }

  for (auto &factor : LeftArmPOFFactors) {
    LeftArmInitializer.addPOFFactor(factor);
  }

  for (auto &factor : LeftArmDepthCamFactors) {
    LeftArmInitializer.addDepthCameraFactor(factor);
  }

  for (auto &factor : LeftArmEulerAngleConstFactors) {
    LeftArmInitializer.addJointConstFactor(factor);
  }

  options.weights[Initializer::FactorIndex::JointLimit] = 1000;
  LeftArmInitializer.updateFactorWeights(options.weights);

  LeftArmInitializer.initialize(ChestPose, LeftArmJoint);
  LeftArmInitializer.solve();

  options.weights[Initializer::FactorIndex::JointLimit] = 250;
  LeftArmInitializer.updateFactorWeights(options.weights);
  LeftArmInitializer.solve();

  LeftArmJoint = LeftArmInitializer.getJoints();

  //------------------------------------------------------------------------------
  // Right Arm Initializer
  //------------------------------------------------------------------------------
  RightArmJoint.resize(2);
  Matrix3 RightArmMat = Matrix3::Identity();
  RightArmMat.col(0) -=
      4000 * ChestPose.R.transpose() * POFMeasurements.col(14);
  projectToSO3(RightArmMat, RightArmJoint[0]);

  Scalar RightElbowAngle =
      acos(POFMeasurements.col(14).dot(POFMeasurements.col(15)));
  RightArmJoint[1].setIdentity();
  RightArmJoint[1](0, 0) = cos(RightElbowAngle);
  RightArmJoint[1](0, 2) = sin(RightElbowAngle);
  RightArmJoint[1](2, 2) = RightArmJoint[1](0, 0);
  RightArmJoint[1](2, 0) = -RightArmJoint[1](0, 2);

  // create factors
  const auto &RightArmIndex = InitialInfo::SMPL::RightArmIndex[1];
  const auto &RightArmKintree = InitialInfo::SMPL::RightArmKinematicsTree;
  const auto &RightArmPinholeCamInfo =
      InitialInfo::SMPL::RightArmPinholeCameraFactorInfo[1];
  const auto &RightArmPOFInfo = InitialInfo::SMPL::RightArmPOFFactorInfo[1];
  const auto &RightArmDepthCamInfo =
      InitialInfo::SMPL::RightArmDepthCameraFactorInfo[1];
  const auto &RightArmEulerAngleConstInfo =
      InitialInfo::SMPL::RightArmEulerAngleConstInfo;

  // pinhole camera factors
  std::vector<std::shared_ptr<Initializer::PinholeCameraFactor>>
      RightArmPinholeCamFactors;
  createInitialPinholeCameraFactors(
      RightArmPinholeCamInfo, RightArmIndex, KeyPoints, KeyPoint2DGMSigma,
      KeyPoint2DGMEps, Measurements2D, KeyPoint2DConfidenceThreshold,
      RightArmPinholeCamFactors);

  // POF factors
  std::vector<std::shared_ptr<Initializer::POFFactor>> RightArmPOFFactors;
  createInitialPOFFactors(RightArmPOFInfo, POFGMSigma, POFGMEps,
                          POFMeasurements, RightArmPOFFactors);

  // depth camera factors
  std::vector<std::shared_ptr<Initializer::DepthCameraFactor>>
      RightArmDepthCamFactors;
  createInitialDepthCameraFactors(
      RightArmDepthCamInfo, RightArmIndex, KeyPoints, KeyPoint3DGMSigma,
      KeyPoint3DGMEps, Measurements3D, TorsoPose.t, RightArmDepthCamFactors);

  // joint constraint factors
  std::vector<std::shared_ptr<Initializer::EulerAngleConstFactor>>
      RightArmEulerAngleConstFactors;

  createInitialEulerAngleConstFactors(RightArmEulerAngleConstInfo,
                                      RightArmEulerAngleConstFactors);

  // left arm joint limit factors
  std::vector<std::shared_ptr<Initializer::JointLimitFactor>>
      RightArmJointLimitFactors;

  createInitialJointLimitFactors(
      RightArmIndex, JointLimitWeight, JointLimitBias, JointLimitPRelua,
      JointLimitPRelub, JointLimitScale, RightArmJointLimitFactors);

  // right arm initializer
  std::array<Vector3, 2> RightArmRelJointLocations;

  for (int i = 0; i < RightArmKintree.size(); i++) {
    RightArmRelJointLocations[i] =
        KeyPoints.segment<3>(3 * RightArmIndex[RightArmKintree[i][1]][1]) -
        KeyPoints.segment<3>(3 * RightArmIndex[RightArmKintree[i][0]][1]);
  }

  Initializer::ArmInitializer RightArmInitializer(
      options, RightArmRelJointLocations,
      Initializer::ArmInitializer::Arm::Right);

  for (auto &factor : RightArmPinholeCamFactors) {
    RightArmInitializer.addPinholeCameraFactor(factor);
  }

  for (auto &factor : RightArmPOFFactors) {
    RightArmInitializer.addPOFFactor(factor);
  }

  for (auto &factor : RightArmDepthCamFactors) {
    RightArmInitializer.addDepthCameraFactor(factor);
  }

  for (auto &factor : RightArmEulerAngleConstFactors) {
    RightArmInitializer.addJointConstFactor(factor);
  }

  options.weights[Initializer::FactorIndex::JointLimit] = 0;
  RightArmInitializer.updateFactorWeights(options.weights);

  RightArmInitializer.initialize(ChestPose, RightArmJoint);
  RightArmInitializer.solve();

  options.weights[Initializer::FactorIndex::JointLimit] = 0;
  RightArmInitializer.solve();

  RightArmJoint = RightArmInitializer.getJoints();

  // leg knee joints correction
  const static Matrix3 KneeJointCorrection =
      (Matrix3() << 1, 0, 0, 0, cos(-0.075), -sin(-0.075), 0, sin(-0.075),
       cos(-0.075))
          .finished();
  LeftLegJoint[1] = LeftLegJoint[1] * KneeJointCorrection;
  RightLegJoint[1] = RightLegJoint[1] * KneeJointCorrection;

  return 0;
}

int initializeRootPose(const AlignedVector<Pose> &Poses, const VectorX &Param,
                       // SMPL model info
                       const std::vector<int> &kintree, const MatrixX &JDirs,
                       const VectorX &J, const MatrixX &KeyPointDirs,
                       const VectorX &KeyPoints,
                       // KeyPoint2D setup
                       Scalar FocalLength, const Matrix3X &Measurements2D,
                       Scalar KeyPoint2DGMSigma, Scalar KeyPoint2DGMEps,
                       const VectorX &KeyPoint2DConfidenceThreshold,
                       // POF setup
                       const Matrix3X &POFMeasurements, Scalar POFGMSigma,
                       Scalar POFGMEps,
                       // initial estimates
                       Pose &RootPose) {
  assert(Poses.size() == 24);
  assert(Param.size() == JDirs.cols());
  assert(Param.size() == KeyPointDirs.cols());

  if (Poses.size() != 24 || Param.size() != JDirs.cols() ||
      Param.size() != KeyPointDirs.cols()) {
    LOG(ERROR) << "Inconsistent keypoint information." << std::endl;
    exit(-1);
  }

  std::vector<std::shared_ptr<Initializer::PinholeCameraFactor>> RootCamFactors;

  createInitialRootPinholeCameraFactors(
      KeyPointInfo::SMPL::JointPinholeCameraFactorInfo,
      KeyPointInfo::SMPL::VertexPinholeCameraFactorInfo, Poses, Param, JDirs, J,
      KeyPointDirs, KeyPoints, KeyPoint2DGMSigma, KeyPoint2DGMEps,
      Measurements2D, KeyPoint2DConfidenceThreshold, RootCamFactors);

  std::vector<std::shared_ptr<Initializer::POFFactor>> RootPOFFactors;

  createInitialRootPOFFactors(KeyPointInfo::SMPL::UnitPOFFactorInfo,
                              KeyPointInfo::SMPL::ScaledPOFFactorInfo,
                              KeyPointInfo::SMPL::RelPOFFactorInfo, Poses,
                              Param, JDirs, J, KeyPointDirs, KeyPoints, kintree,
                              POFGMSigma, POFGMEps, POFMeasurements,
                              RootPOFFactors);

  Matrix6 RootPoseSqrtCov = Matrix6::Zero();
  RootPoseSqrtCov.diagonal().tail<3>().array() = 5;

  std::shared_ptr<Initializer::PoseFactor> TorsoPoseFactor =
      std::make_shared<Initializer::PoseFactor>(0, RootPoseSqrtCov,
                                                Pose::Identity());

  Initializer::Options options;

  options.rel_lin_func_decrease_tol = 4e-3;
  options.delta = 2;
  options.weights[Initializer::FactorIndex::PinholeCamera] =
      FocalLength * FocalLength;
  options.weights[Initializer::FactorIndex::POF] = 100000;
  options.weights[Initializer::FactorIndex::DepthCamera] = 1000;
  options.weights[Initializer::FactorIndex::JointLimit] = 1000;
  options.weights[Initializer::FactorIndex::JointConst] = 1000;
  options.weights[Initializer::FactorIndex::Pose] = 100;

  Initializer::TorsoInitializer RootInitializer(options);

  for (auto & factor: RootCamFactors) {
    RootInitializer.addPinholeCameraFactor(factor);
  }

  for (auto & factor: RootPOFFactors) {
    RootInitializer.addPOFFactor(factor);
  }

  RootInitializer.addPoseFactor(TorsoPoseFactor);

  RootInitializer.initialize(Pose::Identity(), {});
  RootInitializer.solve();

  RootPose.R.noalias() = RootInitializer.getPoses()[0].R * Poses[0].R;
  RootPose.t = RootInitializer.getPoses()[0].t;
  RootPose.t.noalias() += RootInitializer.getPoses()[0].R * Poses[0].t;

  return 0;
}
}  // namespace SMPL
}  // namespace Initialization
}  // namespace scope

