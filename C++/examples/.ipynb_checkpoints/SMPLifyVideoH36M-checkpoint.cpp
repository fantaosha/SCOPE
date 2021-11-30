#include <cnpy/cnpy.h>
#include <scope/optimizer/SMPLify.h>
#include <scope/utils/Initialization.h>
#include <scope/utils/JointConstInfo.h>
#include <scope/utils/KeyPointInfo.h>
#include <scope/utils/Stopwatch.h>

#include <iomanip>
#include <memory>
#include <ostream>

int main(int argc, char **argv) {
  if (argc <= 3) {
    std::cout << "There must be at least 3 inputs." << std::endl;
    return -1;
  }

  std::string model = std::string(argv[1]);
  std::string seq = std::string(argv[2]);
  int start = std::stoi(argv[3]);
  int count = std::stoi(argv[4]);
  int end = start + count;

  const int ParamSize = 11;

  using SMPL = scope::SMPL<ParamSize>;
  using SMPLify = scope::Optimizer::SMPLify<ParamSize, false>;

  const int NumJoints = SMPL::NumJoints;

  const std::string KeyPoint2DHeader =
      "/home/taoshaf/Documents/C++/SCOPE/data/Human3.6M/KeyPoints2D/" + seq +
      "/";

  const std::string KeyPoint3DHeader =
      "/home/taoshaf/Documents/C++/SCOPE/data/Human3.6M/KeyPoints3D/" + seq +
      "/";

  const std::string JointLimitHeader =
      "/home/taoshaf/Documents/C++/SCOPE/model/JointLimitNeuralNetworks/";

  const std::string ResultDir =
      "/home/taoshaf/Documents/C++/SCOPE/results/Human3.6M/" + seq + "/";

  //------------------------------------------------------------------------------
  // Load model info
  //------------------------------------------------------------------------------
  scope::MatrixX SMPLJDirs, SMPLRelJDirs;
  scope::VectorX SMPLJ, SMPLRelJ;
  scope::MatrixX JDirs, RelJDirs, VDirs;
  scope::VectorX J, RelJ, V;

  scope::loadModel<23, ParamSize, 6890>(model, SMPLJDirs, SMPLJ, SMPLRelJDirs,
                                        SMPLRelJ, VDirs, V);

  JDirs = SMPLJDirs.topRows<72>();

  J = SMPLJ.head<72>();

  RelJDirs = SMPLRelJDirs.topRows<69>();

  RelJ = SMPLRelJ.topRows<69>();

  scope::MatrixX ShapeSqrtCov = scope::MatrixX::Identity(ParamSize, ParamSize);
  ShapeSqrtCov.diagonal().segment<5>(5).array() = 0.8;
  ShapeSqrtCov.diagonal()[ParamSize - 1] = 0;

  //------------------------------------------------------------------------------
  // setup model
  //------------------------------------------------------------------------------
  SMPL smpl(RelJDirs, RelJ, VDirs, V);

  scope::MatrixX KeyPointDirs;
  scope::VectorX KeyPoints;

  scope::loadSMPLKeyPoints(JDirs, J, VDirs, V, KeyPointDirs, KeyPoints);

  //------------------------------------------------------------------------------
  // Load joint limits neural networks info
  //------------------------------------------------------------------------------
  scope::AlignedVector<scope::AlignedVector<scope::Matrix6>> JointLimitWeight(
      SMPL::NumJoints);
  scope::AlignedVector<scope::AlignedVector<scope::Vector6>> JointLimitBias(
      SMPL::NumJoints);
  scope::AlignedVector<scope::VectorX> JointLimitPRelua(SMPL::NumJoints);
  scope::AlignedVector<scope::VectorX> JointLimitPRelub(SMPL::NumJoints);

  for (int n = 0; n < SMPL::NumJoints - 2; n++) {
    std::string file =
        JointLimitHeader + "joint_flow_" + std::to_string(n) + ".npz";
    scope::loadJointLimitNeuralNetwork(file, JointLimitWeight[n],
                                       JointLimitBias[n], JointLimitPRelua[n],
                                       JointLimitPRelub[n]);
  }

  {
    int n = 21;

    std::string file =
        JointLimitHeader + "joint_flow_" + std::to_string(24) + ".npz";
    scope::loadJointLimitNeuralNetwork(file, JointLimitWeight[n],
                                       JointLimitBias[n], JointLimitPRelua[n],
                                       JointLimitPRelub[n]);
  }

  {
    int n = 22;

    std::string file =
        JointLimitHeader + "joint_flow_" + std::to_string(39) + ".npz";
    scope::loadJointLimitNeuralNetwork(file, JointLimitWeight[n],
                                       JointLimitBias[n], JointLimitPRelua[n],
                                       JointLimitPRelub[n]);
  }

  //------------------------------------------------------------------------------
  // Setup camera info
  //------------------------------------------------------------------------------
  std::map<std::string, scope::Vector4> CamParamInfo = {
      {"54138969",
       {1145.04940458804, 1143.78109572365, 512.541504956548,
        515.451486977600}},
      {"55011271",
       {1149.67569986785, 1147.59161666764, 508.848621645943,
        508.064917088557}},
      {"58860488",
       {1149.14071676148, 1148.79896856760, 519.815837182153,
        501.402658888552}},
      {"60457274",
       {1145.51133842318, 1144.77392807652, 514.968197319863,
        501.882018537695}}};

  std::string cam = seq.substr(seq.length() - 8);
  scope::Vector4 CamParams = CamParamInfo.at(cam);

  scope::Scalar meanCamParam = CamParams.head<2>().mean();

  //------------------------------------------------------------------------------
  // Setup KeyPoint2D, POF, KeyPoint3D, JointLimit and Pose parameters
  //------------------------------------------------------------------------------
  scope::Scalar KeyPoint2DGMSigma = 0.01;
  scope::Scalar KeyPoint2DGMEps = 0.5;

  scope::VectorX KeyPoint2DConfidenceThreshold = 0.3 * scope::VectorX::Ones(28);
  KeyPoint2DConfidenceThreshold.head<3>().array() = 0.9;
  KeyPoint2DConfidenceThreshold.segment<2>(3).array() = 0.85;
  KeyPoint2DConfidenceThreshold.segment<6>(20).array() = 0.85;

  scope::Scalar POFGMSigma = 0.1;
  scope::Scalar POFGMEps = 0.8;

  scope::Scalar KeyPoint3DGMSigma = 0.01;
  scope::Scalar KeyPoint3DGMEps = 0.5;

  scope::Scalar KeyPoint3DOriginGMSigma = 100;
  scope::Scalar KeyPoint3DOriginGMEps = 0.5;

  scope::Vector<SMPL::NumJoints> InitJointLimitScale;
  InitJointLimitScale.setOnes();
  InitJointLimitScale[0] = 0.05;
  InitJointLimitScale[1] = 0.05;
  InitJointLimitScale[2] = 1.0;
  InitJointLimitScale[3] = 0.05;
  InitJointLimitScale[4] = 0.05;
  InitJointLimitScale[5] = 1.0;
  InitJointLimitScale[11] = 0.2;
  InitJointLimitScale[14] = 0.2;
  InitJointLimitScale[15] = 1;
  InitJointLimitScale[16] = 1;

  scope::Vector<SMPL::NumJoints> JointLimitScale;
  JointLimitScale.setOnes();
  JointLimitScale[0] = 0.05;
  JointLimitScale[1] = 0.05;
  JointLimitScale[2] = 1.0;
  JointLimitScale[3] = 0.2;
  JointLimitScale[4] = 0.2;
  JointLimitScale[5] = 1.0;
  JointLimitScale[8] = 1.0;
  JointLimitScale[11] = 0.1;
  JointLimitScale[14] = 0.1;
  JointLimitScale[15] = 1;
  JointLimitScale[16] = 1;

  scope::Matrix6 SqrtCovPose = scope::Matrix6::Identity();
  SqrtCovPose.diagonal().head<3>().array() = 0;

  //------------------------------------------------------------------------------
  // Setup joint constraint, parameter and joint factors
  //------------------------------------------------------------------------------
  // create joint constraint factors
  const auto &JointConstInfo = scope::JointConstInfo::SMPL::JointConstInfo;
  std::vector<std::shared_ptr<scope::JointConstFactor>> JointConstFactors(
      JointConstInfo.size());

  scope::createJointConstFactors(JointConstInfo, JointConstFactors);

  // create Euler angle constraint factors
  const auto &EulerAngleConstInfo =
      scope::JointConstInfo::SMPL::EulerAngleConstInfo;
  std::vector<std::shared_ptr<scope::EulerAngleConstFactor>>
      EulerAngleConstFactors(EulerAngleConstInfo.size());

  scope::createEulerAngleConstFactors(EulerAngleConstInfo,
                                      EulerAngleConstFactors);

  // create joint limits factors
  std::vector<std::shared_ptr<scope::JointLimitFactor>> JointLimitFactors(
      SMPL::NumJoints);

  scope::createJointLimitFactors(JointLimitWeight, JointLimitBias,
                                 JointLimitPRelua, JointLimitPRelub,
                                 JointLimitScale, JointLimitFactors);

  //------------------------------------------------------------------------------
  // Measurements
  //------------------------------------------------------------------------------
  const int KeyPoint2DSize = 28;
  scope::Matrix3X Measurements2D(3, KeyPoint2DSize);

  const int KeyPoint3DSize = 17;
  scope::Matrix3X Measurements3D(3, KeyPoint3DSize);

  const int POFSize = 16;
  scope::Matrix3X POFMeasurements(3, POFSize);

  for (int index = start; index < end; index++) {
    std::ostringstream ss;
    ss << std::setw(6) << std::setfill('0') << 5 * index + 1;

    //------------------------------------------------------------------------------
    // Load 2D keypoint measurements
    //------------------------------------------------------------------------------
    std::string KeyPoint2DResult = KeyPoint2DHeader + ss.str() + ".npy";

    scope::loadKeyPoint2DMeasurements(KeyPoint2DResult, Measurements2D);

    Measurements2D.topRows<2>().colwise() -= CamParams.tail<2>();
    Measurements2D.row(0) /= CamParams[0];
    Measurements2D.row(1) /= CamParams[1];

    //------------------------------------------------------------------------------
    // Load 3D keypoint measurements
    //------------------------------------------------------------------------------
    std::string KeyPoint3DResult = KeyPoint3DHeader + ss.str() + ".npy";

    scope::loadKeyPoint3DMeasurements(KeyPoint3DResult, Measurements3D);

    //------------------------------------------------------------------------------
    // Load POF measurements
    //------------------------------------------------------------------------------
    scope::loadPOFMeasurements(Measurements3D, POFMeasurements);

    //******************************************************************************
    // Initialization
    //******************************************************************************
    scope::Pose InitialPose;
    scope::Pose TorsoPose;
    scope::AlignedVector<scope::Matrix3> LeftArmJoints;
    scope::AlignedVector<scope::Matrix3> RightArmJoints;
    scope::AlignedVector<scope::Matrix3> LeftLegJoints;
    scope::AlignedVector<scope::Matrix3> RightLegJoints;

    scope::VectorX betas;
    scope::Matrix3 SpineJoint;

    auto InitStart = scope::Stopwatch::tick();

    scope::Initialization::SMPL::initialize(
        ShapeSqrtCov, meanCamParam, Measurements2D, KeyPoint2DGMSigma,
        KeyPoint2DGMEps, KeyPoint2DConfidenceThreshold, POFMeasurements,
        POFGMSigma, POFGMEps, Measurements3D, KeyPoint3DGMSigma,
        KeyPoint3DGMEps, JointLimitWeight, JointLimitBias, JointLimitPRelua,
        JointLimitPRelub, InitJointLimitScale, TorsoPose, SpineJoint,
        LeftArmJoints, RightArmJoints, LeftLegJoints, RightLegJoints, betas);

    auto InitTime = scope::Stopwatch::tock(InitStart);

    const auto &ExtTorsoIndex = scope::InitialInfo::SMPL::ExtTorsoIndex[1];
    const auto &LeftArmIndex = scope::InitialInfo::SMPL::LeftArmIndex[1];
    const auto &RightArmIndex = scope::InitialInfo::SMPL::RightArmIndex[1];
    const auto &LeftLegIndex = scope::InitialInfo::SMPL::LeftLegIndex[1];
    const auto &RightLegIndex = scope::InitialInfo::SMPL::RightLegIndex[1];

    // Initialize joints
    scope::AlignedVector<scope::Matrix3> joints(NumJoints,
                                                scope::Matrix3::Identity());

    joints[ExtTorsoIndex[1][0] - 1] = SpineJoint;

    joints[LeftArmIndex[1][0] - 1] = LeftArmJoints[0];
    joints[LeftArmIndex[2][0] - 1] = LeftArmJoints[1];

    joints[RightArmIndex[1][0] - 1] = RightArmJoints[0];
    joints[RightArmIndex[2][0] - 1] = RightArmJoints[1];

    joints[LeftLegIndex[1][0] - 1] = LeftLegJoints[0];
    joints[LeftLegIndex[2][0] - 1] = LeftLegJoints[1];

    joints[RightLegIndex[1][0] - 1] = RightLegJoints[0];
    joints[RightLegIndex[2][0] - 1] = RightLegJoints[1];

    // Initialize params
    scope::Vector<ParamSize> params = scope::Vector<ParamSize>::Zero();

    // Initialize root params
    const auto &KeyPointOriginIndex =
        scope::KeyPointInfo::SMPL::KeyPointOriginIndex;
    auto root = TorsoPose;
    scope::Vector3 origin;
    scope::getKeyPointOrigin3D(TorsoPose, JDirs, J, KeyPointDirs, KeyPoints,
                               params, KeyPointOriginIndex, origin);
    root.t -= origin - TorsoPose.t;

    // Initialize root
    {
      scope::AlignedVector<scope::Pose> poses;
      scope::VectorX JointLocationData(69);
      Eigen::Map<scope::Vector<69>> JointLocation(JointLocationData.data());

      smpl.FK(poses, JointLocation, root, joints, params);

      scope::Initialization::SMPL::initializeRootPose(
          poses, params, smpl.getKinematicsTree(), JDirs, J, KeyPointDirs,
          KeyPoints, meanCamParam, Measurements2D, KeyPoint2DGMSigma,
          KeyPoint2DGMEps, KeyPoint2DConfidenceThreshold, POFMeasurements,
          POFGMSigma, POFGMEps, root);
    }

    //******************************************************************************
    // Optimization
    //******************************************************************************
    auto SetupStart = scope::Stopwatch::tick();

    //------------------------------------------------------------------------------
    // Setup JointPinholeCamera Factors
    //------------------------------------------------------------------------------
    const auto &JointPinholeCamInfo =
        scope::KeyPointInfo::SMPL::JointPinholeCameraFactorInfo;
    std::vector<std::shared_ptr<scope::JointPinholeCameraFactor>>
        JointPinholeCamFactors;

    scope::createJointPinholeCameraFactors(
        JointPinholeCamInfo, KeyPoint2DGMSigma, KeyPoint2DGMEps, Measurements2D,
        KeyPoint2DConfidenceThreshold, JointPinholeCamFactors);

    //------------------------------------------------------------------------------
    // Setup VertexPinholeCamera Factors
    //------------------------------------------------------------------------------
    const auto &VertexPinholeCamInfo =
        scope::KeyPointInfo::SMPL::VertexPinholeCameraFactorInfo;
    std::vector<std::shared_ptr<scope::VertexPinholeCameraFactor>>
        VertexPinholeCamFactors;

    scope::createVertexPinholeCameraFactors(
        VertexPinholeCamInfo, 0, JDirs, J, KeyPointDirs, KeyPoints,
        KeyPoint2DGMSigma, KeyPoint2DGMEps, Measurements2D,
        KeyPoint2DConfidenceThreshold, VertexPinholeCamFactors);

    //------------------------------------------------------------------------------
    // Setup UnitPOF Factors
    //------------------------------------------------------------------------------
    const auto &UnitPOFInfo = scope::KeyPointInfo::SMPL::UnitPOFFactorInfo;
    std::vector<std::shared_ptr<scope::UnitPOFFactor>> UnitPOFFactors;

    scope::createUnitPOFFactors(UnitPOFInfo, POFGMSigma, POFGMEps,
                                POFMeasurements, UnitPOFFactors);

    //------------------------------------------------------------------------------
    // Setup ScaledPOF Factors
    //------------------------------------------------------------------------------
    const auto &ScaledPOFInfo = scope::KeyPointInfo::SMPL::ScaledPOFFactorInfo;
    std::vector<std::shared_ptr<scope::ScaledPOFFactor>> ScaledPOFFactors;

    scope::createScaledPOFFactors(ScaledPOFInfo, 0, KeyPointDirs, KeyPoints,
                                  POFGMSigma, POFGMEps, POFMeasurements,
                                  ScaledPOFFactors);

    //------------------------------------------------------------------------------
    // Setup RelPOF Factors
    //------------------------------------------------------------------------------
    const auto &RelPOFInfo = scope::KeyPointInfo::SMPL::RelPOFFactorInfo;
    const auto &kintree = smpl.getKinematicsTree();
    std::vector<std::shared_ptr<scope::RelPOFFactor>> RelPOFFactors;

    scope::createRelPOFFactors(RelPOFInfo, 0, JDirs, J, KeyPointDirs, KeyPoints,
                               kintree, POFGMSigma, POFGMEps, POFMeasurements,
                               RelPOFFactors);

    //------------------------------------------------------------------------------
    // Setup Joint Factors
    //------------------------------------------------------------------------------
    scope::Matrix3 JointSqrtCov = scope::Matrix3::Identity();

    std::vector<std::shared_ptr<scope::JointFactor>> JointFactors;

    JointFactors.push_back(std::make_shared<scope::JointFactor>(
        LeftLegIndex[2][0] - 1, JointSqrtCov, LeftLegJoints[1]));
    JointFactors.push_back(std::make_shared<scope::JointFactor>(
        RightLegIndex[2][0] - 1, JointSqrtCov, RightLegJoints[1]));

    //------------------------------------------------------------------------------
    // Setup JointDepthCamera factors
    //------------------------------------------------------------------------------
    const auto &JointDepthCamInfo =
        scope::KeyPointInfo::SMPL::JointDepthCameraFactorInfo;
    std::vector<std::shared_ptr<scope::JointDepthCameraFactor>>
        JointDepthCamFactors;

    scope::createJointDepthCameraFactors(JointDepthCamInfo, KeyPoint3DGMSigma,
                                         KeyPoint3DGMEps, Measurements3D,
                                         TorsoPose.t, JointDepthCamFactors);

    //------------------------------------------------------------------------------
    // Setup VertexDepthCamera factors
    //------------------------------------------------------------------------------
    const auto &VertexDepthCamInfo =
        scope::KeyPointInfo::SMPL::VertexDepthCameraFactorInfo;
    std::vector<std::shared_ptr<scope::VertexDepthCameraFactor>>
        VertexDepthCamFactors;

    scope::createVertexDepthCameraFactors(
        VertexDepthCamInfo, 0, JDirs, J, KeyPointDirs, KeyPoints,
        KeyPoint3DGMSigma, KeyPoint3DGMEps, Measurements3D, TorsoPose.t,
        VertexDepthCamFactors);

    //------------------------------------------------------------------------------
    // Setup OriginDepthCamera factors
    //------------------------------------------------------------------------------
    const auto &OriginDepthCamInfo =
        scope::KeyPointInfo::SMPL::OriginDepthCameraFactorInfo;
    std::vector<std::shared_ptr<scope::VertexDepthCameraFactor>>
        OriginDepthCamFactors;

    scope::createVertexDepthCameraFactors(
        OriginDepthCamInfo, 0, JDirs, J, KeyPointDirs, KeyPoints,
        KeyPoint3DOriginGMSigma, KeyPoint3DOriginGMEps, Measurements3D,
        TorsoPose.t, OriginDepthCamFactors);

    //------------------------------------------------------------------------------
    // Setup pose factor
    //------------------------------------------------------------------------------
    scope::Matrix6 PoseSqrtCov = scope::Matrix6::Identity();
    PoseSqrtCov.diagonal().head<3>().setConstant(2);

    std::shared_ptr<scope::PoseFactor> PoseFactor =
        std::make_shared<scope::PoseFactor>(0, PoseSqrtCov, root);

    //------------------------------------------------------------------------------
    // Setup parameter factor
    //------------------------------------------------------------------------------
    std::shared_ptr<scope::ParameterFactor> ParameterFactor =
        std::make_shared<scope::ParameterFactor>(0, ShapeSqrtCov, params);

    auto SetupTime = scope::Stopwatch::tock(SetupStart);

    // ------------------------------------------------------------
    // Set up the optimizer
    // ------------------------------------------------------------
    scope::Optimizer::Options options;

    options.method = scope::Optimizer::Method::LM;
    options.delta = 1;

    SMPLify smplify(smpl, options);

    scope::Pose RootPose;
    scope::AlignedVector<scope::Matrix3> Joints;
    scope::Vector<ParamSize> Params;

    auto OptStart = scope::Stopwatch::tick();

    // Reset optimizer
    smplify.reset();

    // Add factors
    for (const auto &factor : JointPinholeCamFactors) {
      smplify.addJointPinholeCameraFactor(factor);
    }

    for (const auto &factor : VertexPinholeCamFactors) {
      smplify.addVertexPinholeCameraFactor(factor);
    }

    for (const auto &factor : UnitPOFFactors) {
      smplify.addPOFFactor(factor);
    }

    for (const auto &factor : ScaledPOFFactors) {
      smplify.addPOFFactor(factor);
    }

    for (const auto &factor : RelPOFFactors) {
      smplify.addPOFFactor(factor);
    }

    for (const auto &factor : JointConstFactors) {
      smplify.addJointConstFactor(factor);
    }

    for (const auto &factor : EulerAngleConstFactors) {
      smplify.addJointConstFactor(factor);
    }

    for (int n = 0; n < NumJoints; n++) {
      smplify.addJointLimitFactor(JointLimitFactors[n]);
    }

    for (const auto &factor : JointFactors) {
      smplify.addJointFactor(factor);
    }

    smplify.addParameterFactor(ParameterFactor);

    smplify.addPoseFactor(PoseFactor);

    //---------------------------------------------------------------
    // Stage 0: Initialization
    //---------------------------------------------------------------
    options.weights.head<4>().array() = std::pow(meanCamParam, 2);
    options.weights[scope::Optimizer::FactorIndex::UnitPOF] = 10000;
    options.weights[scope::Optimizer::FactorIndex::ScaledPOF] = 10000;
    options.weights[scope::Optimizer::FactorIndex::RelPOF] = 10000;
    options.weights[scope::Optimizer::FactorIndex::JointDepthCamera] = 32000;
    options.weights[scope::Optimizer::FactorIndex::VertexDepthCamera] = 32000;
    options.weights[scope::Optimizer::FactorIndex::JointLimit] = 1000;
    options.weights[scope::Optimizer::FactorIndex::JointConst] = 1000;
    options.weights[scope::Optimizer::FactorIndex::Joint] = 10000;
    options.weights[scope::Optimizer::FactorIndex::Pose] = 500;
    options.weights[scope::Optimizer::FactorIndex::Parameter] = 1000;

    smplify.updateFactorWeights(options.weights);

    // Initialize joints
    // Initialize parameters
    smplify.initialize(root, joints, {params}, false);

    int NumIters = 0;
    //---------------------------------------------------------------
    // Stage 1
    //---------------------------------------------------------------
    for (int iter = 0; iter < 50; iter++) {
      smplify.optimize();

      if (smplify.getResults().expected_rel_cost_reduction.back() < 5e-2 ||
          smplify.getResults().rel_cost_reduction.back() < 1e-2) {
        break;
      }
    }

    //---------------------------------------------------------------
    // Stage 2
    //---------------------------------------------------------------
    NumIters += smplify.getResults().fobjs.size() - 1;

    RootPose = smplify.getPoses()[0];
    Joints = smplify.getJoints();
    Params = smplify.getParameters()[0];

    //------------------------------------------------------------------------------
    // Setup pose factor
    //------------------------------------------------------------------------------
    PoseSqrtCov = scope::Matrix6::Identity();
    PoseSqrtCov.diagonal().head<3>().setConstant(2);

    PoseFactor = std::make_shared<scope::PoseFactor>(0, PoseSqrtCov, RootPose);

    //------------------------------------------------------------------------------
    // Setup parameter factor
    //------------------------------------------------------------------------------
    ParameterFactor =
        std::make_shared<scope::ParameterFactor>(0, ShapeSqrtCov, Params);

    //------------------------------------------------------------------------------
    // Get KeyPoint3D origin position
    //------------------------------------------------------------------------------
    scope::getKeyPointOrigin3D(
        RootPose, JDirs, J, KeyPointDirs, KeyPoints, Params,
        scope::KeyPointInfo::SMPL::KeyPointOriginIndex, origin);

    //------------------------------------------------------------------------------
    // Setup JointDepthCamera factors
    //------------------------------------------------------------------------------
    scope::createJointDepthCameraFactors(JointDepthCamInfo, KeyPoint3DGMSigma,
                                         KeyPoint3DGMEps, Measurements3D,
                                         origin, JointDepthCamFactors);

    //------------------------------------------------------------------------------
    // Setup VertexDepthCamera factors
    //------------------------------------------------------------------------------
    scope::createVertexDepthCameraFactors(
        VertexDepthCamInfo, 0, JDirs, J, KeyPointDirs, KeyPoints,
        KeyPoint3DGMSigma, KeyPoint3DGMEps, Measurements3D, origin,
        VertexDepthCamFactors);

    // Reset optimizer
    smplify.reset();

    // Add factors
    for (const auto &factor : JointPinholeCamFactors) {
      smplify.addJointPinholeCameraFactor(factor);
    }

    for (const auto &factor : VertexPinholeCamFactors) {
      smplify.addVertexPinholeCameraFactor(factor);
    }

    for (const auto &factor : UnitPOFFactors) {
      smplify.addPOFFactor(factor);
    }

    for (const auto &factor : ScaledPOFFactors) {
      smplify.addPOFFactor(factor);
    }

    for (const auto &factor : RelPOFFactors) {
      smplify.addPOFFactor(factor);
    }

    for (const auto &factor : JointDepthCamFactors) {
      smplify.addDepthCameraFactor(factor);
    }

    for (const auto &factor : VertexDepthCamFactors) {
      smplify.addDepthCameraFactor(factor);
    }

    for (const auto &factor : JointFactors) {
      smplify.addJointFactor(factor);
    }

    for (const auto &factor : JointConstFactors) {
      smplify.addJointConstFactor(factor);
    }

    for (const auto &factor : EulerAngleConstFactors) {
      smplify.addJointConstFactor(factor);
    }

    for (int n = 0; n < NumJoints; n++) {
      smplify.addJointLimitFactor(JointLimitFactors[n]);
    }

    smplify.addParameterFactor(ParameterFactor);

    smplify.addPoseFactor(PoseFactor);

    options.weights.tail<8>().array() = 10;
    options.weights[scope::Optimizer::FactorIndex::UnitPOF] = 6000;
    options.weights[scope::Optimizer::FactorIndex::ScaledPOF] = 6000;
    options.weights[scope::Optimizer::FactorIndex::RelPOF] = 6000;
    options.weights[scope::Optimizer::FactorIndex::JointDepthCamera] = 24000;
    options.weights[scope::Optimizer::FactorIndex::VertexDepthCamera] = 24000;
    options.weights[scope::Optimizer::FactorIndex::JointLimit] = 500;
    options.weights[scope::Optimizer::FactorIndex::JointConst] = 1000;
    options.weights[scope::Optimizer::FactorIndex::Pose] = 400;
    options.weights[scope::Optimizer::FactorIndex::Joint] = 5000;
    options.weights[scope::Optimizer::FactorIndex::Parameter] = 400;
    smplify.updateFactorWeights(options.weights);

    // initialize the optimizer
    smplify.initialize(RootPose, Joints, {Params}, false);

    for (int iter = 0; iter < 50; iter++) {
      smplify.optimize();

      if (smplify.getResults().expected_rel_cost_reduction.back() < 5e-2 ||
          smplify.getResults().rel_cost_reduction.back() < 1e-2) {
        break;
      }
    }

    //---------------------------------------------------------------
    // Stage 3
    //---------------------------------------------------------------
    NumIters += smplify.getResults().fobjs.size() - 1;

    RootPose = smplify.getPoses()[0];
    Joints = smplify.getJoints();
    Params = smplify.getParameters()[0];

    //------------------------------------------------------------------------------
    // Setup pose factor
    //------------------------------------------------------------------------------
    PoseSqrtCov = scope::Matrix6::Identity();
    PoseSqrtCov.diagonal().head<3>().setConstant(2);

    PoseFactor = std::make_shared<scope::PoseFactor>(0, PoseSqrtCov, RootPose);

    //------------------------------------------------------------------------------
    // Setup parameter factor
    //------------------------------------------------------------------------------
    ParameterFactor =
        std::make_shared<scope::ParameterFactor>(0, ShapeSqrtCov, Params);

    //------------------------------------------------------------------------------
    // Get KeyPoint3D origin position
    //------------------------------------------------------------------------------
    scope::getKeyPointOrigin3D(
        RootPose, JDirs, J, KeyPointDirs, KeyPoints, Params,
        scope::KeyPointInfo::SMPL::KeyPointOriginIndex, origin);

    //------------------------------------------------------------------------------
    // Setup JointDepthCamera factors
    //------------------------------------------------------------------------------
    scope::createJointDepthCameraFactors(JointDepthCamInfo, KeyPoint3DGMSigma,
                                         KeyPoint3DGMEps, Measurements3D,
                                         origin, JointDepthCamFactors);

    //------------------------------------------------------------------------------
    // Setup VertexDepthCamera factors
    //------------------------------------------------------------------------------
    scope::createVertexDepthCameraFactors(
        VertexDepthCamInfo, 0, JDirs, J, KeyPointDirs, KeyPoints,
        KeyPoint3DGMSigma, KeyPoint3DGMEps, Measurements3D, origin,
        VertexDepthCamFactors);

    // Reset optimizer
    smplify.reset();

    // Add factors
    for (const auto &factor : JointPinholeCamFactors) {
      smplify.addJointPinholeCameraFactor(factor);
    }

    for (const auto &factor : VertexPinholeCamFactors) {
      smplify.addVertexPinholeCameraFactor(factor);
    }

    for (const auto &factor : UnitPOFFactors) {
      smplify.addPOFFactor(factor);
    }

    for (const auto &factor : ScaledPOFFactors) {
      smplify.addPOFFactor(factor);
    }

    for (const auto &factor : RelPOFFactors) {
      smplify.addPOFFactor(factor);
    }

    for (const auto &factor : JointDepthCamFactors) {
      smplify.addDepthCameraFactor(factor);
    }

    for (const auto &factor : VertexDepthCamFactors) {
      smplify.addDepthCameraFactor(factor);
    }

    for (const auto &factor : JointFactors) {
      smplify.addJointFactor(factor);
    }

    for (const auto &factor : JointConstFactors) {
      smplify.addJointConstFactor(factor);
    }

    for (const auto &factor : EulerAngleConstFactors) {
      smplify.addJointConstFactor(factor);
    }

    for (int n = 0; n < NumJoints; n++) {
      smplify.addJointLimitFactor(JointLimitFactors[n]);
    }

    smplify.addParameterFactor(ParameterFactor);

    smplify.addPoseFactor(PoseFactor);

    // Update weights
    options.weights.tail<8>() *= 0.1;
    options.weights[scope::Optimizer::FactorIndex::UnitPOF] = 3000;
    options.weights[scope::Optimizer::FactorIndex::ScaledPOF] = 3000;
    options.weights[scope::Optimizer::FactorIndex::RelPOF] = 3000;
    options.weights[scope::Optimizer::FactorIndex::JointDepthCamera] = 24000;
    options.weights[scope::Optimizer::FactorIndex::VertexDepthCamera] = 24000;
    options.weights[scope::Optimizer::FactorIndex::JointLimit] = 400;
    options.weights[scope::Optimizer::FactorIndex::JointConst] = 1000;
    options.weights[scope::Optimizer::FactorIndex::Pose] = 200;
    options.weights[scope::Optimizer::FactorIndex::Joint] = 2500;
    options.weights[scope::Optimizer::FactorIndex::Parameter] = 400;
    smplify.updateFactorWeights(options.weights);

    // initialize the optimizer
    smplify.initialize(RootPose, Joints, {Params}, false);

    for (int iter = 0; iter < 100; iter++) {
      smplify.optimize();

      if (smplify.getResults().expected_rel_cost_reduction.back() < 1e-2 ||
          smplify.getResults().rel_cost_reduction.back() < 2e-3) {
        break;
      }
    }

    //---------------------------------------------------------------
    // Stage 4
    //---------------------------------------------------------------
    options.weights[scope::Optimizer::FactorIndex::UnitPOF] = 1500;
    options.weights[scope::Optimizer::FactorIndex::ScaledPOF] = 1500;
    options.weights[scope::Optimizer::FactorIndex::RelPOF] = 1500;
    options.weights[scope::Optimizer::FactorIndex::JointDepthCamera] = 24000;
    options.weights[scope::Optimizer::FactorIndex::VertexDepthCamera] = 24000;
    options.weights[scope::Optimizer::FactorIndex::JointLimit] = 150;
    options.weights[scope::Optimizer::FactorIndex::JointConst] = 1000;
    options.weights[scope::Optimizer::FactorIndex::Pose] = 100;
    options.weights[scope::Optimizer::FactorIndex::Joint] = 500;
    options.weights[scope::Optimizer::FactorIndex::Parameter] = 100;
    smplify.updateFactorWeights(options.weights);

    for (int iter = 0; iter < 100; iter++) {
      smplify.optimize();

      if (smplify.getResults().expected_rel_cost_reduction.back() < 1e-2 ||
          smplify.getResults().rel_cost_reduction.back() < 2e-3) {
        break;
      }
    }

    //---------------------------------------------------------------
    // Stage 5
    //---------------------------------------------------------------
    options.weights[scope::Optimizer::FactorIndex::UnitPOF] = 1500;
    options.weights[scope::Optimizer::FactorIndex::ScaledPOF] = 1500;
    options.weights[scope::Optimizer::FactorIndex::RelPOF] = 1500;
    options.weights[scope::Optimizer::FactorIndex::JointDepthCamera] = 24000;
    options.weights[scope::Optimizer::FactorIndex::VertexDepthCamera] = 24000;
    options.weights[scope::Optimizer::FactorIndex::JointLimit] = 50;
    options.weights[scope::Optimizer::FactorIndex::JointConst] = 1000;
    options.weights[scope::Optimizer::FactorIndex::Pose] = 100;
    options.weights[scope::Optimizer::FactorIndex::Joint] = 100;
    options.weights[scope::Optimizer::FactorIndex::Parameter] = 20;
    smplify.updateFactorWeights(options.weights);

    for (int iter = 0; iter < 100; iter++) {
      smplify.optimize();

      if (smplify.getResults().expected_rel_cost_reduction.back() < 1e-3 ||
          smplify.getResults().rel_cost_reduction.back() < 2e-4) {
        break;
      }
    }

    //---------------------------------------------------------------
    // Stage 6
    //---------------------------------------------------------------
    NumIters += smplify.getResults().fobjs.size() - 1;

    RootPose = smplify.getPoses()[0];
    Joints = smplify.getJoints();
    Params = smplify.getParameters()[0];

    //------------------------------------------------------------------------------
    // Setup pose factor
    //------------------------------------------------------------------------------
    PoseSqrtCov = scope::Matrix6::Identity();
    PoseSqrtCov.diagonal().head<3>().setConstant(2);
    PoseSqrtCov.diagonal().tail<3>().setConstant(2000);

    PoseFactor = std::make_shared<scope::PoseFactor>(0, PoseSqrtCov, RootPose);

    //------------------------------------------------------------------------------
    // Setup parameter factor
    //------------------------------------------------------------------------------
    ParameterFactor =
        std::make_shared<scope::ParameterFactor>(0, ShapeSqrtCov, Params);

    //------------------------------------------------------------------------------
    // Get KeyPoint3D origin position
    //------------------------------------------------------------------------------
    scope::getKeyPointOrigin3D(
        RootPose, JDirs, J, KeyPointDirs, KeyPoints, Params,
        scope::KeyPointInfo::SMPL::KeyPointOriginIndex, origin);

    //------------------------------------------------------------------------------
    // Setup JointDepthCamera factors
    //------------------------------------------------------------------------------
    scope::createJointDepthCameraFactors(JointDepthCamInfo, KeyPoint3DGMSigma,
                                         KeyPoint3DGMEps, Measurements3D,
                                         origin, JointDepthCamFactors);

    //------------------------------------------------------------------------------
    // Setup VertexDepthCamera factors
    //------------------------------------------------------------------------------
    scope::createVertexDepthCameraFactors(
        VertexDepthCamInfo, 0, JDirs, J, KeyPointDirs, KeyPoints,
        KeyPoint3DGMSigma, KeyPoint3DGMEps, Measurements3D, origin,
        VertexDepthCamFactors);

    // Reset optimizer
    smplify.reset();

    // Add factors
    for (const auto &factor : JointPinholeCamFactors) {
      smplify.addJointPinholeCameraFactor(factor);
    }

    for (const auto &factor : VertexPinholeCamFactors) {
      smplify.addVertexPinholeCameraFactor(factor);
    }

    for (const auto &factor : UnitPOFFactors) {
      smplify.addPOFFactor(factor);
    }

    for (const auto &factor : ScaledPOFFactors) {
      smplify.addPOFFactor(factor);
    }

    for (const auto &factor : RelPOFFactors) {
      smplify.addPOFFactor(factor);
    }

    for (const auto &factor : JointDepthCamFactors) {
      smplify.addDepthCameraFactor(factor);
    }

    for (const auto &factor : VertexDepthCamFactors) {
      smplify.addDepthCameraFactor(factor);
    }

    for (const auto &factor : JointFactors) {
      smplify.addJointFactor(factor);
    }

    for (const auto &factor : JointConstFactors) {
      smplify.addJointConstFactor(factor);
    }

    for (const auto &factor : EulerAngleConstFactors) {
      smplify.addJointConstFactor(factor);
    }

    for (int n = 0; n < NumJoints; n++) {
      smplify.addJointLimitFactor(JointLimitFactors[n]);
    }

    smplify.addParameterFactor(ParameterFactor);

    smplify.addPoseFactor(PoseFactor);

    // Update weights
    options.weights[scope::Optimizer::FactorIndex::UnitPOF] = 2500;
    options.weights[scope::Optimizer::FactorIndex::ScaledPOF] = 2500;
    options.weights[scope::Optimizer::FactorIndex::RelPOF] = 2500;
    options.weights[scope::Optimizer::FactorIndex::JointDepthCamera] = 24000;
    options.weights[scope::Optimizer::FactorIndex::VertexDepthCamera] = 24000;
    options.weights[scope::Optimizer::FactorIndex::JointLimit] = 50;
    options.weights[scope::Optimizer::FactorIndex::JointConst] = 250;
    options.weights[scope::Optimizer::FactorIndex::Pose] = 100;
    options.weights[scope::Optimizer::FactorIndex::Joint] = 25;
    options.weights[scope::Optimizer::FactorIndex::Parameter] = 100;
    smplify.updateFactorWeights(options.weights);

    // initialize the optimizer
    smplify.initialize(RootPose, Joints, {Params}, false);

    for (int iter = 0; iter < 100; iter++) {
      smplify.optimize();

      if (smplify.getResults().expected_rel_cost_reduction.back() < 1e-4 ||
          smplify.getResults().rel_cost_reduction.back() < 2e-5) {
        break;
      }
    }

    NumIters += smplify.getResults().fobjs.size() - 1;

    auto OptTime = scope::Stopwatch::tock(OptStart);
    auto Fobj = smplify.getFobj();
    auto TotalTime = OptTime + SetupTime + InitTime;

    std::cout << "======================================================="
              << std::endl;
    std::cout << seq + "_" + ss.str() << std::endl;
    std::cout << "-------------------------------------------------------"
              << std::endl;
    std::cout << "objective value: " << Fobj << std::endl;
    std::cout << "init time: " << InitTime << std::endl;
    std::cout << "setup time: " << SetupTime << std::endl;
    std::cout << "optimization time: " << OptTime << std::endl;
    std::cout << "total time: " << TotalTime << std::endl;
    std::cout << "iterations: " << NumIters << std::endl;

    //---------------------------------------------------------------
    // Results
    //---------------------------------------------------------------
    const auto &Poses = smplify.getPoses();
    Params = smplify.getParameters()[0];
    RootPose = smplify.getPoses()[0];
    Joints = smplify.getJoints();

    // Process root pose and joints
    Eigen::Matrix3Xd Rots;
    Rots.setZero(3, 52);

    auto RootRot = Rots.col(0);
    scope::math::SO3::log(RootPose.R, RootRot);

    for (int i = 0; i < SMPL::NumJoints; i++) {
      auto rot = Rots.col(i + 1);
      scope::math::SO3::log(Joints[i], rot);
    }

    Eigen::Vector3d J0 = J.head<3>();
    J0.noalias() += JDirs.topRows<3>() * Params;

    // get 2D keypoints
    scope::Matrix2X KeyPoints2D =
        scope::Matrix2X::Zero(2, Measurements2D.cols());

    for (const auto &info : JointPinholeCamInfo) {
      const auto &m = info[0];
      const auto &i = info[1];

      KeyPoints2D.col(m) = Poses[i].t.head<2>() / Poses[i].t[2];
    }

    for (const auto &info : VertexPinholeCamInfo) {
      scope::VertexPinholeCameraFactor::Evaluation eval;

      const auto &m = info[0];
      const auto &i = info[1];
      const auto &v = info[2];

      const auto &measurement = Measurements2D.col(m);

      scope::Matrix3X KeyPointDir =
          KeyPointDirs.middleRows<3>(3 * v) - JDirs.middleRows<3>(3 * i);
      scope::Vector3 KeyPoint =
          KeyPoints.segment<3>(3 * v) - J.segment<3>(3 * i);

      std::shared_ptr<scope::VertexPinholeCameraFactor> factor =
          std::make_shared<scope::VertexPinholeCameraFactor>(
              i, 0, KeyPointDir, KeyPoint, 1, 0.5, measurement.head<2>(),
              measurement[2]);

      factor->evaluate(Poses, {}, Joints, {Params}, eval);

      KeyPoints2D.col(m) = eval.point2D;
    }

    // get 3D keypoints
    scope::Matrix3X KeyPoints3D =
        scope::Matrix3X::Zero(3, Measurements3D.cols());

    for (const auto &info : JointDepthCamInfo) {
      const auto &m = std::get<0>(info);
      const auto &i = std::get<1>(info);

      KeyPoints3D.col(m) = Poses[i].t;
    }

    for (const auto &info :
         scope::KeyPointInfo::SMPL::VertexDepthCameraFactorInfo) {
      scope::VertexDepthCameraFactor::Evaluation eval;

      const auto &m = std::get<0>(info);
      const auto &i = std::get<1>(info);
      const auto &v = std::get<2>(info);
      const auto &confidence = std::get<3>(info);

      const scope::Vector3 measurement = Measurements3D.col(m) + Poses[0].t;

      scope::Matrix3X KeyPointDir =
          KeyPointDirs.middleRows<3>(3 * v) - JDirs.middleRows<3>(3 * i);
      scope::Vector3 KeyPoint =
          KeyPoints.segment<3>(3 * v) - J.segment<3>(3 * i);

      std::shared_ptr<scope::VertexDepthCameraFactor> factor =
          std::make_shared<scope::VertexDepthCameraFactor>(
              i, 0, KeyPointDir, KeyPoint, 1, 0.5, measurement, confidence);

      factor->evaluate(Poses, {}, Joints, {Params}, eval);

      KeyPoints3D.col(m) = eval.point3D;
    }

    // save results
    std::string ResultFile = ResultDir + ss.str() + ".npz";

    cnpy::npz_save(ResultFile, "betas", Params.data(),
                   {(size_t)Params.size() - 1}, "w");
    cnpy::npz_save(ResultFile, "poses", Rots.data(), {(size_t)(3 * 52)}, "a");
    cnpy::npz_save(ResultFile, "trans", RootPose.t.data(), {(size_t)(3)}, "a");
    cnpy::npz_save(ResultFile, "measurements", Measurements2D.data(),
                   {size_t(Measurements2D.cols()), size_t(3)}, "a");
    cnpy::npz_save(ResultFile, "camera", CamParams.data(), {(size_t)(4)}, "a");
    cnpy::npz_save(ResultFile, "J0", J0.data(), {(size_t)(3)}, "a");
    cnpy::npz_save(ResultFile, "kpts2D", KeyPoints2D.data(),
                   {size_t(KeyPoints2D.cols()), 2}, "a");
    cnpy::npz_save(ResultFile, "kpts3D", KeyPoints3D.data(),
                   {size_t(KeyPoints3D.cols()), 3}, "a");
    cnpy::npz_save(ResultFile, "opt_time", &OptTime, {1}, "a");
    cnpy::npz_save(ResultFile, "setup_time", &SetupTime, {1}, "a");
    cnpy::npz_save(ResultFile, "init_time", &InitTime, {1}, "a");
    cnpy::npz_save(ResultFile, "total_time", &TotalTime, {1}, "a");
    cnpy::npz_save(ResultFile, "objective", &Fobj, {1}, "a");
    cnpy::npz_save(ResultFile, "iterations", &NumIters, {1}, "a");
  }

  return 0;
}
