#include <glog/logging.h>

#include <memory>

#include <scope/optimizer/Optimizer.h>

namespace scope {
namespace Optimizer {
template <int K, int P, int N, bool CamOpt>
Optimizer<K, P, N, CamOpt>::Optimizer(
    const std::shared_ptr<const Model<K, P, N>> &model, const Options &options)
    : mModel(model),
      mOptions(options),
      mStatus(Status::Uninitialized),
      mRawB(6, K * (P + 3)),
      mB(mRawB.data()),
      mRawM(DSize, NumPoses * DSize),
      mM(mRawM.data()),
      mRawPoseGN(6, NumPoses),
      mPoseGN(mRawPoseGN.data()),
      mRawJointGN(3, NumJoints),
      mJointGN(mRawJointGN.data()),
      mRawParamGN(DParamSize),
      mParamGN(mRawParamGN.data()),
      mRawhGN(6 + DParamSize),
      mhGN(mRawhGN.data()),
      mRawH(DSize, NumJoints * DSize),
      mH(mRawH.data()),
      mRawHxB(DSize, NumJoints * (P + 3)),
      mHxB(mRawHxB.data()),
      mRawKuxp(3 * NumJoints, 6 + DParamSize),
      mKuxp(mRawKuxp.data()),
      mRawH0(6 + DParamSize, 6 + DParamSize),
      mH0(mRawH0.data()),
      mRawku(3 * K),
      mku(mRawku.data()),
      mRawh0(6 + DParamSize),
      mh0(mRawh0.data()),
      mRawRelJointLocations{VectorX(3 * K), VectorX(3 * K)},
      mRelJointLocations{
          Eigen::Map<Vector<3 * K>>(mRawRelJointLocations[0].data()),
          Eigen::Map<Vector<3 * K>>(mRawRelJointLocations[1].data())},
      mRawLambda(6 + DParamSize + 3 * NumJoints),
      mLambda(mRawLambda.data()),
      mRawSquaredError(6 + DParamSize + 3 * NumJoints),
      mSquaredError(mRawSquaredError.data()) {}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::initialize(bool refine) const {
  // TODO: initialize mvShapes

  mvPoses[0][0] = Pose::Identity();
  mvPoses[0][0].R(1, 1) = -1;
  mvPoses[0][0].R(2, 2) = -1;
  mvPoses[0][0].t[2] = 3;  // an guess for the depth
  mvJoints[0].assign(NumJoints, scope::Matrix3::Identity());

  for (int i = 0; i < NumParams; i++) {
    mvParams[0][i].setZero(ParamSizes[i]);
  }

  return initialize(mvPoses[0][0], mvJoints[0], mvParams[0], refine);
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::initialize(const Pose &T0,
                                           const AlignedVector<Matrix3> &joints,
                                           const AlignedVector<VectorX> &params,
                                           bool refine) const {
  assert(joints.size() == NumJoints);

  if (joints.size() != NumJoints) {
    LOG(ERROR) << "There should be " << NumJoints << " joint states."
               << std::endl;

    exit(-1);
  }

  assert(params.size() == NumParams);

  if (params.size() != NumParams) {
    LOG(ERROR) << "There should be " << NumParams << " parameters."
               << std::endl;

    exit(-1);
  }

  for (int i = 0; i < NumParams; i++) {
    assert(params[i].size() == ParamSizes[i]);

    if (params[i].size() != ParamSizes[i]) {
      LOG(ERROR) << "The size of params[" << i << "] should be "
                 << ParamSizes[i] << "." << std::endl;

      exit(-1);
    }
  }

  mvPoses[0][0] = T0;
  mvJoints[0] = joints;
  mvParams[0] = params;

  if (refine) {
    if (evaluateInitialFactors(0)) {
      mStatus = Status::Failed;

      exit(-1);
    }

    mDLambda = 1.0;
    mStepsize = 1.0;

    for (int i = 0; i < 20; i++) {
      initialOptimize();

      if (-mE / mfobj[0] < 1e-4) {
        break;
      }
    }
  }

  const auto &model = mModel;
  model->FK(mvPoses[0], mRelJointLocations[0], mvPoses[0][0], mvJoints[0],
            mvParams[0][0]);

  if (evaluateFactors(0)) {
    mStatus = Status::Failed;

    exit(-1);
  }

  updateGaussNewtonInfo();

  mResults = Results();
  mResults.fobjs.push_back(mfobj[0]);

  mDLambda = 1.5;
  mStepsize = 1.0;

  mStatus = Status::Initialized;

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::optimize() const {
  if (mStatus == Status::Uninitialized) {
    LOG(WARNING) << "The optimization must be initialized." << std::endl;

    exit(-1);
  }

  if (mStatus == Status::Failed) {
    LOG(ERROR) << "The optimization failed." << std::endl;

    exit(-1);
  }

  // mModel->DFK(mvB, mvPoses[0]);

  // for (int pose = 0; pose < NumPoses; pose++) {
  // linearizeFactors(pose);
  //}

  if (linearize()) {
    exit(-1);
  }

  solveGaussNewton();

  updateCostReduction();

  const auto &options = mOptions;

  Scalar stepsize;
  auto &results = mResults;

  const auto &fobj = mfobj;

  mStatus = Status::Aborted;

  Scalar expected_rel_cost_reduction = -mE / fobj[0];
  Scalar rel_cost_reduction = 0;

  if (options.method == Method::LM) {
    stepsize = 1.0;

    update(stepsize);

    if (evaluateFactors(1)) {
      exit(-1);
    }

    if (fobj[1] < fobj[0]) {
      rel_cost_reduction = (fobj[0] - fobj[1]) / fobj[0];

      Scalar rho = (fobj[1] - fobj[0]) / mE;

      if (rho > 0.70) {
        mDLambda = std::max(mDLambda * mOptions.lm_lambda_decrease, Scalar(1));
      } else if (rho < 0.25) {
        mDLambda = std::min(mDLambda * mOptions.lm_lambda_increase,
                            mOptions.lm_lambda_max);
      }

      accept();

      mStepsize = std::max(0.2, mStepsize);

      mStatus = Status::Accepted;
    } else {
      stepsize = std::min(mStepsize, mOptions.stepsize_decrease_ratio);

      for (int k = 1; k < mOptions.max_inner_iters; k++) {
        update(stepsize);

        if (evaluateFactors(1)) {
          exit(-1);
        }

        if (fobj[1] < fobj[0]) {
          rel_cost_reduction = (fobj[0] - fobj[1]) / fobj[0];

          accept();
          mStatus = Status::Accepted;

          mStepsize = std::min((Scalar)1.0,
                               stepsize * mOptions.stepsize_increase_ratio);

          break;
        }

        stepsize *= mOptions.stepsize_decrease_ratio;
      }

      mDLambda = std::min(mDLambda * mOptions.lm_lambda_increase,
                          mOptions.lm_lambda_max);
    }
  } else {
    stepsize = mStepsize;

    for (int k = 0; k < mOptions.max_inner_iters; k++) {
      update(stepsize);

      if (evaluateFactors(1)) {
        exit(-1);
      }

      if (fobj[1] < fobj[0]) {
        rel_cost_reduction = (fobj[0] - fobj[1]) / fobj[0];

        accept();
        mStatus = Status::Accepted;

        mStepsize =
            std::min((Scalar)1.0, stepsize * mOptions.stepsize_increase_ratio);

        break;
      }

      stepsize *= mOptions.stepsize_decrease_ratio;
    }
  }

  results.fobjs.push_back(fobj[0]);
  results.expected_rel_cost_reduction.push_back(expected_rel_cost_reduction);
  results.rel_cost_reduction.push_back(rel_cost_reduction);
  results.stepsizes.push_back(stepsize);

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::evaluate() const {
  if (mStatus == Status::Uninitialized) {
    LOG(WARNING) << "The optimizer is uninitialized." << std::endl;

    return -1;
  }

  const auto &model = mModel;
  model->FK(mvPoses[0], mRelJointLocations[0], mvPoses[0][0], mvJoints[0],
            mvParams[0][0]);

  if (evaluateFactors(0)) {
    mStatus = Status::Failed;

    exit(-1);
  }

  updateGaussNewtonInfo();

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateFactorWeights(
    const Eigen::Matrix<Scalar, NumFactors, 1> &weights) {
  mOptions.weights = weights;

  mfobj[0] = 0.5 * mOptions.weights.dot(mvFobj[0]);

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::reset() {
  mnvFactors.clear();
  mnFactors.clear();

  mvFactors.clear();
  mFactors.clear();

  mnvFactorEvals[0].clear();
  mvFactorEvals[0].clear();

  mvFactorEvals[0].clear();
  mFactorEvals[0].clear();

  mnvFactorEvals[1].clear();
  mvFactorEvals[1].clear();

  mvFactorEvals[1].clear();
  mFactorEvals[1].clear();

  mnvFactorLins.clear();
  mvFactorLins.clear();

  mvFactorLins.clear();
  mFactorLins.clear();

  setupFactors();

  mvFobj[0].setZero();
  mvFobj[1].setZero();

  mfobj[0] = 0;
  mfobj[1] = 0;

  mStatus = Status::Uninitialized;

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::addJointPinholeCameraFactor(
    std::shared_ptr<JointPinholeCameraFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  const auto &pose = factor->getPoses()[0];

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  int index;

  const auto &params = factor->getParams();

  if (params.size() == 0) {
    index = FactorIndex::JointPinholeCamera;
  } else {
    assert(params[0] == CamParamIndex);

    if (params[0] != CamParamIndex) {
      LOG(ERROR) << "The camera parameter must be valid." << std::endl;

      exit(-1);
    }

    index = FactorIndex::FullJointPinholeCamera;
  }

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  mnvFactors[pose][index].push_back(factor);
  mnvFactorEvals[0][pose][index].push_back(evals[0]);
  mnvFactorEvals[1][pose][index].push_back(evals[1]);
  mnvFactorLins[pose][index].push_back(lin);

  mnFactors[pose].push_back(factor);
  mnFactorEvals[0][pose].push_back(evals[0]);
  mnFactorEvals[1][pose].push_back(evals[1]);
  mnFactorLins[pose].push_back(lin);

  return factor->getID();
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::addVertexPinholeCameraFactor(
    std::shared_ptr<VertexPinholeCameraFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  const auto &pose = factor->getPoses()[0];

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &params = factor->getParams();

  assert(params[0] == VertexParamIndex);

  if (params[0] != VertexParamIndex) {
    LOG(ERROR) << "The vertex parameter must be valid." << std::endl;

    exit(-1);
  }

  int index;

  std::shared_ptr<VertexPinholeCameraFactor> newFactor;

  if (params.size() == 1) {
    index = FactorIndex::VertexPinholeCamera;
  } else {
    assert(params[1] == CamParamIndex);

    if (params[1] != CamParamIndex) {
      LOG(ERROR) << "The camera parameter must be valid." << std::endl;

      exit(-1);
    }

    index = FactorIndex::FullVertexPinholeCamera;
  }

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  mnvFactors[pose][index].push_back(factor);
  mnvFactorEvals[0][pose][index].push_back(evals[0]);
  mnvFactorEvals[1][pose][index].push_back(evals[1]);
  mnvFactorLins[pose][index].push_back(lin);

  mnFactors[pose].push_back(factor);
  mnFactorEvals[0][pose].push_back(evals[0]);
  mnFactorEvals[1][pose].push_back(evals[1]);
  mnFactorLins[pose].push_back(lin);

  return factor->getID();
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::addPOFFactor(
    std::shared_ptr<POFFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  const auto &pose = factor->getPoses()[0];

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  int index;

  const auto &params = factor->getParams();
  const auto &joints = factor->getJoints();

  std::shared_ptr<POFFactor> newFactor;

  if (params.size() == 0) {
    index = FactorIndex::UnitPOF;
  } else if (joints.size() == 0) {
    assert(params[0] == VertexParamIndex);

    if (params[0] != VertexParamIndex) {
      LOG(ERROR) << "The vertex parameter must be valid." << std::endl;

      exit(-1);
    }

    index = FactorIndex::ScaledPOF;
  } else {
    assert(params[0] == VertexParamIndex);

    if (params[0] != VertexParamIndex) {
      LOG(ERROR) << "The vertex parameter must be valid." << std::endl;

      exit(-1);
    }

    const auto &joint = factor->getJoints()[0];

    assert(joint == pose - 1 && joint >= 0);

    if (joint < 0 || joint != pose - 1) {
      LOG(ERROR) << "The joint must be valid." << std::endl;

      exit(-1);
    }

    index = FactorIndex::RelPOF;
  }

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  mnvFactors[pose][index].push_back(factor);
  mnvFactorEvals[0][pose][index].push_back(evals[0]);
  mnvFactorEvals[1][pose][index].push_back(evals[1]);
  mnvFactorLins[pose][index].push_back(lin);

  mnFactors[pose].push_back(factor);
  mnFactorEvals[0][pose].push_back(evals[0]);
  mnFactorEvals[1][pose].push_back(evals[1]);
  mnFactorLins[pose].push_back(lin);

  return factor->getID();
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::addDepthCameraFactor(
    std::shared_ptr<DepthCameraFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  const auto &pose = factor->getPoses()[0];

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  int index;

  const auto &params = factor->getParams();

  std::shared_ptr<DepthCameraFactor> newFactor;

  if (params.size() == 0) {
    index = FactorIndex::JointDepthCamera;
  } else {
    assert(params[0] == VertexParamIndex);

    if (params[0] != VertexParamIndex) {
      LOG(ERROR) << "The vertex parameter must be valid." << std::endl;

      exit(-1);
    }

    index = FactorIndex::VertexDepthCamera;
  }

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  mnvFactors[pose][index].push_back(factor);
  mnvFactorEvals[0][pose][index].push_back(evals[0]);
  mnvFactorEvals[1][pose][index].push_back(evals[1]);
  mnvFactorLins[pose][index].push_back(lin);

  mnFactors[pose].push_back(factor);
  mnFactorEvals[0][pose].push_back(evals[0]);
  mnFactorEvals[1][pose].push_back(evals[1]);
  mnFactorLins[pose].push_back(lin);

  return factor->getID();
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::addJointLimitFactor(
    std::shared_ptr<JointLimitFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  const auto &joint = factor->getJoints()[0];

  assert(joint >= 0 && joint < NumJoints);

  if (joint < 0 || joint >= NumJoints) {
    LOG(ERROR) << "The joint must be valid." << std::endl;

    exit(-1);
  }

  const int pose = joint + 1;
  const int index = FactorIndex::JointLimit;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  mnvFactors[pose][index].push_back(factor);
  mnvFactorEvals[0][pose][index].push_back(evals[0]);
  mnvFactorEvals[1][pose][index].push_back(evals[1]);
  mnvFactorLins[pose][index].push_back(lin);

  mnFactors[pose].push_back(factor);
  mnFactorEvals[0][pose].push_back(evals[0]);
  mnFactorEvals[1][pose].push_back(evals[1]);
  mnFactorLins[pose].push_back(lin);

  return factor->getID();
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::addPoseConstFactor(
    std::shared_ptr<PoseConstFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  const auto &pose = factor->getPoses()[0];

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const int index = FactorIndex::PoseConst;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  mnvFactors[pose][index].push_back(factor);
  mnvFactorEvals[0][pose][index].push_back(evals[0]);
  mnvFactorEvals[1][pose][index].push_back(evals[1]);
  mnvFactorLins[pose][index].push_back(lin);

  mnFactors[pose].push_back(factor);
  mnFactorEvals[0][pose].push_back(evals[0]);
  mnFactorEvals[1][pose].push_back(evals[1]);
  mnFactorLins[pose].push_back(lin);

  return factor->getID();
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::addJointConstFactor(
    std::shared_ptr<JointConstFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  const auto &joint = factor->getJoints()[0];

  assert(joint >= 0 && joint < NumJoints);

  if (joint < 0 || joint >= NumJoints) {
    LOG(ERROR) << "The joint must be valid." << std::endl;

    exit(-1);
  }

  const int pose = joint + 1;
  const int index = FactorIndex::JointConst;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  mnvFactors[pose][index].push_back(factor);
  mnvFactorEvals[0][pose][index].push_back(evals[0]);
  mnvFactorEvals[1][pose][index].push_back(evals[1]);
  mnvFactorLins[pose][index].push_back(lin);

  mnFactors[pose].push_back(factor);
  mnFactorEvals[0][pose].push_back(evals[0]);
  mnFactorEvals[1][pose].push_back(evals[1]);
  mnFactorLins[pose].push_back(lin);

  return factor->getID();
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::addPoseFactor(
    std::shared_ptr<PoseFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  const auto &pose = factor->getPoses()[0];

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const int index = FactorIndex::Pose;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  mnvFactors[pose][index].push_back(factor);
  mnvFactorEvals[0][pose][index].push_back(evals[0]);
  mnvFactorEvals[1][pose][index].push_back(evals[1]);
  mnvFactorLins[pose][index].push_back(lin);

  mnFactors[pose].push_back(factor);
  mnFactorEvals[0][pose].push_back(evals[0]);
  mnFactorEvals[1][pose].push_back(evals[1]);
  mnFactorLins[pose].push_back(lin);

  return factor->getID();
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::addJointFactor(
    std::shared_ptr<JointFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  const auto &joint = factor->getJoints()[0];

  assert(joint >= 0 && joint < NumJoints);

  if (joint < 0 || joint >= NumJoints) {
    LOG(ERROR) << "The joint must be valid." << std::endl;

    exit(-1);
  }

  const int pose = joint + 1;
  const int index = FactorIndex::Joint;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  mnvFactors[pose][index].push_back(factor);
  mnvFactorEvals[0][pose][index].push_back(evals[0]);
  mnvFactorEvals[1][pose][index].push_back(evals[1]);
  mnvFactorLins[pose][index].push_back(lin);

  mnFactors[pose].push_back(factor);
  mnFactorEvals[0][pose].push_back(evals[0]);
  mnFactorEvals[1][pose].push_back(evals[1]);
  mnFactorLins[pose].push_back(lin);

  return factor->getID();
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::addParameterFactor(
    std::shared_ptr<ParameterFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  const auto &param = factor->getParams()[0];

  assert(param >= 0 && param < NumParams);

  if (param < 0 || param >= NumParams) {
    LOG(ERROR) << "The parameter must be valid." << std::endl;

    exit(-1);
  }

  const int pose = 0;
  const int index = FactorIndex::Parameter;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  mnvFactors[pose][index].push_back(factor);
  mnvFactorEvals[0][pose][index].push_back(evals[0]);
  mnvFactorEvals[1][pose][index].push_back(evals[1]);
  mnvFactorLins[pose][index].push_back(lin);

  mnFactors[pose].push_back(factor);
  mnFactorEvals[0][pose].push_back(evals[0]);
  mnFactorEvals[1][pose].push_back(evals[1]);
  mnFactorLins[pose].push_back(lin);

  return factor->getID();
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::getNumPoses() const {
  return NumPoses;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::getNumShapes() const {
  return NumShapes;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::getNumJoints() const {
  return NumJoints;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::getNumParameters() const {
  return NumParams;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::setupModelInfo() {
  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::setupFactors() {
  mnvFactors.resize(
      NumPoses,
      std::vector<std::vector<std::shared_ptr<const Factor>>>(NumFactors));
  mnFactors.resize(NumPoses);
  mvFactors.resize(NumFactors);

  mnvFactorEvals[0].resize(
      NumPoses,
      std::vector<std::vector<std::shared_ptr<const Factor::Evaluation>>>(
          NumFactors));
  mnFactorEvals[0].resize(NumPoses);
  mvFactorEvals[0].resize(NumFactors);

  mnvFactorEvals[1].resize(
      NumPoses,
      std::vector<std::vector<std::shared_ptr<const Factor::Evaluation>>>(
          NumFactors));
  mnFactorEvals[1].resize(NumPoses);
  mvFactorEvals[1].resize(NumFactors);

  mnvFactorLins.resize(
      NumPoses,
      std::vector<std::vector<std::shared_ptr<const Factor::Linearization>>>(
          NumFactors));
  mnFactorLins.resize(NumPoses);
  mvFactorLins.resize(NumFactors);

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::getFaceParameterIndex() const {
  exit(-1);
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::getVertexParameterIndex() const {
  return VertexParamIndex;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::getCameraParameterIndex() const {
  return CamParamIndex;
}

template <int K, int P, int N, bool CamOpt>
const Results &Optimizer<K, P, N, CamOpt>::getResults() const {
  return mResults;
}

template <int K, int P, int N, bool CamOpt>
Scalar Optimizer<K, P, N, CamOpt>::getFobj() const {
  return mfobj[0];
}

template <int K, int P, int N, bool CamOpt>
AlignedVector<Pose> const &Optimizer<K, P, N, CamOpt>::getPoses() const {
  return mvPoses[0];
}

template <int K, int P, int N, bool CamOpt>
AlignedVector<VectorX> const &Optimizer<K, P, N, CamOpt>::getShapes() const {
  return mvShapes[0];
}

template <int K, int P, int N, bool CamOpt>
AlignedVector<Matrix3> const &Optimizer<K, P, N, CamOpt>::getJoints() const {
  return mvJoints[0];
}

template <int K, int P, int N, bool CamOpt>
AlignedVector<VectorX> const &Optimizer<K, P, N, CamOpt>::getParameters()
    const {
  return mvParams[0];
}

template <int K, int P, int N, bool CamOpt>
std::vector<std::vector<std::shared_ptr<const Factor::Evaluation>>> const &
Optimizer<K, P, N, CamOpt>::getEvaluations() const {
  return mvFactorEvals[0];
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::setupOptimization() {
  mvPoses[0].resize(NumPoses);
  mvPoses[1].resize(NumPoses);

  mvShapes[0].resize(NumShapes);
  mvShapes[1].resize(NumShapes);

  mvJoints[0].resize(NumJoints);
  mvJoints[1].resize(NumJoints);

  mvParams[0].resize(NumParams);
  mvParams[1].resize(NumParams);

  mvJointChange.resize(NumJoints);
  mvDJointChange.resize(NumJoints);

  mvmx.resize(NumPoses);
  mvmp.resize(NumPoses);
  mvmu.resize(NumPoses);

  mvPoseDG.resize(NumPoses);
  mvJointDG.resize(NumJoints);
  mvParamDG.resize(NumJoints);

  mPoseGN.resize(6, NumPoses);
  mJointGN.resize(3, NumJoints);

  mvPoseGN.reserve(NumPoses);
  mvJointGN.reserve(NumJoints);

  for (int i = 0; i < NumPoses; i++) {
    mvPoseGN.push_back(Eigen::Map<Vector6>(mPoseGN.data() + 6 * i));
  }

  for (int ii = 0; ii < NumJoints; ii++) {
    mvJointGN.push_back(Eigen::Map<Vector3>(mJointGN.data() + 3 * ii));
  }

  mvHuuInv.resize(NumJoints);

  mNumCollisions = 0;

  if (mOptions.check_collisions) {
    mCollisionParamSize = MaxNumCollisions;
  } else {
    mCollisionParamSize = 0;
  }

  mDCollisionParamOffset = 0;

  mH.setZero();
  mM.setZero();
  mHxB.setZero();
  mKuxp.setZero();
  mB.setZero();
  mku.setZero();

  int InnerSize[4] = {MaxDSize * MaxDSize, MaxDSize * (3 + VertexParamSize),
                      3 * DJointOffset, 6 * (3 + VertexParamSize)};

  mvH.reserve(NumJoints);
  mvM.reserve(NumPoses);
  mvHxB.reserve(NumJoints);
  mvKuxp.reserve(NumJoints);
  mvB.reserve(NumJoints);
  mvBp.reserve(NumJoints);
  mvBu.reserve(NumJoints);
  mvku.reserve(NumJoints);

  for (int ii = 0; ii < NumJoints; ii++) {
    mvH.push_back(
        Eigen::Map<Matrix<DSize, DSize>>(mH.data() + ii * InnerSize[0]));

    mvM.push_back(
        Eigen::Map<Matrix<DSize, DSize>>(mM.data() + ii * InnerSize[0]));

    mvHxB.push_back(
        Eigen::Map<Matrix<DSize, 3 + P>>(mHxB.data() + ii * InnerSize[1]));

    mvKuxp.push_back(Eigen::Map<RowMajorMatrix<3, 6 + ParamSize>>(
        mKuxp.data() + ii * InnerSize[2]));

    mvB.push_back(Eigen::Map<Matrix<6, 3 + P>>(mB.data() + ii * InnerSize[3]));

    mvBp.push_back(Eigen::Map<Matrix<3, P>, 0, Eigen::Stride<6, 1>>(
        mB.data() + 3 + ii * InnerSize[3]));

    mvBu.push_back(Eigen::Map<Matrix63>(mB.data() + 6 * P + ii * InnerSize[3]));

    mvku.push_back(Eigen::Map<Vector3>(mku.data() + ii * 3));
  }

  mvM.push_back(
      Eigen::Map<Matrix<DSize, DSize>>(mM.data() + NumJoints * InnerSize[0]));

  mvhx.resize(NumJoints);
  mvhp.resize(NumJoints);
  mvhu.resize(NumJoints);

  mvE.resize(NumJoints);

  mvDCostDx.resize(NumPoses);
  mvDCostDu.resize(NumPoses);
  mvDCostDp.resize(NumPoses);

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::init() const {
  scope::Matrix3 M = scope::Matrix3::Zero();
  scope::Vector3 m = scope::Vector3::Zero();
  scope::Matrix3 Minv;
  scope::Vector3 dt;

  if (linearize()) {
    exit(-1);
  }

  for (int i = 0; i < NumPoses; i++) {
    M += mvM[i].template block<3, 3>(3, 3);
    m += mvmx[i].template tail<3>();
  }

  M.diagonal().array() *= mDLambda;
  M.diagonal().array() += mOptions.delta;
  Minv.noalias() = M.inverse();

  dt.noalias() = -Minv * m;

  mE = 0.5 * dt.dot(m);
  M.diagonal().array() -= mOptions.delta;
  mE -= (1 - 1 / mDLambda) * M.diagonal().dot(dt.cwiseAbs2());

  mvPoses[1] = mvPoses[0];
  mvShapes[1] = mvShapes[0];
  mvJoints[1] = mvJoints[0];
  mvParams[1] = mvParams[0];

  Scalar stepsize = 1.0;

  const auto &fobj = mfobj;

  for (int iter = 0; iter < mOptions.max_inner_iters; iter++) {
    for (int i = 0; i < NumPoses; i++) {
      mvPoses[1][i].t = mvPoses[0][i].t + stepsize * dt;
    }

    if (evaluateFactors(1)) {
      exit(-1);
    }

    if (fobj[1] < fobj[0]) {
      Scalar rho = (fobj[1] - fobj[0]) / mE;

      if (rho > 0.75) {
        mDLambda = std::max(mDLambda * mOptions.lm_lambda_decrease, Scalar(1));
      } else if (rho < 0.25) {
        mDLambda = std::min(mDLambda * mOptions.lm_lambda_increase,
                            mOptions.lm_lambda_max);
      }

      accept();

      mStatus = Status::Accepted;

      return 0;
    }

    stepsize *= mOptions.stepsize_decrease_ratio;
  }

  mDLambda = std::max(mDLambda * mOptions.lm_lambda_decrease, Scalar(1));

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::initialOptimize() const {
  const auto &model = mModel;

  if (linearizeFactors(0)) {
    exit(-1);
  }

  mH0 = mvM[0]
            .template topLeftCorner<DPoseSize + DParamSize,
                                    DPoseSize + DParamSize>();
  mH0.template topRightCorner<DPoseSize, DParamSize>() =
      mH0.template bottomLeftCorner<DParamSize, DPoseSize>().transpose();

  if (mOptions.method == Method::LM) {
    mH0.diagonal().array() *= mDLambda;
  }

  mH0.diagonal().array() += mOptions.delta;

  mh0.template head<DPoseSize>() = mvmx[0];
  mh0.template tail<DParamSize>() = mvmp[0];

  mhGN.setZero();
  // mHchol0.compute(mRawH0);
  // mhGN = -mHchol0.solve(mh0);
  scope::Matrix6 Hinv = mH0.template topLeftCorner<6, 6>().inverse();
  mhGN.template head<6>().noalias() = -Hinv * mvmx[0];

  mvPoseGN[0] = mhGN.template head<DPoseSize>();
  mParamGN = mhGN.template tail<DParamSize>();

  mE = 0.5 * mh0.dot(mhGN);
  mH0.diagonal().array() -= mOptions.delta;
  mE -= (0.5 - 0.5 / mDLambda) * mH0.diagonal().dot(mhGN.cwiseAbs2());

  Scalar stepsize = 1.0;

  const auto &fobj = mfobj;

  for (int iter = 0; iter < mOptions.max_inner_iters; iter++) {
    mvParams[1] = mvParams[0];

    for (int i = 0; i < NumParams; i++) {
      mvParams[1][i] +=
          stepsize * mParamGN.segment(DParamOffsets[i], ParamSizes[i]);
    }

    mDRootPoseChange = mvPoseGN[0] * stepsize;
    Pose::exp(mDRootPoseChange, mRootPoseChange);
    mvPoses[1][0].R.noalias() = mRootPoseChange.R * mvPoses[0][0].R;
    mvPoses[1][0].t = mRootPoseChange.t;
    mvPoses[1][0].t.noalias() += mRootPoseChange.R * mvPoses[0][0].t;

    if (evaluateInitialFactors(1)) {
      exit(-1);
    }

    if (fobj[1] < fobj[0]) {
      Scalar rho = (fobj[1] - fobj[0]) / mE;

      if (rho > 0.75 && iter == 0) {
        mDLambda = std::max(mDLambda * mOptions.lm_lambda_decrease, Scalar(1));
      } else if (rho < 0.25 && iter > 0) {
        mDLambda = std::min(mDLambda * mOptions.lm_lambda_increase,
                            mOptions.lm_lambda_max);
      }

      mfobj[0] = mfobj[1];

      mvPoses[0][0] = mvPoses[1][0];
      mvParams[0].swap(mvParams[1]);

      mvFobj[0].swap(mvFobj[1]);

      mnvFactorEvals[0].swap(mnvFactorEvals[1]);
      mnFactorEvals[0].swap(mnFactorEvals[1]);

      mvFactorEvals[0].swap(mvFactorEvals[1]);
      mFactorEvals[0].swap(mFactorEvals[1]);

      mStatus = Status::Accepted;

      return 0;
    }

    stepsize *= mOptions.stepsize_decrease_ratio;
  }

  mDLambda = std::max(mDLambda * mOptions.lm_lambda_decrease, Scalar(1));

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::update(Scalar stepsize) const {
  assert(stepsize > 0);

  const auto &model = mModel;

  mvParams[1] = mvParams[0];

  for (int i = 0; i < NumParams; i++) {
    mvParams[1][i] +=
        stepsize * mParamGN.segment(DParamOffsets[i], ParamSizes[i]);
  }

  mDRootPoseChange = mvPoseGN[0] * stepsize;
  Pose::exp(mDRootPoseChange, mRootPoseChange);
  mvPoses[1][0].R.noalias() = mRootPoseChange.R * mvPoses[0][0].R;
  mvPoses[1][0].t = mRootPoseChange.t;
  mvPoses[1][0].t.noalias() += mRootPoseChange.R * mvPoses[0][0].t;

  for (int ii = 0; ii < NumJoints; ii++) {
    mvDJointChange[ii] = mvJointGN[ii] * stepsize;
    math::SO3::exp(mvDJointChange[ii], mvJointChange[ii]);
    mvJoints[1][ii].noalias() = mvJointChange[ii] * mvJoints[0][ii];
  }

  model->FK(mvPoses[1], mRelJointLocations[1], mvPoses[1][0], mvJoints[1],
            mvParams[1][0]);

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::evaluateFactors(int n) const {
  assert(n >= 0 && n <= 1);

  const auto &poses = mvPoses[n];
  const auto &shapes = mvShapes[n];
  const auto &joints = mvJoints[n];
  const auto &params = mvParams[n];

  auto &fobj = mfobj[n];
  auto &Fobj = mvFobj[n];
  auto &Evals = mFactorEvals[n];
  auto &FactorEvals = mvFactorEvals[n];

  Fobj.setZero();

  const auto &model = mModel;

#pragma omp parallel for
  for (int i = 0; i < mFactors.size(); i++) {
    const auto &factor = mFactors[i];
    auto &eval = *Evals[i];

    factor->evaluate(poses, shapes, joints, params, eval);
  }

  for (int k = 0; k < NumFactors; k++) {
    const auto &factors = mvFactors[k];
    const auto &evals = FactorEvals[k];

    for (int i = 0; i < factors.size(); i++) {
      const auto &factor = factors[i];

      auto &eval = *evals[i];

      assert(eval.status == Factor::Status::VALID);

      if (eval.status != Factor::Status::VALID) {
        LOG(ERROR) << "Failed to evaluate the factor." << std::endl;

        mStatus = Status::Failed;

        exit(-1);
      }

      Fobj[k] += eval.f;
    }
  }

  fobj = 0.5 * mOptions.weights.dot(Fobj);

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::evaluateInitialFactors(int n) const {
  assert(n >= 0 && n <= 1);

  const auto &poses = mvPoses[n];
  const auto &shapes = mvShapes[n];
  const auto &joints = mvJoints[n];
  const auto &params = mvParams[n];

  auto &fobj = mfobj[n];
  auto &Fobj = mvFobj[n];

  const auto &factors = mnFactors[0];

  auto &Evals = mnFactorEvals[n][0];
  auto &FactorEvals = mnvFactorEvals[n][0];

  Fobj.setZero();

  const auto &model = mModel;

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];
    auto &eval = *Evals[i];

    factor->evaluate(poses, shapes, joints, params, eval);
  }

  for (int k = 0; k < NumFactors; k++) {
    const auto &factors = mnvFactors[0][k];
    const auto &evals = FactorEvals[k];

    for (int i = 0; i < factors.size(); i++) {
      const auto &factor = factors[i];

      auto &eval = *evals[i];

      assert(eval.status == Factor::Status::VALID);

      if (eval.status != Factor::Status::VALID) {
        LOG(ERROR) << "Failed to evaluate the factor." << std::endl;

        mStatus = Status::Failed;

        exit(-1);
      }

      Fobj[k] += eval.f;
    }
  }

  fobj = 0.5 * mOptions.weights.dot(Fobj);

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateGaussNewtonInfo() const {
  if (!mOptions.check_collisions) {
    return 0;
  }

  int CollisionParamSize = mCollisionParamSize;

  mNumCollisions = 0;

  const auto &factors = mvFactors[FactorIndex::Collision];
  const auto &evals = mvFactorEvals[0][FactorIndex::Collision];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &eval = *evals[i];

    if (eval.f > 0) {
      mNumCollisions++;
    }
  }

  mCollisionParamSize = std::min(mNumCollisions, MaxNumCollisions);

  if (mCollisionParamSize == CollisionParamSize) {
    return 0;
  }

#if 0
  mDCollisionParamOffset = DParamOffset;

  mDCamParamOffset = mDCollisionParamOffset + mCollisionParamSize;
  mDVertexParamOffset = mDCamParamOffset + mCamParamSize;
  mDFaceParamOffset = mDVertexParamOffset + mVertexParamSize;

  mDParamSize = mParamSize + mCollisionParamSize;

  mDJointOffset = mDParamOffset + mDParamSize;

  mDSize = mDJointOffset + mDJointSize;

  for (int ii = 0; ii < NumJoints; ii++) {
    new (&mvH[ii]) Eigen::Map<scope::MatrixX>(mvH[ii].data(), mDSize, mDSize);
    new (&mvM[ii]) Eigen::Map<scope::MatrixX>(mvM[ii].data(), mDSize, mDSize);
    new (&mvHxB[ii])
        Eigen::Map<scope::MatrixX>(mvHxB[ii].data(), mDSize, mvH[ii].cols());
    new (&mvKuxp[ii]) Eigen::Map<scope::RowMajorMatrix3X>(mvKuxp[ii].data(), 3,
                                                          mDJointOffset);
  }

  new (&mvM[NumJoints])
      Eigen::Map<scope::MatrixX>(mvM[NumJoints].data(), mDSize, mDSize);
#endif

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::accept() const {
  mfobj[0] = mfobj[1];

  mvPoses[0].swap(mvPoses[1]);
  mvShapes[0].swap(mvShapes[1]);
  mvJoints[0].swap(mvJoints[1]);
  mvParams[0].swap(mvParams[1]);
  mRelJointLocations[0].swap(mRelJointLocations[1]);

  mvFobj[0].swap(mvFobj[1]);

  mnvFactorEvals[0].swap(mnvFactorEvals[1]);
  mnFactorEvals[0].swap(mnFactorEvals[1]);

  mvFactorEvals[0].swap(mvFactorEvals[1]);
  mFactorEvals[0].swap(mFactorEvals[1]);

  updateGaussNewtonInfo();

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::linearize() const {
  const auto &poses = mvPoses[0];
  const auto &shapes = mvShapes[0];
  const auto &joints = mvJoints[0];
  const auto &params = mvParams[0];

  // linearize forward kinematics
  const auto &model = mModel;
  model->DFK(mvB, poses);

  // linearize factors
  const auto &Evals = mFactorEvals[0];
  auto &Lins = mFactorLins;

#pragma omp parallel for
  for (int i = 0; i < mFactors.size(); i++) {
    const auto &factor = mFactors[i];
    const auto &eval = *Evals[i];
    auto &lin = *Lins[i];

    assert(eval.status == Factor::Status::VALID);

    factor->linearize(poses, shapes, joints, params, eval, lin);

    assert(lin.status == Factor::Status::VALID);
  }

  for (const auto &lin : Lins) {
    if (lin->status != Factor::Status::VALID) {
      LOG(ERROR) << "Failed to linearize the factor." << std::endl;

      mStatus = Status::Failed;

      exit(-1);
    }
  }

  mM.setZero();

  mvmx.assign(NumPoses, Vector6::Zero());
  mvmp.assign(NumPoses, Vector<DParamSize>::Zero());
  mvmu.assign(NumPoses, Vector3::Zero());

  updateJointPinholeCameraFactorGaussNewton();
  updateVertexPinholeCameraFactorGaussNewton();
  updateFullJointPinholeCameraFactorGaussNewton();
  updateFullVertexPinholeCameraFactorGaussNewton();
  updateUnitPOFFactorGaussNewton();
  updateScaledPOFFactorGaussNewton();
  updateRelPOFFactorGaussNewton();
  updateJointDepthCameraFactorGaussNewton();
  updateVertexDepthCameraFactorGaussNewton();
  updateJointLimitFactorGaussNewton();
  updateCollisionFactorGaussNewton();
  updatePoseConstFactorGaussNewton();
  updateJointConstFactorGaussNewton();
  updatePoseFactorGaussNewton();
  updateShapeFactorGaussNewton();
  updateJointFactorGaussNewton();
  updateParamFactorGaussNewton();

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::linearizeFactors(int pose) const {
  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &poses = mvPoses[0];
  const auto &shapes = mvShapes[0];
  const auto &joints = mvJoints[0];
  const auto &params = mvParams[0];

  const auto &factors = mnFactors[pose];
  const auto &evals = mnFactorEvals[0][pose];
  auto &lins = mnFactorLins[pose];

  // linearize factors
  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];
    const auto &eval = *evals[i];
    auto &lin = *lins[i];

    assert(eval.status == Factor::Status::VALID);

    factor->linearize(poses, shapes, joints, params, eval, lin);

    assert(lin.status == Factor::Status::VALID);
  }

  mvM[pose].setZero();

  mvmx[pose].setZero();
  mvmp[pose].setZero();
  mvmu[pose].setZero();

  updateJointPinholeCameraFactorGaussNewton(pose);
  updateVertexPinholeCameraFactorGaussNewton(pose);
  updateFullJointPinholeCameraFactorGaussNewton(pose);
  updateFullVertexPinholeCameraFactorGaussNewton(pose);
  updateUnitPOFFactorGaussNewton(pose);
  updateScaledPOFFactorGaussNewton(pose);
  updateRelPOFFactorGaussNewton(pose);
  updateJointDepthCameraFactorGaussNewton(pose);
  updateVertexDepthCameraFactorGaussNewton(pose);
  updateJointLimitFactorGaussNewton(pose);
  updateCollisionFactorGaussNewton(pose);
  updatePoseConstFactorGaussNewton(pose);
  updateJointConstFactorGaussNewton(pose);
  updatePoseFactorGaussNewton(pose);
  updateShapeFactorGaussNewton(pose);
  updateJointFactorGaussNewton(pose);
  updateParamFactorGaussNewton(pose);

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateJointPinholeCameraFactorGaussNewton()
    const {
  const auto &factors = mvFactors[FactorIndex::JointPinholeCamera];
  const auto &evals = mvFactorEvals[0][FactorIndex::JointPinholeCamera];
  const auto &lins = mvFactorLins[FactorIndex::JointPinholeCamera];

  const auto &weight = mOptions.weights[FactorIndex::JointPinholeCamera];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin = std::dynamic_pointer_cast<
        const JointPinholeCameraFactor::Linearization>(lins[i]);

    const auto &pose = factor->getPoses()[0];

    const auto &JPose =
        Eigen::Map<const Matrix<2, 6>>(lin->jacobians[0][0].data());

    const auto &gPose = lin->gPose;

    auto &M = mvM[pose];
    auto &mx = mvmx[pose];

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;
    mx += weight * gPose;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateVertexPinholeCameraFactorGaussNewton()
    const {
  const auto &options = mOptions;

  const auto &factors = mvFactors[FactorIndex::VertexPinholeCamera];
  const auto &evals = mvFactorEvals[0][FactorIndex::VertexPinholeCamera];
  const auto &lins = mvFactorLins[FactorIndex::VertexPinholeCamera];

  const auto &weight = options.weights[FactorIndex::VertexPinholeCamera];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin = std::dynamic_pointer_cast<
        const VertexPinholeCameraFactor::Linearization>(lins[i]);

    const auto &pose = factor->getPoses()[0];

    const auto &JPose =
        Eigen::Map<const Matrix<2, 6>>(lin->jacobians[0][0].data());
    const auto &JVertex =
        Eigen::Map<const Matrix<2, P>>(lin->jacobians[3][0].data());

    const auto &gPose = lin->gPose;
    const auto &gVertex = Eigen::Map<const Vector<P>>(lin->gVertex.data());

    auto &M = mvM[pose];

    auto &mx = mvmx[pose];
    auto &mp = mvmp[pose];

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;

    M.template block<VertexParamSize, DPoseSize>(DVertexParamOffset,
                                                 DPoseOffset)
        .noalias() += weight * JVertex.transpose() * JPose;

    M.template block<VertexParamSize, VertexParamSize>(DVertexParamOffset,
                                                       DVertexParamOffset)
        .noalias() += weight * JVertex.transpose() * JVertex;

    mx += weight * gPose;
    mp.template segment<VertexParamSize>(DVertexParamOffset - DParamOffset) +=
        weight * gVertex;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateFullJointPinholeCameraFactorGaussNewton()
    const {
  const auto &options = mOptions;

  const auto &factors = mvFactors[FactorIndex::FullJointPinholeCamera];
  const auto &evals = mvFactorEvals[0][FactorIndex::FullJointPinholeCamera];
  const auto &lins = mvFactorLins[FactorIndex::FullJointPinholeCamera];

  const auto &weight = options.weights[FactorIndex::FullJointPinholeCamera];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin = std::dynamic_pointer_cast<
        const FullJointPinholeCameraFactor::Linearization>(lins[i]);

    const auto &pose = factor->getPoses()[0];
    const auto &cam = factor->getParams()[0];

    const auto &JPose =
        Eigen::Map<const Matrix<2, 6>>(lin->jacobians[0][0].data());
    const auto &JCam =
        Eigen::Map<const Matrix<2, 4>>(lin->jacobians[3][0].data());

    const auto &gPose = lin->gPose;
    const auto &gCam = lin->gCam;

    auto &M = mvM[pose];

    auto &mx = mvmx[pose];
    auto &mp = mvmp[pose];

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;
    M.template block<4, DPoseSize>(DCamParamOffset, DPoseOffset).noalias() +=
        weight * JCam.transpose() * JPose;
    M.template block<4, 4>(DCamParamOffset, DCamParamOffset) =
        weight * JCam.transpose() * JCam;

    mx += weight * gPose;
    mp.template segment<4>(DCamParamOffset - DParamOffset) += weight * gCam;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateFullVertexPinholeCameraFactorGaussNewton()
    const {
  const auto &options = mOptions;

  const auto &factors = mvFactors[FactorIndex::FullVertexPinholeCamera];
  const auto &evals = mvFactorEvals[0][FactorIndex::FullVertexPinholeCamera];
  const auto &lins = mvFactorLins[FactorIndex::FullVertexPinholeCamera];

  const auto &weight = options.weights[FactorIndex::FullVertexPinholeCamera];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin = std::dynamic_pointer_cast<
        const FullVertexPinholeCameraFactor::Linearization>(lins[i]);

    const auto &pose = factor->getPoses()[0];
    const auto &cam = factor->getParams()[0];

    const auto &JPose =
        Eigen::Map<const Matrix<2, 6>>(lin->jacobians[0][0].data());
    const auto &JVertex =
        Eigen::Map<const Matrix<2, P>>(lin->jacobians[3][0].data());
    const auto &JCam =
        Eigen::Map<const Matrix<2, 4>>(lin->jacobians[3][1].data());

    const auto &gPose = lin->gPose;
    const auto &gVertex = Eigen::Map<const Vector<P>>(lin->gVertex.data());
    const auto &gCam = lin->gCam;

    auto &M = mvM[pose];

    auto &mx = mvmx[pose];
    auto &mp = mvmp[pose];

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;
    M.template block<VertexParamSize, DPoseSize>(DVertexParamOffset,
                                                 DPoseOffset)
        .noalias() += weight * JVertex.transpose() * JPose;
    M.template block<4, DPoseSize>(DCamParamOffset, DPoseOffset).noalias() +=
        weight * JCam.transpose() * JPose;
    M.template block<VertexParamSize, VertexParamSize>(DVertexParamOffset,
                                                       DVertexParamOffset) =
        weight * JVertex.transpose() * JVertex;
    M.template block<VertexParamSize, 4>(DVertexParamOffset, DCamParamOffset) =
        weight * JVertex.transpose() * JCam;
    M.template block<4, VertexParamSize>(DCamParamOffset, DVertexParamOffset) =
        M.template block<VertexParamSize, 4>(DVertexParamOffset,
                                             DCamParamOffset)
            .transpose();
    M.template block<4, 4>(DCamParamOffset, DCamParamOffset) =
        weight * JCam.transpose() * JCam;

    mx += weight * gPose;
    mp.template segment<VertexParamSize>(DVertexParamOffset - DParamOffset) +=
        weight * gVertex;
    mp.template segment<4>(DCamParamOffset - DParamOffset) += weight * gCam;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateUnitPOFFactorGaussNewton() const {
  const auto &factors = mvFactors[FactorIndex::UnitPOF];
  const auto &evals = mvFactorEvals[0][FactorIndex::UnitPOF];
  const auto &lins = mvFactorLins[FactorIndex::UnitPOF];

  const auto &weight = mOptions.weights[FactorIndex::UnitPOF];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin =
        std::dynamic_pointer_cast<const UnitPOFFactor::Linearization>(lins[i]);

    const auto &pose = factor->getPoses()[0];

    assert(pose >= 0 && pose < NumPoses);

    const auto &JR =
        Eigen::Map<const Matrix<3, 3>>(lin->jacobians[0][0].data());

    const auto &gR = Eigen::Map<const Vector3>(lin->gPose.data());

    auto &M = mvM[pose];
    auto &mx = mvmx[pose];

    M.template topLeftCorner<3, 3>().noalias() += weight * JR.transpose() * JR;

    mx.template head<3>() += weight * gR;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateScaledPOFFactorGaussNewton() const {
  const auto &factors = mvFactors[FactorIndex::ScaledPOF];
  const auto &evals = mvFactorEvals[0][FactorIndex::ScaledPOF];
  const auto &lins = mvFactorLins[FactorIndex::ScaledPOF];

  const auto &weight = mOptions.weights[FactorIndex::ScaledPOF];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin =
        std::dynamic_pointer_cast<const ScaledPOFFactor::Linearization>(
            lins[i]);

    const auto &pose = factor->getPoses()[0];

    assert(pose >= 0 && pose < NumPoses);

    const auto &JR =
        Eigen::Map<const Matrix<3, 3>>(lin->jacobians[0][0].data());
    const auto &JVertex =
        Eigen::Map<const Matrix<3, P>>(lin->jacobians[3][0].data());

    const auto &gR = Eigen::Map<const Vector3>(lin->gPose.data());
    const auto &gVertex = Eigen::Map<const Vector<P>>(lin->gVertex.data());

    auto &M = mvM[pose];
    auto &mx = mvmx[pose];
    auto &mp = mvmp[pose];

    M.template topLeftCorner<3, 3>().noalias() += weight * JR.transpose() * JR;

    M.template block<VertexParamSize, 3>(DVertexParamOffset, DPoseOffset)
        .noalias() += weight * JVertex.transpose() * JR;

    M.template block<VertexParamSize, VertexParamSize>(DVertexParamOffset,
                                                       DVertexParamOffset)
        .noalias() += weight * JVertex.transpose() * JVertex;

    mx.template head<3>() += weight * gR;

    mp.template segment<VertexParamSize>(DVertexParamOffset - DParamOffset) +=
        weight * gVertex;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateRelPOFFactorGaussNewton() const {
  const auto &factors = mvFactors[FactorIndex::RelPOF];
  const auto &evals = mvFactorEvals[0][FactorIndex::RelPOF];
  const auto &lins = mvFactorLins[FactorIndex::RelPOF];

  const auto &weight = mOptions.weights[FactorIndex::RelPOF];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin =
        std::dynamic_pointer_cast<const RelPOFFactor::Linearization>(lins[i]);

    const auto &pose = factor->getPoses()[0];

    assert(pose > 0 && pose < NumPoses);

    const auto &JR =
        Eigen::Map<const Matrix<3, 3>>(lin->jacobians[0][0].data());
    const auto &JVertex =
        Eigen::Map<const Matrix<3, P>>(lin->jacobians[3][0].data());
    const auto &JJoint =
        Eigen::Map<const Matrix<3, 3>>(lin->jacobians[2][0].data());

    const auto &gR = Eigen::Map<const Vector3>(lin->gPose.data());
    const auto &gVertex = Eigen::Map<const Vector<P>>(lin->gVertex.data());
    const auto &gJoint = lin->gJoint;

    auto &M = mvM[pose];
    auto &mx = mvmx[pose];
    auto &mu = mvmu[pose];
    auto &mp = mvmp[pose];

    M.template topLeftCorner<3, 3>().noalias() += weight * JR.transpose() * JR;

    M.template block<VertexParamSize, 3>(DVertexParamOffset, DPoseOffset)
        .noalias() += weight * JVertex.transpose() * JR;

    M.template block<DJointSize, 3>(DJointOffset, DPoseOffset).noalias() +=
        weight * JJoint.transpose() * JR;

    M.template block<VertexParamSize, VertexParamSize>(DVertexParamOffset,
                                                       DVertexParamOffset)
        .noalias() += weight * JVertex.transpose() * JVertex;

    M.template block<DJointSize, VertexParamSize>(DJointOffset,
                                                  DVertexParamOffset)
        .noalias() += weight * JJoint.transpose() * JVertex;

    M.template block<DJointSize, DJointSize>(DJointOffset, DJointOffset)
        .noalias() += weight * JJoint.transpose() * JJoint;

    mx.template head<3>() += weight * gR;

    mp.template segment<VertexParamSize>(DVertexParamOffset - DParamOffset) +=
        weight * gVertex;

    mu.noalias() += weight * gJoint;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateJointDepthCameraFactorGaussNewton()
    const {
  const auto &factors = mvFactors[FactorIndex::JointDepthCamera];
  const auto &evals = mvFactorEvals[0][FactorIndex::JointDepthCamera];
  const auto &lins = mvFactorLins[FactorIndex::JointDepthCamera];

  const auto &weight = mOptions.weights[FactorIndex::JointDepthCamera];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin =
        std::dynamic_pointer_cast<const JointDepthCameraFactor::Linearization>(
            lins[i]);

    const auto &pose = factor->getPoses()[0];

    const auto &JPose =
        Eigen::Map<const Matrix<3, 6>>(lin->jacobians[0][0].data());

    const auto &gPose = lin->gPose;

    auto &M = mvM[pose];
    auto &mx = mvmx[pose];

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;
    mx += weight * gPose;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateVertexDepthCameraFactorGaussNewton()
    const {
  const auto &options = mOptions;

  const auto &factors = mvFactors[FactorIndex::VertexDepthCamera];
  const auto &evals = mvFactorEvals[0][FactorIndex::VertexDepthCamera];
  const auto &lins = mvFactorLins[FactorIndex::VertexDepthCamera];

  const auto &weight = options.weights[FactorIndex::VertexDepthCamera];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin =
        std::dynamic_pointer_cast<const VertexDepthCameraFactor::Linearization>(
            lins[i]);

    const auto &pose = factor->getPoses()[0];

    const auto &JPose =
        Eigen::Map<const Matrix<3, 6>>(lin->jacobians[0][0].data());
    const auto &JVertex =
        Eigen::Map<const Matrix<3, P>>(lin->jacobians[3][0].data());

    const auto &gPose = lin->gPose;
    const auto &gVertex = Eigen::Map<const Vector<P>>(lin->gVertex.data());

    auto &M = mvM[pose];

    auto &mx = mvmx[pose];
    auto &mp = mvmp[pose];

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;

    M.template block<VertexParamSize, DPoseSize>(DVertexParamOffset,
                                                 DPoseOffset)
        .noalias() += weight * JVertex.transpose() * JPose;

    M.template block<VertexParamSize, VertexParamSize>(DVertexParamOffset,
                                                       DVertexParamOffset)
        .noalias() += weight * JVertex.transpose() * JVertex;

    mx += weight * gPose;
    mp.template segment<VertexParamSize>(DVertexParamOffset - DParamOffset) +=
        weight * gVertex;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateJointLimitFactorGaussNewton() const {
  const auto &options = mOptions;

  const auto &factors = mvFactors[FactorIndex::JointLimit];
  const auto &evals = mvFactorEvals[0][FactorIndex::JointLimit];
  const auto &lins = mvFactorLins[FactorIndex::JointLimit];

  const auto &weight = options.weights[FactorIndex::JointLimit];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin = lins[i];
    const auto &eval = evals[i];

    const auto &pose = factor->getJoints()[0] + 1;

    const auto &JJoint =
        Eigen::Map<const Matrix<6, DJointSize>>(lin->jacobians[2][0].data());
    const auto &Error = eval->error;

    auto &M = mvM[pose];
    auto &mu = mvmu[pose];

    M.template bottomRightCorner<DJointSize, DJointSize>().noalias() +=
        weight * JJoint.transpose() * JJoint;
    mu.noalias() += weight * JJoint.transpose() * Error;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateCollisionFactorGaussNewton() const {
  // TODO: add collision factors
  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updatePoseConstFactorGaussNewton() const {
  const auto &options = mOptions;

  const auto &factors = mvFactors[FactorIndex::PoseConst];
  const auto &evals = mvFactorEvals[0][FactorIndex::PoseConst];
  const auto &lins = mvFactorLins[FactorIndex::PoseConst];

  const auto &weight = options.weights[FactorIndex::PoseConst];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin = lins[i];
    const auto &eval = evals[i];

    const auto &pose = factor->getPoses()[0];

    const auto &JPose = Eigen::Map<const Matrix6>(lin->jacobians[0][0].data());
    const auto &Error = Eigen::Map<const Vector6>(eval->error.data());

    auto &M = mvM[pose];
    auto &mx = mvmx[pose];

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;
    mx.noalias() += weight * JPose.transpose() * Error;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateJointConstFactorGaussNewton() const {
  const auto &options = mOptions;

  const auto &factors = mvFactors[FactorIndex::JointConst];
  const auto &evals = mvFactorEvals[0][FactorIndex::JointConst];
  const auto &lins = mvFactorLins[FactorIndex::JointConst];

  const auto &weight = options.weights[FactorIndex::JointConst];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin = lins[i];
    const auto &eval = evals[i];

    const auto &pose = factor->getJoints()[0] + 1;

    const auto &JJoint = Eigen::Map<const Matrix3>(lin->jacobians[2][0].data());
    const auto &Error = Eigen::Map<const Vector3>(eval->error.data());

    auto &M = mvM[pose];
    auto &mu = mvmu[pose];

    M.template bottomRightCorner<DJointSize, DJointSize>().noalias() +=
        weight * JJoint.transpose() * JJoint;
    mu.noalias() += weight * JJoint.transpose() * Error;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updatePoseFactorGaussNewton() const {
  const auto &options = mOptions;

  const auto &factors = mvFactors[FactorIndex::Pose];
  const auto &evals = mvFactorEvals[0][FactorIndex::Pose];
  const auto &lins = mvFactorLins[FactorIndex::Pose];

  const auto &weight = options.weights[FactorIndex::Pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin = lins[i];
    const auto &eval = evals[i];

    const auto &pose = factor->getPoses()[0];

    const auto &JPose = Eigen::Map<const Matrix6>(lin->jacobians[0][0].data());
    const auto &Error = Eigen::Map<const Vector6>(eval->error.data());

    auto &M = mvM[pose];
    auto &mx = mvmx[pose];

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;
    mx.noalias() += weight * JPose.transpose() * Error;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateShapeFactorGaussNewton() const {
  // TODO: add shape factors

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateJointFactorGaussNewton() const {
  const auto &options = mOptions;

  const auto &factors = mvFactors[FactorIndex::Joint];
  const auto &evals = mvFactorEvals[0][FactorIndex::Joint];
  const auto &lins = mvFactorLins[FactorIndex::Joint];

  const auto &weight = options.weights[FactorIndex::Joint];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin = lins[i];
    const auto &eval = evals[i];

    const auto &pose = factor->getJoints()[0] + 1;

    const auto &JJoint = Eigen::Map<const Matrix3>(lin->jacobians[2][0].data());
    const auto &Error = Eigen::Map<const Vector3>(eval->error.data());

    auto &M = mvM[pose];
    auto &mu = mvmu[pose];

    M.template bottomRightCorner<DJointSize, DJointSize>().noalias() +=
        weight * JJoint.transpose() * JJoint;
    mu.noalias() += weight * JJoint.transpose() * Error;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateParamFactorGaussNewton() const {
  const auto &options = mOptions;

  const auto &factors = mvFactors[FactorIndex::Parameter];
  const auto &evals = mvFactorEvals[0][FactorIndex::Parameter];
  const auto &lins = mvFactorLins[FactorIndex::Parameter];

  const auto &weight = options.weights[FactorIndex::Parameter];

  auto &M = mvM[0];
  auto &mp = mvmp[0];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin = lins[i];
    const auto &eval = evals[i];

    const auto &param = factor->getParams()[0];

    switch (param) {
      case VertexParamIndex: {
        const auto &JParam =
            Eigen::Map<const Matrix<P, P>>(lin->jacobians[3][0].data());
        const auto &Error = Eigen::Map<const Vector<P>>(eval->error.data());

        M.template block<VertexParamSize, VertexParamSize>(DVertexParamOffset,
                                                           DVertexParamOffset)
            .noalias() += weight * JParam.transpose() * JParam;
        mp.template segment<VertexParamSize>(DVertexParamOffset -
                                             DParamOffset) +=
            weight * JParam.transpose() * Error;

        break;
      }

      case CamParamIndex: {
        const auto &JParam =
            Eigen::Map<const Matrix<4, 4>>(lin->jacobians[3][0].data());
        const auto &Error = Eigen::Map<const Vector<4>>(eval->error.data());

        M.template block<4, 4>(DCamParamOffset, DCamParamOffset).noalias() +=
            weight * JParam.transpose() * JParam;
        mp.template segment<4>(DCamParamOffset - DParamOffset) +=
            weight * JParam.transpose() * Error;

        break;
      }

      case FaceParamIndex: {
        // TODO: add face parameters
        break;
      }
    }
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateJointPinholeCameraFactorGaussNewton(
    int pose) const {
  static const int index = FactorIndex::JointPinholeCamera;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mx = mvmx[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getPoses()[0]);

    const auto &lin = std::dynamic_pointer_cast<
        const JointPinholeCameraFactor::Linearization>(lins[i]);

    const auto &JPose =
        Eigen::Map<const Matrix<2, 6>>(lin->jacobians[0][0].data());

    const auto &gPose = lin->gPose;

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;
    mx += weight * gPose;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateFullJointPinholeCameraFactorGaussNewton(
    int pose) const {
  static const int index = FactorIndex::FullJointPinholeCamera;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mx = mvmx[pose];
  auto &mp = mvmp[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getPoses()[0]);

    const auto &lin = std::dynamic_pointer_cast<
        const FullJointPinholeCameraFactor::Linearization>(lins[i]);

    const auto &JPose =
        Eigen::Map<const Matrix<2, 6>>(lin->jacobians[0][0].data());
    const auto &JCam =
        Eigen::Map<const Matrix<2, 4>>(lin->jacobians[3][0].data());

    const auto &gPose = lin->gPose;
    const auto &gCam = lin->gCam;

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;
    M.template block<4, DPoseSize>(DCamParamOffset, DPoseOffset).noalias() +=
        weight * JCam.transpose() * JPose;
    M.template block<4, 4>(DCamParamOffset, DCamParamOffset) =
        weight * JCam.transpose() * JCam;

    mx += weight * gPose;
    mp.template segment<4>(DCamParamOffset) += weight * gCam;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateVertexPinholeCameraFactorGaussNewton(
    int pose) const {
  static const int index = FactorIndex::VertexPinholeCamera;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mx = mvmx[pose];
  auto &mp = mvmp[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getPoses()[0]);

    const auto &lin = std::dynamic_pointer_cast<
        const VertexPinholeCameraFactor::Linearization>(lins[i]);

    const auto &JPose =
        Eigen::Map<const Matrix<2, 6>>(lin->jacobians[0][0].data());
    const auto &JVertex =
        Eigen::Map<const Matrix<2, P>>(lin->jacobians[3][0].data());

    const auto &gPose = lin->gPose;
    const auto &gVertex = Eigen::Map<const Vector<P>>(lin->gVertex.data());

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;

    M.template block<VertexParamSize, DPoseSize>(DVertexParamOffset,
                                                 DPoseOffset)
        .noalias() += weight * JVertex.transpose() * JPose;

    M.template block<VertexParamSize, VertexParamSize>(DVertexParamOffset,
                                                       DVertexParamOffset)
        .noalias() += weight * JVertex.transpose() * JVertex;

    mx += weight * gPose;
    mp.template segment<VertexParamSize>(DVertexParamOffset - DParamOffset) +=
        weight * gVertex;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateFullVertexPinholeCameraFactorGaussNewton(
    int pose) const {
  static const int index = FactorIndex::FullVertexPinholeCamera;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mx = mvmx[pose];
  auto &mp = mvmp[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getPoses()[0]);

    const auto &lin = std::dynamic_pointer_cast<
        const FullVertexPinholeCameraFactor::Linearization>(lins[i]);

    const auto &JPose =
        Eigen::Map<const Matrix<2, 6>>(lin->jacobians[0][0].data());
    const auto &JVertex =
        Eigen::Map<const Matrix<2, P>>(lin->jacobians[3][0].data());
    const auto &JCam =
        Eigen::Map<const Matrix<2, 4>>(lin->jacobians[3][1].data());

    const auto &gPose = lin->gPose;
    const auto &gVertex = Eigen::Map<const Vector<P>>(lin->gVertex.data());
    const auto &gCam = lin->gCam;

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;
    M.template block<VertexParamSize, DPoseSize>(DVertexParamOffset,
                                                 DPoseOffset)
        .noalias() += weight * JVertex.transpose() * JPose;
    M.template block<4, DPoseSize>(DCamParamOffset, DPoseOffset).noalias() +=
        weight * JCam.transpose() * JPose;
    M.template block<VertexParamSize, VertexParamSize>(DVertexParamOffset,
                                                       DVertexParamOffset) =
        weight * JVertex.transpose() * JVertex;
    M.template block<VertexParamSize, 4>(DVertexParamOffset, DCamParamOffset) =
        weight * JVertex.transpose() * JCam;
    M.template block<4, VertexParamSize>(DCamParamOffset, DVertexParamOffset) =
        M.template block<VertexParamSize, 4>(DVertexParamOffset,
                                             DCamParamOffset)
            .transpose();
    M.template block<4, 4>(DCamParamOffset, DCamParamOffset) =
        weight * JCam.transpose() * JCam;

    mx += weight * gPose;
    mp.template segment<VertexParamSize>(DVertexParamOffset - DParamOffset) +=
        weight * gVertex;
    mp.template segment<4>(DCamParamOffset - DParamOffset) += weight * gCam;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateUnitPOFFactorGaussNewton(int pose) const {
  static const int index = FactorIndex::UnitPOF;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mx = mvmx[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getPoses()[0]);

    const auto &lin =
        std::dynamic_pointer_cast<const UnitPOFFactor::Linearization>(lins[i]);

    const auto &JR =
        Eigen::Map<const Matrix<3, 3>>(lin->jacobians[0][0].data());

    const auto &gR = Eigen::Map<const Vector3>(lin->gPose.data());

    M.template topLeftCorner<3, 3>().noalias() += weight * JR.transpose() * JR;

    mx.template head<3>() += weight * gR;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateScaledPOFFactorGaussNewton(
    int pose) const {
  static const int index = FactorIndex::ScaledPOF;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mx = mvmx[pose];
  auto &mp = mvmp[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getPoses()[0]);

    const auto &lin =
        std::dynamic_pointer_cast<const ScaledPOFFactor::Linearization>(
            lins[i]);

    const auto &JR =
        Eigen::Map<const Matrix<3, 3>>(lin->jacobians[0][0].data());
    const auto &JVertex =
        Eigen::Map<const Matrix<3, P>>(lin->jacobians[3][0].data());

    const auto &gR = Eigen::Map<const Vector3>(lin->gPose.data());
    const auto &gVertex = Eigen::Map<const Vector<P>>(lin->gVertex.data());

    M.template topLeftCorner<3, 3>().noalias() += weight * JR.transpose() * JR;

    M.template block<VertexParamSize, 3>(DVertexParamOffset, DPoseOffset)
        .noalias() += weight * JVertex.transpose() * JR;

    M.template block<VertexParamSize, VertexParamSize>(DVertexParamOffset,
                                                       DVertexParamOffset)
        .noalias() += weight * JVertex.transpose() * JVertex;

    mx.template head<3>() += weight * gR;

    mp.template segment<VertexParamSize>(DVertexParamOffset - DParamOffset) +=
        weight * gVertex;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateRelPOFFactorGaussNewton(int pose) const {
  static const int index = FactorIndex::RelPOF;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  assert(pose > 0 && pose < NumPoses);

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mx = mvmx[pose];
  auto &mu = mvmu[pose];
  auto &mp = mvmp[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getPoses()[0]);

    const auto &lin =
        std::dynamic_pointer_cast<const RelPOFFactor::Linearization>(lins[i]);

    const auto &JR =
        Eigen::Map<const Matrix<3, 3>>(lin->jacobians[0][0].data());
    const auto &JVertex =
        Eigen::Map<const Matrix<3, P>>(lin->jacobians[3][0].data());
    const auto &JJoint =
        Eigen::Map<const Matrix<3, 3>>(lin->jacobians[2][0].data());

    const auto &gR = Eigen::Map<const Vector3>(lin->gPose.data());
    const auto &gVertex = Eigen::Map<const Vector<P>>(lin->gVertex.data());
    const auto &gJoint = lin->gJoint;

    M.template topLeftCorner<3, 3>().noalias() += weight * JR.transpose() * JR;

    M.template block<VertexParamSize, 3>(DVertexParamOffset, DPoseOffset)
        .noalias() += weight * JVertex.transpose() * JR;

    M.template block<DJointSize, 3>(DJointOffset, DPoseOffset).noalias() +=
        weight * JJoint.transpose() * JR;

    M.template block<VertexParamSize, VertexParamSize>(DVertexParamOffset,
                                                       DVertexParamOffset)
        .noalias() += weight * JVertex.transpose() * JVertex;

    M.template block<DJointSize, VertexParamSize>(DJointOffset,
                                                  DVertexParamOffset)
        .noalias() += weight * JJoint.transpose() * JVertex;

    M.template block<DJointSize, DJointSize>(DJointOffset, DJointOffset)
        .noalias() += weight * JJoint.transpose() * JJoint;

    mx.template head<3>() += weight * gR;

    mp.template segment<VertexParamSize>(DVertexParamOffset - DParamOffset) +=
        weight * gVertex;

    mu += weight * gJoint;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateJointDepthCameraFactorGaussNewton(
    int pose) const {
  static const int index = FactorIndex::JointDepthCamera;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mx = mvmx[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getPoses()[0]);

    const auto &lin =
        std::dynamic_pointer_cast<const JointDepthCameraFactor::Linearization>(
            lins[i]);

    const auto &JPose =
        Eigen::Map<const Matrix<3, 6>>(lin->jacobians[0][0].data());

    const auto &gPose = lin->gPose;

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;
    mx += weight * gPose;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateVertexDepthCameraFactorGaussNewton(
    int pose) const {
  static const int index = FactorIndex::VertexDepthCamera;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mx = mvmx[pose];
  auto &mp = mvmp[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getPoses()[0]);

    const auto &lin =
        std::dynamic_pointer_cast<const VertexDepthCameraFactor::Linearization>(
            lins[i]);

    const auto &JPose =
        Eigen::Map<const Matrix<3, 6>>(lin->jacobians[0][0].data());
    const auto &JVertex =
        Eigen::Map<const Matrix<3, P>>(lin->jacobians[3][0].data());

    const auto &gPose = lin->gPose;
    const auto &gVertex = Eigen::Map<const Vector<P>>(lin->gVertex.data());

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;

    M.template block<VertexParamSize, DPoseSize>(DVertexParamOffset,
                                                 DPoseOffset)
        .noalias() += weight * JVertex.transpose() * JPose;

    M.template block<VertexParamSize, VertexParamSize>(DVertexParamOffset,
                                                       DVertexParamOffset)
        .noalias() += weight * JVertex.transpose() * JVertex;

    mx += weight * gPose;
    mp.template segment<VertexParamSize>(DVertexParamOffset - DParamOffset) +=
        weight * gVertex;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateJointLimitFactorGaussNewton(
    int pose) const {
  static const int index = FactorIndex::JointLimit;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mu = mvmu[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getJoints()[0] + 1);

    const auto &lin = lins[i];
    const auto &eval = evals[i];

    const auto &JJoint =
        Eigen::Map<const Matrix<6, DJointSize>>(lin->jacobians[2][0].data());
    const auto &Error = eval->error;

    M.template bottomRightCorner<DJointSize, DJointSize>().noalias() +=
        weight * JJoint.transpose() * JJoint;
    mu.noalias() += weight * JJoint.transpose() * Error;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateCollisionFactorGaussNewton(
    int pose) const {
  // TODO: add collision factors
  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updatePoseConstFactorGaussNewton(
    int pose) const {
  static const int index = FactorIndex::PoseConst;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mx = mvmx[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getPoses()[0]);

    const auto &lin = lins[i];
    const auto &eval = evals[i];

    const auto &JPose = Eigen::Map<const Matrix6>(lin->jacobians[0][0].data());
    const auto &Error = Eigen::Map<const Vector6>(eval->error.data());

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;
    mx.noalias() += weight * JPose.transpose() * Error;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateJointConstFactorGaussNewton(
    int pose) const {
  static const int index = FactorIndex::JointConst;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mu = mvmu[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getJoints()[0] + 1);

    const auto &lin = lins[i];
    const auto &eval = evals[i];

    const auto &JJoint = Eigen::Map<const Matrix3>(lin->jacobians[2][0].data());
    const auto &Error = Eigen::Map<const Vector3>(eval->error.data());

    M.template bottomRightCorner<DJointSize, DJointSize>().noalias() +=
        weight * JJoint.transpose() * JJoint;
    mu.noalias() += weight * JJoint.transpose() * Error;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updatePoseFactorGaussNewton(int pose) const {
  static const int index = FactorIndex::Pose;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mx = mvmx[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getPoses()[0]);

    const auto &lin = lins[i];
    const auto &eval = evals[i];

    const auto &JPose = Eigen::Map<const Matrix6>(lin->jacobians[0][0].data());
    const auto &Error = Eigen::Map<const Vector6>(eval->error.data());

    M.template topLeftCorner<DPoseSize, DPoseSize>().noalias() +=
        weight * JPose.transpose() * JPose;
    mx.noalias() += weight * JPose.transpose() * Error;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateShapeFactorGaussNewton(int pose) const {
  // TODO: add shape factors

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateJointFactorGaussNewton(int pose) const {
  static const int index = FactorIndex::Joint;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mu = mvmu[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    assert(pose == factor->getJoints()[0] + 1);

    const auto &lin = lins[i];
    const auto &eval = evals[i];

    const auto &JJoint = Eigen::Map<const Matrix3>(lin->jacobians[2][0].data());
    const auto &Error = Eigen::Map<const Vector3>(eval->error.data());

    M.template bottomRightCorner<DJointSize, DJointSize>().noalias() +=
        weight * JJoint.transpose() * JJoint;
    mu.noalias() += weight * JJoint.transpose() * Error;
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateParamFactorGaussNewton(int pose) const {
  static const int index = FactorIndex::Parameter;

  assert(pose >= 0 && pose < NumPoses);

  if (pose < 0 || pose >= NumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &factors = mnvFactors[pose][index];

  if (factors.empty()) {
    return 0;
  }

  assert(pose == 0);

  const auto &evals = mnvFactorEvals[0][pose][index];
  const auto &lins = mnvFactorLins[pose][index];

  const auto &weight = mOptions.weights[index];

  auto &M = mvM[pose];
  auto &mp = mvmp[pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = factors[i];

    const auto &lin = lins[i];
    const auto &eval = evals[i];

    const auto &param = factor->getParams()[0];

    switch (param) {
      case VertexParamIndex: {
        const auto &JParam =
            Eigen::Map<const Matrix<P, P>>(lin->jacobians[3][0].data());
        const auto &Error = Eigen::Map<const Vector<P>>(eval->error.data());

        M.template block<VertexParamSize, VertexParamSize>(DVertexParamOffset,
                                                           DVertexParamOffset)
            .noalias() += weight * JParam.transpose() * JParam;
        mp.template segment<VertexParamSize>(DVertexParamOffset -
                                             DParamOffset) +=
            weight * JParam.transpose() * Error;

        break;
      }

      case CamParamIndex: {
        const auto &JParam =
            Eigen::Map<const Matrix<4, 4>>(lin->jacobians[3][0].data());
        const auto &Error = Eigen::Map<const Vector<4>>(eval->error.data());

        M.template block<4, 4>(DCamParamOffset, DCamParamOffset).noalias() +=
            weight * JParam.transpose() * JParam;
        mp.template segment<4>(DCamParamOffset - DParamOffset) +=
            weight * JParam.transpose() * Error;

        break;
      }

      case FaceParamIndex: {
        // TODO: add face parameters
        break;
      }
    }
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::computeGradient() const {
  for (int i = NumPoses - 1; i > 0; i--) {
    backwardDG(i);
  }

  const auto &model = mModel;

  const auto &links = model->getLinks();
  const auto &root = links[0];

  mvPoseDG[0] = mvmx[0];
  mParamDG = mvmp[0];

  for (const auto &j : root.children()) {
    const auto &jj = links[j].joint();

    mvPoseDG[0] += mvPoseDG[j];
    mParamDG += mvParamDG[jj];
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::solveGaussNewton() const {
  for (int i = NumPoses - 1; i > 0; i--) {
    factorize(i);
    backward(i);
  }

  factorizeN();
  backwardN();

  for (int i = 1; i < NumPoses; i++) {
    forward(i);
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateDCost() const {
  mDCost.setZero();

  for (int i = 1, ii = 0; i < NumPoses; i++, ii++) {
    mvDCostDx[i].noalias() =
        0.5 * mvM[i].template topLeftCorner<DPoseSize, DPoseSize>() *
        mvPoseGN[i];
    mDCost[0] += mvPoseGN[i].dot(mvDCostDx[i]);

    mvDCostDp[i].noalias() = mvM[i].template block<ParamSize, DPoseSize>(
                                 DCamParamOffset, DPoseOffset) *
                             mvPoseGN[i];
    mvDCostDp[i].noalias() +=
        0.5 *
        mvM[i].template block<ParamSize, ParamSize>(
            DCamParamOffset, DCamParamOffset, ParamSize, ParamSize) *
        mParamGN.template head<ParamSize>();
    mDCost[0] += mParamGN.template head<ParamSize>().dot(mvDCostDp[i]);

    mvDCostDu[i].noalias() = mvM[i].template block<DJointSize, DPoseSize>(
                                 DJointOffset, DPoseOffset) *
                             mvPoseGN[i];
    mvDCostDu[i].noalias() += mvM[i].template block<DJointSize, ParamSize>(
                                  DJointOffset, DCamParamOffset) *
                              mParamGN.template head<ParamSize>();
    mvDCostDu[i].noalias() +=
        0.5 * mvM[i].template bottomRightCorner<DJointSize, DJointSize>() *
        mvJointGN[ii];
    mDCost[0] += mvJointGN[ii].dot(mvDCostDu[i]);

    mDCost[1] += mvPoseGN[i].dot(mvmx[i]);
    mDCost[1] += mvJointGN[ii].dot(mvmu[i]);
    mDCost[1] += mParamGN.template head<ParamSize>().dot(
        mvmp[i].template head<ParamSize>());
  }

  mvDCostDx[0].noalias() =
      0.5 * mvM[0].template topLeftCorner<DPoseSize, DPoseSize>() * mvPoseGN[0];
  mDCost[0] += mvPoseGN[0].dot(mvDCostDx[0]);

  mvDCostDp[0].noalias() = mvM[0].template block<ParamSize, DPoseSize>(
                               DCamParamOffset, DPoseOffset) *
                           mvPoseGN[0];
  mvDCostDp[0].noalias() += 0.5 *
                            mvM[0].template block<ParamSize, ParamSize>(
                                DCamParamOffset, DCamParamOffset) *
                            mParamGN.template head<ParamSize>();
  mDCost[0] += mParamGN.template head<ParamSize>().dot(mvDCostDp[0]);

  mDCost[1] += mvPoseGN[0].dot(mvmx[0]);
  mDCost[1] += mParamGN.template head<ParamSize>().dot(
      mvmp[0].template head<ParamSize>());

  if (mOptions.check_collisions) {
    // TODO: handle collisions
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
inline int Optimizer<K, P, N, CamOpt>::factorize(int i) const {
  assert(i < NumPoses && i > 0);

  const auto &model = mModel;

  const auto &links = model->getLinks();

  const auto &link = links[i];

  assert(i == link.id());

  const int ii = link.joint();

  const auto &children = link.children();

  mvH[ii] = mvM[i];

  for (const auto &j : children) {
    const auto &jj = links[j].joint();

    mvH[ii]
        .template topLeftCorner<DPoseSize + DParamSize,
                                DPoseSize + DParamSize>() +=
        mvH[jj]
            .template topLeftCorner<DPoseSize + DParamSize,
                                    DPoseSize + DParamSize>();
  }

  mvHxB[ii].noalias() = mvH[ii].template leftCols<6>() * mvB[ii];

  // update Hpx and Hpp
  mvH[ii].template block<VertexParamSize, DPoseSize + DParamSize>(
      DVertexParamOffset, DPoseOffset) +=
      mvHxB[ii]
          .template block<DPoseSize + DParamSize, VertexParamSize>(DPoseOffset,
                                                                   0)
          .transpose();

  // update Hux, Hup, Huu
  mvH[ii].template bottomRows<DJointSize>() +=
      mvHxB[ii].template rightCols<DJointSize>().transpose();

  // update Hpp and Hup
  mvH[ii].template block<DParamSize + DJointSize, VertexParamSize>(
      DParamOffset, DVertexParamOffset) +=
      mvHxB[ii].template block<DParamSize + DJointSize, VertexParamSize>(
          DParamOffset, 0);

  // update Hpp and Hup
  mvH[ii]
      .template block<VertexParamSize + 3, VertexParamSize>(DVertexParamOffset,
                                                            DVertexParamOffset)
      .noalias() += mvHxB[ii].template middleRows<3>(3).transpose() * mvBp[ii];

  // update Huu
  mvH[ii].template block<DJointSize, DJointSize>(DPoseSize + DParamSize,
                                                 DPoseSize + DParamSize) +=
      mvHxB[ii].template block<DJointSize, DJointSize>(DJointOffset,
                                                       VertexParamSize);
  mvH[ii]
      .template block<DJointSize, DJointSize>(DJointOffset, DJointOffset)
      .noalias() +=
      mvBu[ii].transpose() * mvHxB[ii].template block<DPoseSize, DJointSize>(
                                 DPoseOffset, VertexParamSize);

  if (mOptions.method == Method::LM) {
    mvH[ii].diagonal().template segment<DJointSize>(DJointOffset).array() *=
        mDLambda;
    mLambda.template segment<DJointSize>(6 + DParamSize + 3 * ii) =
        mvH[ii].diagonal().template segment<DJointSize>(DJointOffset);
  }

  mvH[ii].diagonal().template segment<DJointSize>(DJointOffset).array() +=
      mOptions.delta;

  mvHuuInv[ii] =
      mvH[ii].template bottomRightCorner<DJointSize, DJointSize>().inverse();

  mvKuxp[ii].template leftCols<DPoseSize + DParamSize>().noalias() =
      -mvHuuInv[ii] *
      mvH[ii].template bottomLeftCorner<DJointSize, DPoseSize + DParamSize>();

  // update Hxx
  mvH[ii]
      .template topLeftCorner<DPoseSize + DParamSize, DPoseSize + DParamSize>()
      .noalias() +=
      mvH[ii]
          .template bottomLeftCorner<DJointSize, DPoseSize + DParamSize>()
          .transpose() *
      mvKuxp[ii];

  return 0;
}

template <int K, int P, int N, bool CamOpt>
inline int Optimizer<K, P, N, CamOpt>::backward(int i) const {
  assert(i < NumPoses && i > 0);

  const auto &model = mModel;

  const auto &links = model->getLinks();

  const auto &link = links[i];

  assert(i == link.id());

  const auto &ii = link.joint();
  const auto &children = link.children();

  mvhx[ii] = mvmx[i];
  mvhp[ii] = mvmp[i];
  mvhu[ii] = mvmu[i];
  mvE[ii] = 0;

  for (const auto &j : children) {
    const auto &jj = links[j].joint();

    mvhx[ii] += mvhx[jj];
    mvhp[ii] += mvhp[jj];
    mvE[ii] += mvE[jj];
  }

  mvhp[ii].template segment<VertexParamSize>(VertexParamOffset).noalias() +=
      mvBp[ii].transpose() * mvhx[ii].template bottomRows<3>();
  mvhu[ii].noalias() += mvBu[ii].transpose() * mvhx[ii];

  mvku[ii].noalias() = -mvHuuInv[ii] * mvhu[ii];

  mvE[ii] += 0.5 * mvhu[ii].dot(mvku[ii]);

  mvhx[ii].noalias() +=
      mvH[ii]
          .template block<DJointSize, DPoseSize>(DJointOffset, DPoseOffset)
          .transpose() *
      mvku[ii];
  mvhp[ii].noalias() +=
      mvH[ii]
          .template block<DJointSize, DParamSize>(DJointOffset, DParamOffset)
          .transpose() *
      mvku[ii];

  return 0;
}

template <int K, int P, int N, bool CamOpt>
inline int Optimizer<K, P, N, CamOpt>::forward(int i) const {
  assert(i < NumPoses && i > 0);

  const auto &model = mModel;

  const auto &link = model->getLinks()[i];

  const auto &parent = link.parent();
  const auto &ii = link.joint();

  mvJointGN[ii] = mvku[ii];

  mvJointGN[ii].noalias() +=
      mvKuxp[ii].template leftCols<6>() * mvPoseGN[parent];

  mvPoseGN[i] = mvPoseGN[parent];
  mvPoseGN[i].noalias() += mvBu[ii] * mvJointGN[ii];
  mvPoseGN[i].template tail<3>().noalias() +=
      mvBp[ii] * mParamGN.template segment<VertexParamSize>(DVertexParamOffset -
                                                            DParamOffset);

  return 0;
}

template <int K, int P, int N, bool CamOpt>
inline int Optimizer<K, P, N, CamOpt>::factorizeN() const {
  const auto &model = mModel;

  const auto &links = model->getLinks();
  const auto &root = links[0];

  mH0 = mvM[0].template topLeftCorner<DJointOffset, DJointOffset>();

  for (const auto &j : root.children()) {
    const auto &jj = links[j].joint();

    mH0 += mvH[jj]
               .template topLeftCorner<DPoseSize + DParamSize,
                                       DPoseSize + DParamSize>();
  }

  mH0.template block<DPoseSize, DParamSize>(DPoseOffset, DParamOffset) =
      mH0.template block<DParamSize, DPoseSize>(DParamOffset, DPoseOffset)
          .transpose();

  if (mOptions.method == Method::LM) {
    mH0.diagonal().array() *= mDLambda;
    mLambda.template head<6 + DParamSize>() = mH0.diagonal();
  }

  mH0.diagonal().array() += mOptions.delta;

  if (mOptions.extra_hip_parameter) {
    assert(mvParams[0].size() == 1);

    mHchol0.compute(
        mRawH0.template topLeftCorner<DPoseSize + DParamSize - 1,
                                      DPoseSize + DParamSize - 1>());
  } else {
    mHchol0.compute(mRawH0);
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
inline int Optimizer<K, P, N, CamOpt>::backwardN() const {
  const auto &model = mModel;

  const auto &links = model->getLinks();
  const auto &root = links[0];

  mh0.template head<DPoseSize>() = mvmx[0];
  mh0.template tail<DParamSize>() = mvmp[0];
  mE = 0;

  for (const auto &j : root.children()) {
    const auto &jj = links[j].joint();

    mh0.template head<DPoseSize>() += mvhx[jj];
    mh0.template tail<DParamSize>() += mvhp[jj];
    mE += mvE[jj];
  }

  if (mOptions.extra_hip_parameter) {
    assert(mvParams[0].size() == 1);

    mhGN.template head<DPoseSize + DParamSize - 1>().noalias() =
        -mHchol0.solve(mh0.template head<DPoseSize + DParamSize - 1>());

    mE += 0.5 * mhGN.template head<DPoseSize + DParamSize - 1>().dot(
                    mh0.template head<DPoseSize + DParamSize - 1>());

    Scalar LowerBnd =
        mOptions.extra_hip_parameter_bnd[0] - mvParams[0][0][DParamSize - 1];
    Scalar UpperBnd =
        mOptions.extra_hip_parameter_bnd[1] - mvParams[0][0][DParamSize - 1];

    Scalar H, b;

    mS.noalias() = mHchol0.solve(
        mH0.template topRightCorner<DPoseSize + DParamSize - 1, 1>());

    H = mH0(DPoseSize + DParamSize - 1, DPoseSize + DParamSize - 1) -
        mS.dot(mH0.template topRightCorner<DPoseSize + DParamSize - 1, 1>());
    b = mS.dot(mh0.template head<DPoseSize + DParamSize - 1>()) -
        mh0[DPoseSize + DParamSize - 1];

    mhGN[DPoseSize + DParamSize - 1] = b / H;
    mhGN[DPoseSize + DParamSize - 1] =
        std::max(mhGN[DPoseSize + DParamSize - 1], LowerBnd);
    mhGN[DPoseSize + DParamSize - 1] =
        std::min(mhGN[DPoseSize + DParamSize - 1], UpperBnd);

    mE += 0.5 * H * mhGN[DPoseSize + DParamSize - 1] *
              mhGN[DPoseSize + DParamSize - 1] -
          b * mhGN[DPoseSize + DParamSize - 1];

    mhGN.template head<DPoseSize + DParamSize - 1>() -=
        mhGN[DPoseSize + DParamSize - 1] * mS;
  } else {
    mhGN = -mHchol0.solve(mh0);

    mE += 0.5 * mh0.dot(mhGN);
  }

  mvPoseGN[0] = mhGN.template head<DPoseSize>();
  mParamGN = mhGN.template tail<DParamSize>();

  mku.noalias() += mKuxp.template rightCols<DParamSize>() * mParamGN;

  return 0;
}

template <int K, int P, int N, bool CamOpt>
inline int Optimizer<K, P, N, CamOpt>::backwardPass(
    const std::vector<int> &subtree) const {
  for (auto it = subtree.crbegin(); it != subtree.crend(); it++) {
    factorize(*it);
    backward(*it);
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
inline int Optimizer<K, P, N, CamOpt>::forwardPass(
    const std::vector<int> &subtree) const {
  for (auto it = subtree.cbegin(); it != subtree.cend(); it++) {
    forward(*it);
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::updateCostReduction() const {
  if (mOptions.method == Method::LM) {
    mLambda *= 0.5 - 0.5 / mDLambda;
    mLambda.array() += 0.5 * mOptions.delta;
    mSquaredError.template head<6 + DParamSize>() = mhGN.cwiseAbs2();
    mSquaredError.template tail<3 * K>() =
        Eigen::Map<const scope::Vector<3 * K>>(mJointGN.data()).cwiseAbs2();
    mExtraCost = mLambda.dot(mSquaredError);
  } else {
    mExtraCost =
        0.5 * mOptions.delta * (mhGN.squaredNorm() + mJointGN.squaredNorm());
  }

  mE -= mExtraCost;

  if (mOptions.check_collisions) {
    // TODO: handle collisions
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::backwardDG(int i) const {
  assert(i < NumPoses && i > 0);

  const auto &model = mModel;

  const auto &links = model->getLinks();

  const auto &link = links[i];

  assert(i == link.id());

  const auto &ii = link.joint();
  const auto &children = link.children();

  mvPoseDG[i] = mvmx[i];
  mvJointDG[ii] = mvmu[i];
  mvParamDG[ii] = mvmp[i];

  for (const auto &j : children) {
    const auto &jj = links[j].joint();

    mvPoseDG[i] += mvPoseDG[j];
    mvParamDG[ii] += mvParamDG[jj];
  }

  mvParamDG[ii]
      .template segment<VertexParamSize>(DVertexParamOffset - DParamOffset)
      .noalias() += mvBp[ii].transpose() * mvPoseDG[i].template bottomRows<3>();
  mvJointDG[ii].noalias() += mvBu[ii].transpose() * mvPoseDG[i];

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::checkFactor(
    std::shared_ptr<const Factor> factor) {
  if (factor == nullptr) {
    LOG(WARNING) << "The factor should not be null." << std::endl;

    exit(-1);
  }

  return 0;
}

template <int K, int P, int N, bool CamOpt>
int Optimizer<K, P, N, CamOpt>::addFactor(
    int index, std::shared_ptr<Factor> factor,
    std::array<std::shared_ptr<Factor::Evaluation>, 2> &evals,
    std::shared_ptr<Factor::Linearization> &lin) {
  assert(index >= 0 && index < NumFactors);

  if (index < 0 || index >= NumFactors) {
    LOG(ERROR) << "The factor index must be valid." << std::endl;

    exit(-1);
  }

  if (checkFactor(factor)) {
    exit(-1);
  }

  evals[0] = factor->newEvaluation();
  evals[1] = factor->newEvaluation();
  lin = factor->newLinearization();

  mFactors.push_back(factor);
  mFactorEvals[0].push_back(evals[0]);
  mFactorEvals[1].push_back(evals[1]);
  mFactorLins.push_back(lin);

  mvFactors[index].push_back(factor);
  mvFactorEvals[0][index].push_back(evals[0]);
  mvFactorEvals[1][index].push_back(evals[1]);
  mvFactorLins[index].push_back(lin);

  if (mStatus != Status::Uninitialized) {
    auto &eval = *evals[0];

    factor->evaluate(mvPoses[0], mvShapes[0], mvJoints[0], mvParams[0], eval);

    if (eval.status != Factor::Status::VALID) {
      LOG(ERROR) << "Failed to evaluate the factor." << std::endl;

      mvFactors[index].pop_back();
      mvFactorEvals[0][index].pop_back();
      mvFactorEvals[1][index].pop_back();
      mvFactorLins[index].pop_back();

      exit(-1);
    }

    mvFobj[0][index] += eval.f;
    mfobj[0] += 0.5 * mOptions.weights[index] * eval.f;
  }

  factor->setID(mFactors.size() - 1);

  assert(factor.get() == mFactors.back().get());
  assert(factor->isActive() == true);

  return 0;
}

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::NumPoses;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::NumShapes;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::NumJoints;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::NumParams;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::MaxNumCollisions;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::CamParamIndex;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::VertexParamIndex;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::FaceParamIndex;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::CamParamOffset;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::VertexParamOffset;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::FaceParamOffset;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::CamParamSize;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::VertexParamSize;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::FaceParamSize;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::ParamSize;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::Size;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::MaxDSize;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::DPoseSize;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::DJointSize;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::DParamSize;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::DSize;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::DPoseOffset;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::DParamOffset;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::DJointOffset;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::DCamParamOffset;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::DVertexParamOffset;

template <int K, int P, int N, bool CamOpt>
const int Optimizer<K, P, N, CamOpt>::DFaceParamOffset;

template <int K, int P, int N, bool CamOpt>
const std::vector<int> Optimizer<K, P, N, CamOpt>::DParamOffsets =
    CamOpt ? std::vector<int>{0, 4} : std::vector<int>{0};

template <int K, int P, int N, bool CamOpt>
const std::vector<int> Optimizer<K, P, N, CamOpt>::ParamSizes =
    CamOpt ? std::vector<int>{4, P} : std::vector<int>{P};

template class Optimizer<2, 2, 20, false>;
template class Optimizer<23, 10, 6890, false>;
template class Optimizer<51, 10, 6890, false>;
template class Optimizer<23, 11, 6890, false>;
template class Optimizer<51, 11, 6890, false>;
}  // namespace Optimizer
}  // namespace scope
