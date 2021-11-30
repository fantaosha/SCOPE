#include <glog/logging.h>

#include <scope/initializer/Initializer.h>
#include <memory>

namespace scope {
namespace Initializer {
Initializer::Initializer(int NumJoints, const Options &options)
    : mOptions(options),
      mStatus(Status::Uninitialized),
      mNumPoses(NumJoints + 1),
      mNumJoints(NumJoints) {
  assert(mNumJoints >= 0);

  mvFactors.resize(NumFactors);

  mvMxx.resize(mNumPoses);
  mvMuu.resize(mNumJoints);

  mvmx.resize(mNumPoses);
  mvmu.resize(mNumJoints);

  mvPoses[0].resize(mNumPoses);
  mvPoses[1].resize(mNumPoses);

  mvJoints[0].resize(mNumJoints);
  mvJoints[1].resize(mNumJoints);

  mvFactorEvals[0].resize(NumFactors);
  mvFactorEvals[1].resize(NumFactors);

  mvFactorLins.resize(NumFactors);

  mfobj[0] = 0;
  mfobj[1] = 0;

  mvFobj[0].resize(NumFactors);
  mvFobj[1].resize(NumFactors);

  mDLambda = 3;
}

int Initializer::initialize(const Pose &pose,
                            const AlignedVector<Matrix3> &joints) const {
  assert(joints.size() == mNumJoints);

  if (joints.size() != mNumJoints) {
    LOG(ERROR) << "There should be " << mNumJoints << " joint states."
               << std::endl;

    exit(-1);
  }

  mvPoses[0][0] = pose;
  mvJoints[0] = joints;

  FKintree(0);

  mvPoses[1] = mvPoses[0];
  mvJoints[1] = mvJoints[0];

  if (evaluate(0)) {
    mStatus = Status::Failed;

    exit(-1);
  }

  mResults = Results();
  mResults.fobjs.push_back(mfobj[0]);

  mStatus = Status::Initialized;

  return 0;
}

int Initializer::solve() const {
  mDLambda = 1.0;
  mStepsize = 1.0;

  for (int iter = 0; iter < mOptions.max_iterations; iter++) {
    optimize();

    if (mResults.expected_rel_cost_reduction.back() <
        mOptions.rel_lin_func_decrease_tol) {
      mStatus = Status::Converged;

      break;
    }
  }

  return 0;
}

int Initializer::updateFactorWeights(
    const Eigen::Matrix<Scalar, NumFactors, 1> &weights) {
  mOptions.weights = weights;

  if (mStatus != Status::Uninitialized) {
    mfobj[0] = 0.5 * mOptions.weights.dot(mvFobj[0]);
  }

  return 0;
}

int Initializer::optimize() const {
  if (mStatus == Status::Uninitialized) {
    LOG(WARNING) << "The optimization must be initialized." << std::endl;

    exit(-1);
  }

  if (mStatus == Status::Failed) {
    LOG(ERROR) << "The optimization failed." << std::endl;

    exit(-1);
  }

  if (linearize()) {
    exit(-1);
  }

  if (solveGaussNewton()) {
    exit(-1);
  }

  const auto &options = mOptions;

  auto &results = mResults;

  const auto &fobj = mfobj;

  mStatus = Status::Aborted;

  results.expected_rel_cost_reduction.push_back(-mE / fobj[0]);

  Scalar stepsize = 1.0;

  update(stepsize);

  if (evaluate(1)) {
    exit(-1);
  }

  if (fobj[1] < fobj[0]) {
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

      if (evaluate(1)) {
        exit(-1);
      }

      if (fobj[1] < fobj[0]) {
        accept();
        mStatus = Status::Accepted;

        mStepsize =
            std::min((Scalar)1.0, stepsize * mOptions.stepsize_increase_ratio);

        break;
      }

      stepsize *= mOptions.stepsize_decrease_ratio;
    }

    mDLambda = std::min(mDLambda * mOptions.lm_lambda_increase,
                        mOptions.lm_lambda_max);
  }

  return 0;
}

int Initializer::reset() {
  mvFactors.clear();
  mFactors.clear();

  mvFactorEvals[0].clear();
  mFactorEvals[0].clear();

  mvFactorEvals[1].clear();
  mFactorEvals[1].clear();

  mvFactorLins.clear();
  mFactorLins.clear();

  mvFactors.resize(NumFactors);

  mvFactorEvals[0].resize(NumFactors);
  mvFactorEvals[1].resize(NumFactors);
  mvFactorLins.resize(NumFactors);

  mvFobj[0].setZero();
  mvFobj[1].setZero();

  mfobj[0] = 0;
  mfobj[1] = 0;

  mStatus = Status::Uninitialized;

  return 0;
}

int Initializer::addPinholeCameraFactor(
    std::shared_ptr<PinholeCameraFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  if (factor->getPose() >= mNumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const int index = FactorIndex::PinholeCamera;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  return factor->getID();
}

int Initializer::addPOFFactor(std::shared_ptr<POFFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  if (factor->getPose() >= mNumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const int index = FactorIndex::POF;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  return factor->getID();
}

int Initializer::addDepthCameraFactor(
    std::shared_ptr<DepthCameraFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  if (factor->getPose() >= mNumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const int index = FactorIndex::DepthCamera;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  return factor->getID();
}

int Initializer::addJointLimitFactor(std::shared_ptr<JointLimitFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  if (factor->getJoint() >= mNumJoints) {
    LOG(ERROR) << "The joint must be valid." << std::endl;

    exit(-1);
  }

  const int index = FactorIndex::JointLimit;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  return factor->getID();
}

int Initializer::addPoseConstFactor(std::shared_ptr<PoseConstFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  if (factor->getPose() >= mNumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const int index = FactorIndex::PoseConst;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  return factor->getID();
}

int Initializer::addJointConstFactor(std::shared_ptr<JointConstFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  assert(factor->getJoint() < mNumJoints);

  if (factor->getJoint() >= mNumJoints) {
    LOG(ERROR) << "The joint must be valid." << std::endl;

    exit(-1);
  }

  const int index = FactorIndex::JointConst;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  return factor->getID();
}

int Initializer::addPoseFactor(std::shared_ptr<PoseFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  if (factor->getPose() >= mNumPoses) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const int index = FactorIndex::Pose;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  return factor->getID();
}

int Initializer::addJointFactor(std::shared_ptr<JointFactor> factor) {
  if (checkFactor(factor)) {
    exit(-1);
  }

  assert(factor->getJoint() < mNumJoints);

  if (factor->getJoint() >= mNumJoints) {
    LOG(ERROR) << "The joint must be valid." << std::endl;

    exit(-1);
  }

  const int index = FactorIndex::Joint;

  std::array<std::shared_ptr<Factor::Evaluation>, 2> evals;
  std::shared_ptr<Factor::Linearization> lin;

  if (addFactor(index, factor, evals, lin)) {
    exit(-1);
  }

  return factor->getID();
}

int Initializer::accept() const {
  mfobj[0] = mfobj[1];

  mvPoses[0].swap(mvPoses[1]);
  mvJoints[0].swap(mvJoints[1]);

  mvFobj[0].swap(mvFobj[1]);

  mvFactorEvals[0].swap(mvFactorEvals[1]);
  mFactorEvals[0].swap(mFactorEvals[1]);

  return 0;
}

int Initializer::solveGaussNewton() const {
  mLambda = mH.diagonal() * (mDLambda - 1);
  mLambda.array() += mOptions.delta;
  mH.diagonal() += mLambda;

  mHchol.compute(mH);
  mhGN = -mHchol.solve(mh);

  mE = 0.5 * mhGN.dot(mh);
  mSquaredError = mhGN.cwiseAbs2();
  mE -= 0.5 * mLambda.dot(mSquaredError);

  return 0;
}

int Initializer::evaluate(int n) const {
  assert(n >= 0 && n <= 1);

  const auto &poses = mvPoses[n];
  const auto &joints = mvJoints[n];

  auto &fobj = mfobj[n];
  auto &Fobj = mvFobj[n];
  auto &Evals = mFactorEvals[n];
  auto &FactorEvals = mvFactorEvals[n];

  Fobj.setZero();

  for (int i = 0; i < mFactors.size(); i++) {
    const auto &factor = mFactors[i];
    auto &eval = *Evals[i];

    factor->evaluate(poses, joints, eval);
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

int Initializer::linearize() const {
  const auto &poses = mvPoses[0];
  const auto &joints = mvJoints[0];

  DFKintree();

  const auto &Evals = mFactorEvals[0];
  auto &Lins = mFactorLins;

  for (int i = 0; i < mFactors.size(); i++) {
    const auto &factor = mFactors[i];
    const auto &eval = *Evals[i];
    auto &lin = *Lins[i];

    assert(eval.status == Factor::Status::VALID);

    factor->linearize(poses, joints, eval, lin);

    assert(lin.status == Factor::Status::VALID);
  }

  for (const auto &lin : Lins) {
    if (lin->status != Factor::Status::VALID) {
      LOG(ERROR) << "Failed to linearize the factor." << std::endl;

      mStatus = Status::Failed;

      exit(-1);
    }
  }

  mvMxx.assign(mNumPoses, Matrix6::Zero());
  mvMuu.assign(mNumJoints, Matrix3::Zero());

  mvmx.assign(mNumPoses, Vector6::Zero());
  mvmu.assign(mNumJoints, Vector3::Zero());

  updatePinholeCameraFactorGaussNewton();
  updatePOFFactorGaussNewton();
  updateDepthCameraFactorGaussNewton();
  updateJointLimitFactorGaussNewton();
  updatePoseConstFactorGaussNewton();
  updateJointConstFactorGaussNewton();
  updatePoseFactorGaussNewton();
  updateJointFactorGaussNewton();

  updateGaussNewton();

  return 0;
}

int Initializer::updatePinholeCameraFactorGaussNewton() const {
  const auto &factors = mvFactors[FactorIndex::PinholeCamera];
  const auto &evals = mvFactorEvals[0][FactorIndex::PinholeCamera];
  const auto &lins = mvFactorLins[FactorIndex::PinholeCamera];

  const auto &weight = mOptions.weights[FactorIndex::PinholeCamera];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor =
        std::dynamic_pointer_cast<const PinholeCameraFactor>(factors[i]);

    const auto &lin =
        std::dynamic_pointer_cast<const PinholeCameraFactor::Linearization>(
            lins[i]);

    const auto &J = lin->jacobian;
    const auto &g = lin->g;

    const int &pose = factor->getPose();

    mvMxx[pose].noalias() += weight * J.transpose() * J;
    mvmx[pose] += weight * g;
  }

  return 0;
}

int Initializer::updatePOFFactorGaussNewton() const {
  const auto &factors = mvFactors[FactorIndex::POF];
  const auto &evals = mvFactorEvals[0][FactorIndex::POF];
  const auto &lins = mvFactorLins[FactorIndex::POF];

  const auto &weight = mOptions.weights[FactorIndex::POF];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor = std::dynamic_pointer_cast<const POFFactor>(factors[i]);

    const auto &lin =
        std::dynamic_pointer_cast<const POFFactor::Linearization>(lins[i]);

    const auto &JR = lin->jacobian;

    const int &pose = factor->getPose();

    mvMxx[pose].topLeftCorner<3, 3>().noalias() += weight * JR.transpose() * JR;
    mvmx[pose].head<3>().noalias() +=
        weight * lin->scaledError * JR.transpose();
  }

  return 0;
}

int Initializer::updateDepthCameraFactorGaussNewton() const {
  const auto &factors = mvFactors[FactorIndex::DepthCamera];
  const auto &evals = mvFactorEvals[0][FactorIndex::DepthCamera];
  const auto &lins = mvFactorLins[FactorIndex::DepthCamera];

  const auto &weight = mOptions.weights[FactorIndex::DepthCamera];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor =
        std::dynamic_pointer_cast<const DepthCameraFactor>(factors[i]);

    const auto &lin =
        std::dynamic_pointer_cast<const DepthCameraFactor::Linearization>(
            lins[i]);

    const auto &J = lin->jacobian;
    const auto &g = lin->g;

    const int &pose = factor->getPose();

    mvMxx[pose].noalias() += weight * J.transpose() * J;
    mvmx[pose].noalias() += weight * lin->g;
  }

  return 0;
}

int Initializer::updateJointLimitFactorGaussNewton() const {
  const auto &factors = mvFactors[FactorIndex::JointLimit];
  const auto &evals = mvFactorEvals[0][FactorIndex::JointLimit];
  const auto &lins = mvFactorLins[FactorIndex::JointLimit];

  const auto &weight = mOptions.weights[FactorIndex::JointLimit];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor =
        std::dynamic_pointer_cast<const JointLimitFactor>(factors[i]);

    const auto &lin =
        std::dynamic_pointer_cast<const JointLimitFactor::Linearization>(
            lins[i]);
    const auto &eval = evals[i];

    const auto &JJoint = lin->jacobian;
    const auto &Error = Eigen::Map<const Vector6>(eval->error.data());

    const int &joint = factor->getJoint();

    mvMuu[joint].noalias() += weight * JJoint.transpose() * JJoint;
    mvmu[joint].noalias() += weight * JJoint.transpose() * Error;
  }

  return 0;
}

int Initializer::updatePoseConstFactorGaussNewton() const {
  const auto &factors = mvFactors[FactorIndex::PoseConst];
  const auto &evals = mvFactorEvals[0][FactorIndex::PoseConst];
  const auto &lins = mvFactorLins[FactorIndex::PoseConst];

  const auto &weight = mOptions.weights[FactorIndex::PoseConst];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor =
        std::dynamic_pointer_cast<const PoseConstFactor>(factors[i]);

    const auto &lin =
        std::dynamic_pointer_cast<const PoseConstFactor::Linearization>(
            lins[i]);
    const auto &eval = evals[i];

    const auto &JPose = lin->jacobian;
    const auto &Error = Eigen::Map<const Vector6>(eval->error.data());

    const int &pose = factor->getPose();

    mvMxx[pose].noalias() += weight * JPose.transpose() * JPose;
    mvmx[pose].noalias() += weight * JPose.transpose() * Error;
  }

  return 0;
}

int Initializer::updateJointConstFactorGaussNewton() const {
  const auto &factors = mvFactors[FactorIndex::JointConst];
  const auto &evals = mvFactorEvals[0][FactorIndex::JointConst];
  const auto &lins = mvFactorLins[FactorIndex::JointConst];

  const auto &weight = mOptions.weights[FactorIndex::JointConst];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor =
        std::dynamic_pointer_cast<const JointConstFactor>(factors[i]);

    const auto &lin =
        std::dynamic_pointer_cast<const JointConstFactor::Linearization>(
            lins[i]);
    const auto &eval = evals[i];

    const auto &JJoint = lin->jacobian;
    const auto &Error = Eigen::Map<const Vector3>(eval->error.data());

    const int &joint = factor->getJoint();

    mvMuu[joint].noalias() += weight * JJoint.transpose() * JJoint;
    mvmu[joint].noalias() += weight * JJoint.transpose() * Error;
  }

  return 0;
}

int Initializer::updatePoseFactorGaussNewton() const {
  const auto &factors = mvFactors[FactorIndex::Pose];
  const auto &evals = mvFactorEvals[0][FactorIndex::Pose];
  const auto &lins = mvFactorLins[FactorIndex::Pose];

  const auto &weight = mOptions.weights[FactorIndex::Pose];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor =
        std::dynamic_pointer_cast<const PoseFactor>(factors[i]);

    const auto &lin =
        std::dynamic_pointer_cast<const PoseFactor::Linearization>(lins[i]);
    const auto &eval = evals[i];

    const auto &JPose = lin->jacobian;
    const auto &Error = Eigen::Map<const Vector6>(eval->error.data());

    const int &pose = factor->getPose();

    mvMxx[pose].noalias() += weight * JPose.transpose() * JPose;
    mvmx[pose].noalias() += weight * JPose.transpose() * Error;
  }

  return 0;
}

int Initializer::updateJointFactorGaussNewton() const {
  const auto &factors = mvFactors[FactorIndex::Joint];
  const auto &evals = mvFactorEvals[0][FactorIndex::Joint];
  const auto &lins = mvFactorLins[FactorIndex::Joint];

  const auto &weight = mOptions.weights[FactorIndex::Joint];

  for (int i = 0; i < factors.size(); i++) {
    const auto &factor =
        std::dynamic_pointer_cast<const JointFactor>(factors[i]);

    const auto &lin =
        std::dynamic_pointer_cast<const JointFactor::Linearization>(lins[i]);
    const auto &eval = evals[i];

    const auto &JJoint = lin->jacobian;
    const auto &Error = Eigen::Map<const Vector3>(eval->error.data());

    const int &joint = factor->getJoint();

    mvMuu[joint].noalias() += weight * JJoint.transpose() * JJoint;
    mvmu[joint].noalias() += weight * JJoint.transpose() * Error;
  }

  return 0;
}

int Initializer::addFactor(
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

    factor->evaluate(mvPoses[0], mvJoints[0], eval);

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
  assert(factor->getID() == mFactors.size() - 1);

  return 0;
}

int Initializer::checkFactor(std::shared_ptr<const Factor> factor) {
  if (factor == nullptr) {
    LOG(WARNING) << "The factor should not be null." << std::endl;

    exit(-1);
  }

  return 0;
}
}  // namespace Initializer
}  // namespace scope
