#pragma once

#include <memory>
#include <vector>

#include <scope/initializer/factor/Factors.h>

namespace scope {
namespace Initializer {
struct FactorIndex {
  static const int PinholeCamera = 0;
  static const int POF = 1;
  static const int DepthCamera = 2;
  static const int JointLimit = 3;
  static const int PoseConst = 4;
  static const int JointConst = 5;
  static const int Pose = 6;
  static const int Joint = 7;
};

static const int NumFactors = FactorIndex::Joint + 1;

enum class Status {
  Uninitialized,
  Initialized,
  Accepted,
  Failed,
  Aborted,
  Converged
};

struct Options {
  /** A small positive number used to stablize the linear system */
  Scalar delta = 1e-1;

  /** Stopping criterion based upon the relative decrease in function value */
  Scalar rel_func_decrease_tol = 5e-3;

  /** Stopping tolerance for the relative decrease in linearized approximation
   */
  Scalar rel_lin_func_decrease_tol = 5e-3;

  /** Maximum elapsed computation time (in seconds) */
  double max_total_computation_time = 30;

  /** Stopping criterion based upon the norm of an accepted update step */
  Scalar stepsize_tol = 1e-3;

  /** Maximum permitted number of iterations */
  int max_iterations = 1000;

  /** Weights for each factor */
  scope::Vector<NumFactors> weights =
      100 * Eigen::Matrix<Scalar, NumFactors, 1>::Ones();

  /** Maximum number of inner iterations */
  int max_inner_iters = 20;

  /** Parameter to increase the stepsize */
  Scalar stepsize_increase_ratio = 1.2;

  /** Parameter to reduce the stepsize */
  Scalar stepsize_decrease_ratio = 0.5;

  /** Parameters for LM method */
  Scalar lm_lambda_increase = 1.5;
  Scalar lm_lambda_decrease = 0.8;
  Scalar lm_lambda_max = 1e3;
};

struct Results {
  /** Status */
  Status status = Status::Initialized;

  /** The objective values */
  std::vector<Scalar> fobjs;

  /** The stepsizes */
  std::vector<Scalar> stepsizes;

  /** The expected relative cost reduction */
  std::vector<Scalar> expected_rel_cost_reduction;

  /** The total elapsed computation time */
  double total_computation_time;

  /** The elapsed optimization time for each iteration*/
  std::vector<double> elapsed_optimization_times;
};

class Initializer {
 public:
  // the number of poses and joints
  const int mNumPoses;
  const int mNumJoints;

  // factors
  std::vector<std::vector<std::shared_ptr<const Factor>>> mvFactors;
  std::vector<std::shared_ptr<const Factor>> mFactors;

  // optimizer options
  Options mOptions;
  mutable Results mResults;

  // status
  mutable Status mStatus;

  // stepsize
  mutable Scalar mStepsize;

  mutable AlignedVector<Pose> mvPoses[2];
  mutable AlignedVector<Matrix3> mvJoints[2];

  mutable AlignedVector<Matrix6> mvMxx;
  mutable AlignedVector<Matrix3> mvMuu;

  mutable AlignedVector<Vector6> mvmx;
  mutable AlignedVector<Vector3> mvmu;

  // the objective value
  mutable Scalar mfobj[2];

  // objective values for each factor
  mutable scope::Vector<NumFactors> mvFobj[2];

  mutable std::vector<std::vector<std::shared_ptr<const Factor::Evaluation>>>
      mvFactorEvals[2];
  mutable std::vector<std::shared_ptr<Factor::Evaluation>> mFactorEvals[2];

  // linearizations
  std::vector<std::vector<std::shared_ptr<const Factor::Linearization>>>
      mvFactorLins;
  mutable std::vector<std::shared_ptr<Factor::Linearization>> mFactorLins;

  mutable Eigen::LDLT<MatrixX> mHchol;

  mutable MatrixX mH;
  mutable VectorX mh;
  mutable VectorX mhGN;

  mutable Scalar mE;
  mutable VectorX mSquaredError;

  mutable Scalar mDLambda;
  mutable VectorX mLambda;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Initializer(int NumJoints, const Options &options);

  Initializer(const Initializer &) = delete;

  Initializer &operator=(const Initializer &) = delete;

  virtual int initialize(const Pose &T0,
                         const AlignedVector<Matrix3> &joints) const;
  int solve() const;

  int updateFactorWeights(const Eigen::Matrix<Scalar, NumFactors, 1> &weights);

  int reset();

  int addPinholeCameraFactor(std::shared_ptr<PinholeCameraFactor> factor);
  int addJointLimitFactor(std::shared_ptr<JointLimitFactor> factor);
  int addPOFFactor(std::shared_ptr<POFFactor> factor);
  int addDepthCameraFactor(std::shared_ptr<DepthCameraFactor> factor);
  int addPoseConstFactor(std::shared_ptr<PoseConstFactor> factor);
  int addJointConstFactor(std::shared_ptr<JointConstFactor> factor);
  int addPoseFactor(std::shared_ptr<PoseFactor> factor);
  int addJointFactor(std::shared_ptr<JointFactor> factor);

  const Results &getResults() const { return mResults; }

  const AlignedVector<Pose> &getPoses() const { return mvPoses[0]; }

  const AlignedVector<Matrix3> &getJoints() const { return mvJoints[0]; }

  Scalar getFobj() const { return mfobj[0]; }

  const std::vector<std::vector<std::shared_ptr<const Factor::Evaluation>>>
      &getEvaluations() const {
    return mvFactorEvals[0];
  }

 public:
  int checkFactor(std::shared_ptr<const Factor> factor);

  int updatePinholeCameraFactorGaussNewton() const;
  int updateJointLimitFactorGaussNewton() const;
  int updatePOFFactorGaussNewton() const;
  int updateDepthCameraFactorGaussNewton() const;
  int updatePoseConstFactorGaussNewton() const;
  int updateJointConstFactorGaussNewton() const;
  int updatePoseFactorGaussNewton() const;
  int updateJointFactorGaussNewton() const;

  int optimize() const;
  virtual int accept() const;
  virtual int solveGaussNewton() const;

  int evaluate(int n) const;
  int linearize() const;

  int addFactor(int index, std::shared_ptr<Factor> factor,
                std::array<std::shared_ptr<Factor::Evaluation>, 2> &evals,
                std::shared_ptr<Factor::Linearization> &lin);

  virtual int FKintree(int n) const = 0;
  virtual int DFKintree() const = 0;

  virtual int updateGaussNewton() const = 0;

  virtual int update(Scalar stepsize) const = 0;
};
}  // namespace Initializer
}  // namespace scope
