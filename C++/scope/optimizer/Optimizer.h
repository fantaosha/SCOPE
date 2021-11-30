#pragma once

#include <memory>
#include <vector>

#include <scope/factor/Factors.h>
#include <scope/model/Model.h>

// TODO: how to deal with collisions

namespace scope {
namespace Optimizer {
struct FactorIndex {
  static const int JointPinholeCamera = 0;
  static const int FullJointPinholeCamera = 1;
  static const int VertexPinholeCamera = 2;
  static const int FullVertexPinholeCamera = 3;
  static const int UnitPOF = 4;
  static const int ScaledPOF = 5;
  static const int RelPOF = 6;
  static const int JointDepthCamera = 7;
  static const int VertexDepthCamera = 8;
  static const int JointLimit = 9;
  static const int Collision = 10;
  static const int JointConst = 11;
  static const int PoseConst = 12;
  static const int Pose = 13;
  static const int Shape = 14;
  static const int Joint = 15;
  static const int Parameter = 16;
};

static const int NumFactors = FactorIndex::Parameter + 1;

enum class Status {
  Converged,
  Failed,
  Accepted,
  Aborted,
  Initialized,
  Uninitialized
};

enum class Method { GaussNewton, LM };

struct Options {
  /** Method */
  Method method = Method::LM;

  /** A small positive number used to stablize the linear system */
  Scalar delta = 1e-3;

  /** Stopping criterion based upon the relative decrease in function value */
  Scalar rel_func_decrease_tol = 1e-4;

  /** Stopping tolerance for the relative decrease in linearized approximation
   */
  Scalar rel_lin_func_decrease_tol = 1e-3;

  /** Maximum elapsed computation time (in seconds) */
  double max_computation_time = 30;

  /** Stopping criterion based upon the norm of an accepted update step */
  Scalar stepsize_tol = 1e-3;

  /** Maximum permitted number of iterations */
  int max_iterations = 1000;

  /** Check collisions or not */
  bool check_collisions = false;

  /** Weights for each factor */
  scope::Vector<NumFactors> weights =
      100 * Eigen::Matrix<Scalar, NumFactors, 1>::Ones();

  /** Threshold for collision checking */
  Scalar collision_threshold = 0;

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

  /** Extra parameter for hip */
  bool extra_hip_parameter = true;
  double extra_hip_parameter_bnd[2] = {-0.1, 0.1};
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

  /** The actual relative cost reduction */
  std::vector<Scalar> rel_cost_reduction;

  /** The total elapsed computation time */
  double total_computation_time;

  /** The elapsed optimization time for each iteration*/
  std::vector<double> elapsed_optimization_times;
};

template <int K, int P, int N, bool CamOpt>
class Optimizer {
 public:
 protected:
  // optimizer options
  Options mOptions;
  mutable Results mResults;

  // human model used
  std::shared_ptr<const Model<K, P, N>> mModel;

  // the number of poses
  static const int NumPoses = K + 1;
  // the number of shapes
  static const int NumShapes = 0;
  // the number of joints
  static const int NumJoints = K;
  // the number of implicit parameters
  static const int NumParams = 1 + CamOpt;

  /** Maximum number of collisions considered when computing a descent
   * direction */
  static const int MaxNumCollisions = 0;

  static const int CamParamIndex = CamOpt ? 0 : -1;
  static const int VertexParamIndex = CamOpt ? 1 : 0;
  static const int FaceParamIndex = -2;

  static const int CamParamOffset = 0;
  static const int VertexParamOffset = CamOpt ? 4 : 0;
  static const int FaceParamOffset = VertexParamOffset + P;

  static const int CamParamSize = CamOpt ? 4 : 0;
  static const int VertexParamSize = P;
  static const int FaceParamSize = 0;

  static const std::vector<int> ParamSizes;

  // factors
  std::vector<std::vector<std::vector<std::shared_ptr<const Factor>>>>
      mnvFactors;
  std::vector<std::vector<std::shared_ptr<const Factor>>> mnFactors;

  std::vector<std::vector<std::shared_ptr<const Factor>>> mvFactors;
  std::vector<std::shared_ptr<const Factor>> mFactors;

  // the size of parameters
  static const int ParamSize = CamOpt ? P + 4 : P;
  // the dimension of optimization variables
  static const int Size = 6 + 3 + ParamSize;
  // maximum DSize
  static const int MaxDSize = Size;

  // dimension of 3D Pose
  static const int DPoseSize = 6;
  // dimension of each joint state
  static const int DJointSize = 3;
  // the number of implicit parameters used in descent direction computation,
  // which depends on the number of collisions
  static const int DParamSize = ParamSize;
  // the size of the Hessian matrix
  static const int DSize = Size;

  static const int DPoseOffset = 0;
  static const int DParamOffset = 6;
  static const int DJointOffset = DParamOffset + DParamSize;

  static const int DCamParamOffset = DParamOffset + CamParamOffset;
  static const int DVertexParamOffset = DParamOffset + VertexParamOffset;
  static const int DFaceParamOffset = DParamOffset + FaceParamOffset;

  static const std::vector<int> DParamOffsets;

  // status
  mutable Status mStatus;

  // stepsize
  mutable Scalar mStepsize;

  // graident norm
  mutable Scalar mGradientNorm;

  mutable int mCollisionParamSize;
  mutable int mDCollisionParamOffset;

  // the objective value
  mutable Scalar mfobj[2];
  // poses of body parts
  mutable AlignedVector<Pose> mvPoses[2];
  // shape parameters
  mutable AlignedVector<VectorX> mvShapes[2];
  // joint states
  mutable AlignedVector<Matrix3> mvJoints[2];
  // model parameters and camera intrinsics
  mutable AlignedVector<VectorX> mvParams[2];
  // relative joint locations
  VectorX mRawRelJointLocations[2];
  mutable Eigen::Map<Vector<3 * K>> mRelJointLocations[2];

  // objective values for each factor
  mutable scope::Vector<NumFactors> mvFobj[2];

  // number of collisions detected
  mutable int mNumCollisions;

  // evaluations
  mutable std::vector<
      std::vector<std::vector<std::shared_ptr<const Factor::Evaluation>>>>
      mnvFactorEvals[2];
  mutable std::vector<std::vector<std::shared_ptr<Factor::Evaluation>>>
      mnFactorEvals[2];

  mutable std::vector<std::vector<std::shared_ptr<const Factor::Evaluation>>>
      mvFactorEvals[2];
  mutable std::vector<std::shared_ptr<Factor::Evaluation>> mFactorEvals[2];

  // intermediates used for update
  mutable Pose mRootPoseChange;
  mutable Vector6 mDRootPoseChange;
  mutable AlignedVector<Matrix3> mvJointChange;
  mutable AlignedVector<Vector3> mvDJointChange;

  // linearizations
  std::vector<
      std::vector<std::vector<std::shared_ptr<const Factor::Linearization>>>>
      mnvFactorLins;
  mutable std::vector<std::vector<std::shared_ptr<Factor::Linearization>>>
      mnFactorLins;

  std::vector<std::vector<std::shared_ptr<const Factor::Linearization>>>
      mvFactorLins;
  mutable std::vector<std::shared_ptr<Factor::Linearization>> mFactorLins;

  Matrix6X mRawB;
  Eigen::Map<Matrix<6, K *(P + 3)>> mB;
  mutable AlignedVector<Eigen::Map<Matrix<6, P + 3>>> mvB;
  AlignedVector<Eigen::Map<Matrix<3, P>, 0, Eigen::Stride<6, 1>>> mvBp;
  AlignedVector<Eigen::Map<Matrix63>> mvBu;

  // [Mxx  *   *
  //  Mpx Mpp  *
  //  Mux Mup Muu]
  Matrix<DSize, Eigen::Dynamic> mRawM;
  mutable Eigen::Map<Matrix<DSize, NumPoses * DSize>> mM;
  mutable AlignedVector<Eigen::Map<scope::Matrix<DSize, DSize>>> mvM;

  mutable AlignedVector<Vector6> mvmx;
  mutable AlignedVector<Vector<DParamSize>> mvmp;
  mutable AlignedVector<Vector3> mvmu;

  // Gradient
  mutable AlignedVector<Vector6> mvPoseDG;
  mutable AlignedVector<Vector3> mvJointDG;
  mutable AlignedVector<Vector<P>> mvParamDG;
  mutable VectorX mParamDG;

  // Gauss-Newton direction
  Matrix6X mRawPoseGN;
  Matrix3X mRawJointGN;
  Eigen::Map<Matrix<6, NumPoses>> mPoseGN;
  Eigen::Map<Matrix<3, NumJoints>> mJointGN;
  mutable AlignedVector<Eigen::Map<Vector6>> mvPoseGN;
  mutable AlignedVector<Eigen::Map<Vector3>> mvJointGN;

  VectorX mRawParamGN;
  mutable Eigen::Map<Vector<DParamSize>> mParamGN;
  VectorX mRawhGN;
  mutable Eigen::Map<Vector<6 + DParamSize>> mhGN;

  mutable VectorX mS;

  // [Hxx  *   *
  //  Hpx Hpp  *
  //  Hux Hup Huu]
  Matrix<DSize, Eigen::Dynamic> mRawH;
  mutable Eigen::Map<Matrix<DSize, NumJoints * DSize>> mH;
  mutable AlignedVector<Eigen::Map<Matrix<DSize, DSize>>> mvH;

  Matrix<DSize, Eigen::Dynamic> mRawHxB;
  Eigen::Map<Matrix<DSize, NumJoints *(P + 3)>> mHxB;
  mutable AlignedVector<Eigen::Map<Matrix<DSize, P + 3>>> mvHxB;

  RowMajorMatrix<3 * NumJoints, Eigen::Dynamic> mRawKuxp;
  Eigen::Map<RowMajorMatrix<3 * NumJoints, 6 + DParamSize>> mKuxp;
  mutable AlignedVector<Eigen::Map<RowMajorMatrix<3, 6 + DParamSize>>> mvKuxp;

  Matrix<6 + DParamSize, Eigen::Dynamic> mRawH0;
  mutable Eigen::Map<Matrix<6 + DParamSize, 6 + DParamSize>> mH0;

  mutable AlignedVector<Matrix3> mvHuuInv;
  mutable Eigen::LDLT<MatrixX> mHchol0;

  mutable AlignedVector<Vector6> mvhx;
  mutable AlignedVector<Vector<P>> mvhp;
  mutable AlignedVector<Vector3> mvhu;

  VectorX mRawku;
  mutable Eigen::Map<Vector<3 * K>> mku;
  mutable AlignedVector<Eigen::Map<Vector3>> mvku;

  mutable scope::Vector<K> mvE;

  VectorX mRawh0;
  mutable Eigen::Map<Vector<6 + DParamSize>> mh0;

  // expected cost reduction
  mutable Scalar mExtraCost;
  mutable Scalar mE;

  // cost reduction
  mutable Vector2 mDCost;

  mutable AlignedVector<Vector6> mvDCostDx;
  mutable AlignedVector<Vector<ParamSize>> mvDCostDp;
  mutable AlignedVector<Vector3> mvDCostDu;

  // LM parameters
  mutable Scalar mDLambda;

  scope::VectorX mRawLambda;
  mutable Eigen::Map<Vector<6 + DParamSize + 3 * K>> mLambda;

  scope::VectorX mRawSquaredError;
  mutable Eigen::Map<Vector<6 + DParamSize + 3 * K>> mSquaredError;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Optimizer(const std::shared_ptr<const Model<K, P, N>> &model,
            const Options &options);

  Optimizer(const Optimizer &) = delete;

  Optimizer &operator=(const Optimizer &) = delete;

  int initialize(bool refine = true) const;

  int initialize(const Pose &T0, const AlignedVector<Matrix3> &joints,
                 const AlignedVector<VectorX> &params,
                 bool refine = true) const;

  int optimize() const;

  int evaluate() const;

  int updateFactorWeights(const Eigen::Matrix<Scalar, NumFactors, 1> &weights);

  int reset();

  int addJointPinholeCameraFactor(
      std::shared_ptr<JointPinholeCameraFactor> factor);
  int addVertexPinholeCameraFactor(
      std::shared_ptr<VertexPinholeCameraFactor> factor);
  int addPOFFactor(std::shared_ptr<POFFactor> factor);
  int addDepthCameraFactor(std::shared_ptr<DepthCameraFactor> factor);
  int addJointLimitFactor(std::shared_ptr<JointLimitFactor> factor);
  int addPoseConstFactor(std::shared_ptr<PoseConstFactor> factor);
  int addJointConstFactor(std::shared_ptr<JointConstFactor> factor);
  int addPoseFactor(std::shared_ptr<PoseFactor> factor);
  int addJointFactor(std::shared_ptr<JointFactor> factor);
  int addParameterFactor(std::shared_ptr<ParameterFactor> factor);

  int getNumPoses() const;
  int getNumShapes() const;
  int getNumJoints() const;
  int getNumParameters() const;

  int getFaceParameterIndex() const;
  int getVertexParameterIndex() const;
  int getCameraParameterIndex() const;

  const Results &getResults() const;

  Scalar getFobj() const;

  AlignedVector<Pose> const &getPoses() const;
  AlignedVector<VectorX> const &getShapes() const;
  AlignedVector<Matrix3> const &getJoints() const;
  AlignedVector<VectorX> const &getParameters() const;

  std::vector<std::vector<std::shared_ptr<const Factor::Evaluation>>> const &
  getEvaluations() const;

 protected:
  // Gauss-Newton directions;
  AlignedVector<Vector6> const &getPoseDG() { return mvPoseDG; }

  AlignedVector<Vector3> const &getJointDG() { return mvJointDG; }

  VectorX const &getParamDG() { return mParamDG; }

  // setup
  virtual int setupModelInfo();
  virtual int setupParameterInfo() = 0;
  virtual int setupFactors();
  virtual int setupOptimization();

  // initialization
  virtual int init() const;
  virtual int initialOptimize() const;

  // evaluation
  virtual int update(Scalar stepsize) const;
  virtual int evaluateFactors(int n) const;
  virtual int evaluateInitialFactors(int n) const;

  virtual int updateGaussNewtonInfo() const;
  virtual int accept() const;

  // linearization
  virtual int linearize() const;
  virtual int linearizeFactors(int pose) const;

  // update hessians and gradients
  virtual int updateJointPinholeCameraFactorGaussNewton() const;
  virtual int updateFullJointPinholeCameraFactorGaussNewton() const;
  virtual int updateVertexPinholeCameraFactorGaussNewton() const;
  virtual int updateFullVertexPinholeCameraFactorGaussNewton() const;
  virtual int updateUnitPOFFactorGaussNewton() const;
  virtual int updateScaledPOFFactorGaussNewton() const;
  virtual int updateRelPOFFactorGaussNewton() const;
  virtual int updateJointDepthCameraFactorGaussNewton() const;
  virtual int updateVertexDepthCameraFactorGaussNewton() const;
  virtual int updateJointLimitFactorGaussNewton() const;
  virtual int updateCollisionFactorGaussNewton() const;
  virtual int updatePoseConstFactorGaussNewton() const;
  virtual int updateJointConstFactorGaussNewton() const;
  virtual int updatePoseFactorGaussNewton() const;
  virtual int updateShapeFactorGaussNewton() const;
  virtual int updateJointFactorGaussNewton() const;
  virtual int updateParamFactorGaussNewton() const;

  virtual int updateJointPinholeCameraFactorGaussNewton(int pose) const;
  virtual int updateFullJointPinholeCameraFactorGaussNewton(int pose) const;
  virtual int updateVertexPinholeCameraFactorGaussNewton(int pose) const;
  virtual int updateFullVertexPinholeCameraFactorGaussNewton(int pose) const;
  virtual int updateUnitPOFFactorGaussNewton(int pose) const;
  virtual int updateScaledPOFFactorGaussNewton(int pose) const;
  virtual int updateRelPOFFactorGaussNewton(int pose) const;
  virtual int updateJointDepthCameraFactorGaussNewton(int pose) const;
  virtual int updateVertexDepthCameraFactorGaussNewton(int pose) const;
  virtual int updateJointLimitFactorGaussNewton(int pose) const;
  virtual int updateCollisionFactorGaussNewton(int pose) const;
  virtual int updatePoseConstFactorGaussNewton(int pose) const;
  virtual int updateJointConstFactorGaussNewton(int pose) const;
  virtual int updatePoseFactorGaussNewton(int pose) const;
  virtual int updateShapeFactorGaussNewton(int pose) const;
  virtual int updateJointFactorGaussNewton(int pose) const;
  virtual int updateParamFactorGaussNewton(int pose) const;

  // compute gradient
  virtual int computeGradient() const;

  // solve Gauss Newton direction
  virtual int solveGaussNewton() const;

  /** compute DCost */
  virtual int updateDCost() const;

  /** Gauss-Newton direction */
  // factorize non-root link
  virtual int factorize(int i) const;

  // backward pass for non-root link
  virtual int backward(int i) const;

  // forward pass for non-root link
  virtual int forward(int i) const;

  // factorize root link
  virtual int factorizeN() const;

  // backward pass for root link
  virtual int backwardN() const;

  // backward pass through a subtree
  virtual int backwardPass(const std::vector<int> &subtree) const;

  // forkward pass through a subtree
  virtual int forwardPass(const std::vector<int> &subtree) const;

  // update cost reduction
  virtual int updateCostReduction() const;

  // backward paoss for gradient evaluation
  virtual int backwardDG(int i) const;

  // add a factor
  int addFactor(int index, std::shared_ptr<Factor> factor,
                std::array<std::shared_ptr<Factor::Evaluation>, 2> &evals,
                std::shared_ptr<Factor::Linearization> &lin);

  // check if a factor is valid
  int checkFactor(std::shared_ptr<const Factor> factor);
};
}  // namespace Optimizer
}  // namespace scope
