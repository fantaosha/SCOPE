#pragma once

#include <cnpy/cnpy.h>
#include <glog/logging.h>
#include <scope/factor/Factors.h>
#include <scope/initializer/factor/Factors.h>
#include <scope/utils/InitialInfo.h>
#include <scope/utils/KeyPointInfo.h>

namespace scope {
template <typename M1, typename M2>
int projectToSO3(const Eigen::MatrixBase<M1> &M, Eigen::MatrixBase<M2> &R) {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(M1, 3, 3)
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(M2, 3, 3)

  Eigen::JacobiSVD<Matrix3> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Scalar detU = svd.matrixU().determinant();
  Scalar detV = svd.matrixV().determinant();

  if (detU * detV > 0) {
    R.noalias() = svd.matrixU() * svd.matrixV().transpose();
  } else {
    Matrix3 Uprime = svd.matrixU();
    Uprime.col(Uprime.cols() - 1) *= -1;
    R.noalias() = Uprime * svd.matrixV().transpose();
  }

  return 0;
}

template <int K, int P, int N>
int loadModel(const std::string &file, MatrixX &JDirs, VectorX &J,
              MatrixX &RelJDirs, VectorX &RelJ, MatrixX &VDirs, VectorX &V) {
  cnpy::npz_t model_data = cnpy::npz_load(file);

  cnpy::NpyArray subdata;

  // load JDirs
  subdata = model_data.at("JDirs");
  assert(subdata.shape[0] == P - 1 && subdata.shape[1] == 3 * K + 3);

  if (subdata.shape[0] != P - 1 && subdata.shape[1] != 3 * K + 3) {
    LOG(ERROR) << "Bad input file format." << std::endl;

    exit(-1);
  }

  JDirs.setZero(3 * K + 3, P);
  JDirs.leftCols<P - 1>() =
      Eigen::Map<Eigen::MatrixXd>(subdata.data<double>(), subdata.shape[1],
                                  subdata.shape[0])
          .cast<Scalar>();

  // load J
  subdata = model_data.at("J");
  assert(subdata.shape[0] == 1 && subdata.shape[1] == 3 * K + 3);

  if (subdata.shape[0] != 1 && subdata.shape[1] != 3 * K + 3) {
    LOG(ERROR) << "Bad input file format." << std::endl;

    exit(-1);
  }

  J = Eigen::Map<Eigen::VectorXd>(subdata.data<double>(), subdata.shape[1],
                                  subdata.shape[0]);

  // load RelJDirs
  subdata = model_data.at("RelJDirs");
  assert(subdata.shape[0] == P - 1 && subdata.shape[1] == 3 * K + 3);

  if (subdata.shape[0] != P - 1 && subdata.shape[1] != 3 * K + 3) {
    LOG(ERROR) << "Bad input file format." << std::endl;

    exit(-1);
  }

  RelJDirs.setZero(3 * K, P);
  RelJDirs.leftCols<P - 1>() =
      Eigen::Map<Eigen::MatrixXd>(subdata.data<double>(), subdata.shape[1],
                                  subdata.shape[0])
          .bottomRows<3 * K>();

  // load RelJ
  subdata = model_data.at("RelJ");
  assert(subdata.shape[0] == 1 && subdata.shape[1] == 3 * K + 3);

  if (subdata.shape[0] != 1 && subdata.shape[1] != 3 * K + 3) {
    LOG(ERROR) << "Bad input file format." << std::endl;

    exit(-1);
  }

  RelJ = Eigen::Map<Eigen::VectorXd>(subdata.data<double>(), subdata.shape[1],
                                     subdata.shape[0])
             .tail<3 * K>();

  // load vDirs
  subdata = model_data.at("vDirs");
  assert(subdata.shape[0] == P - 1 && subdata.shape[1] == 3 * N);

  if (subdata.shape[0] != P && subdata.shape[1] != 3 * N) {
    LOG(ERROR) << "Bad input file format." << std::endl;

    exit(-1);
  }

  VDirs.setZero(3 * N, P);
  VDirs.leftCols<P - 1>() = Eigen::Map<Eigen::MatrixXd>(
      subdata.data<double>(), subdata.shape[1], subdata.shape[0]);

  // load v
  subdata = model_data.at("v");
  assert(subdata.shape[0] == 3 * N);

  if (subdata.shape[0] != 3 * N) {
    LOG(ERROR) << "Bad input file format." << std::endl;

    exit(-1);
  }

  V = Eigen::Map<Eigen::VectorXd>(subdata.data<double>(), subdata.shape[0], 1);

  return 0;
}

int loadSMPLKeyPoints(const MatrixX &JDirs, const VectorX &J,
                      const MatrixX &VDirs, const VectorX &V,
                      MatrixX &KeyPointDirs, VectorX &KeyPoints);

int load3DPWAnnotation(const std::string &file, int &gender, VectorX &betas,
                       VectorX &poses, VectorX &cam);

int loadKeyPoint2DMeasurements(const std::string &file, Matrix3X &Measurements);

int loadKeyPoint3DMeasurements(const std::string &file, Matrix3X &Measurements);

int loadKeyPoint2Dand3DMeasurements(const std::string &file, Vector4 & CamParam,
                                    Matrix3X &Measurements2D,
                                    Matrix3X &Measurements3D);

int loadPOFMeasurements(const Matrix3X &KeyPoints3D, Matrix3X &Measurements);

int loadJointLimitNeuralNetwork(const std::string &file,
                                AlignedVector<Matrix6> &weight,
                                AlignedVector<Vector6> &bias, VectorX &PRelua,
                                VectorX &PRelub);

int loadJointLimitPrior(const std::string &file,
                        AlignedVector<AlignedVector<Matrix6>> &weight,
                        AlignedVector<AlignedVector<Vector6>> &bias,
                        AlignedVector<VectorX> &PRelua,
                        AlignedVector<VectorX> &PRelub);

int createInitialSMPLSkeleton(const Matrix3X &Measurements3D, VectorX &J);

int createInitialPinholeCameraFactors(
    const std::vector<std::array<int, 3>> &Info,
    const std::vector<std::array<int, 2>> &Index, const VectorX &KeyPoints,
    Scalar gm_est_sigma, Scalar gm_est_eps, const Matrix3X &Measurements,
    const VectorX &ConfidenceThreshold2D,
    std::vector<std::shared_ptr<Initializer::PinholeCameraFactor>> &factors);

int createInitialPOFFactors(
    const AlignedVector<std::tuple<int, int, scope::Vector3, Scalar>> &Info,
    Scalar gm_est_sigma, Scalar gm_est_eps, const Matrix3X &Measurements,
    std::vector<std::shared_ptr<Initializer::POFFactor>> &factors);

int createInitialDepthCameraFactors(
    const std::vector<std::tuple<int, int, int, Scalar>> &Info,
    const std::vector<std::array<int, 2>> &Index, const VectorX &KeyPoints,
    Scalar gm_est_sigma, Scalar gm_est_eps, const Matrix3X &Measurements,
    const Vector3 &root,
    std::vector<std::shared_ptr<Initializer::DepthCameraFactor>> &factors);

int createInitialJointLimitFactors(
    const std::vector<std::array<int, 2>> &Index,
    const AlignedVector<AlignedVector<scope::Matrix6>> &weight,
    const AlignedVector<AlignedVector<Vector6>> &bias,
    const AlignedVector<VectorX> &PRelua, const AlignedVector<VectorX> &PRelub,
    const VectorX &scale,
    std::vector<std::shared_ptr<Initializer::JointLimitFactor>> &factors);

int createInitialPoseConstFactors(
    const AlignedVector<std::tuple<int, Vector6, Vector6, Vector6>>
        &PoseConstInfo,
    const AlignedVector<Pose> &PoseRef,
    std::vector<std::shared_ptr<Initializer::PoseConstFactor>> &factors);

int createInitialJointConstFactors(
    const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
        &JointConstInfo,
    std::vector<std::shared_ptr<Initializer::JointConstFactor>> &factors);

int createInitialEulerAngleConstFactors(
    const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
        &JointConstInfo,
    std::vector<std::shared_ptr<Initializer::EulerAngleConstFactor>> &factors);

int createInitialRootPinholeCameraFactors(
    const std::vector<std::array<int, 2>> &JointPinholeInfo,
    const std::vector<std::array<int, 3>> &VertexPinholeInfo,
    const AlignedVector<Pose> &Poses, const VectorX &Param,
    const MatrixX &JDirs, const VectorX &J, const MatrixX &KeyPointDirs,
    const VectorX &KeyPoints, Scalar gm_est_sigma, Scalar gm_est_eps,
    const Matrix3X &Measurements, const VectorX &ConfidenceThreshold2D,
    std::vector<std::shared_ptr<Initializer::PinholeCameraFactor>> &factors);

int createInitialRootPOFFactors(
    const AlignedVector<std::tuple<int, int, scope::Vector3, Scalar>>
        &UnitPOFInfo,
    const std::vector<std::tuple<int, int, int, int, Scalar>> &ScaledPOFInfo,
    const std::vector<std::tuple<int, int, int, int, int, Scalar>> &RelPOFInfo,
    const AlignedVector<Pose> &Poses, const VectorX &Param,
    const MatrixX &JDirs, const VectorX &J, const MatrixX &KeyPointDirs,
    const VectorX &KeyPoints, const std::vector<int> &kintree,
    Scalar gm_est_sigma, Scalar gm_est_eps, const Matrix3X &Measurements,
    std::vector<std::shared_ptr<Initializer::POFFactor>> &factors);

int createJointPinholeCameraFactors(
    const std::vector<std::array<int, 2>> &Info, Scalar gm_est_sigma,
    Scalar gm_est_eps, const Matrix3X &Measurements,
    const VectorX &ConfidenceThreshold2D,
    std::vector<std::shared_ptr<JointPinholeCameraFactor>> &factors);

int createVertexPinholeCameraFactors(
    const std::vector<std::array<int, 3>> &Info, int VertexParam,
    const MatrixX &JDirs, const VectorX &J, const MatrixX &KeyPointDirs,
    const VectorX &KeyPoints, Scalar gm_est_sigma, Scalar gm_est_eps,
    const Matrix3X &Measurements, const VectorX &ConfidenceThreshold2D,
    std::vector<std::shared_ptr<VertexPinholeCameraFactor>> &factors);

int createUnitPOFFactors(
    const AlignedVector<std::tuple<int, int, scope::Vector3, Scalar>> &Info,
    Scalar gm_est_sigma, Scalar gm_est_eps, const Matrix3X &Measurements,
    std::vector<std::shared_ptr<UnitPOFFactor>> &factors);

int createScaledPOFFactors(
    const std::vector<std::tuple<int, int, int, int, Scalar>> &Info,
    int VertexParam, const MatrixX &KeyPointDirs, const VectorX &KeyPoints,
    Scalar gm_est_sigma, Scalar gm_est_eps, const Matrix3X &Measurements,
    std::vector<std::shared_ptr<ScaledPOFFactor>> &factors);

int createRelPOFFactors(
    const std::vector<std::tuple<int, int, int, int, int, Scalar>> &Info,
    int VertexParam, const MatrixX &JDirs, const VectorX &J,
    const MatrixX &KeyPointDirs, const VectorX &KeyPoints,
    const std::vector<int> &kintree, Scalar gm_est_sigma, Scalar gm_est_eps,
    const Matrix3X &Measurements,
    std::vector<std::shared_ptr<RelPOFFactor>> &factors);

int createJointDepthCameraFactors(
    const std::vector<std::tuple<int, int, Scalar>> &Info, Scalar gm_est_sigma,
    Scalar gm_est_eps, const Matrix3X &Measurements, const Vector3 &root,
    std::vector<std::shared_ptr<JointDepthCameraFactor>> &factors);

int createVertexDepthCameraFactors(
    const std::vector<std::tuple<int, int, int, Scalar>> &Info, int VertexParam,
    const MatrixX &JDirs, const VectorX &J, const MatrixX &KeyPointDirs,
    const VectorX &KeyPoints, Scalar gm_est_sigma, Scalar gm_est_eps,
    const Matrix3X &Measurements, const Vector3 &root,
    std::vector<std::shared_ptr<VertexDepthCameraFactor>> &factors);

int createJointConstFactors(
    const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
        &JointConstInfo,
    std::vector<std::shared_ptr<JointConstFactor>> &factors);

int createEulerAngleConstFactors(
    const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
        &EulerAngleConstInfo,
    std::vector<std::shared_ptr<EulerAngleConstFactor>> &factors);

int createJointLimitFactors(
    const AlignedVector<AlignedVector<scope::Matrix6>> &weight,
    const AlignedVector<AlignedVector<Vector6>> &bias,
    const AlignedVector<VectorX> &PRelua, const AlignedVector<VectorX> &PRelub,
    const VectorX &scale,
    std::vector<std::shared_ptr<JointLimitFactor>> &factors);

int solveReducedEPnP(const std::vector<int> &KeyPoint3Dto2D,
                     const Matrix3X &Measurements3D,
                     const Matrix3X &Measurements2D,
                     const VectorX &ConfidenceThreshold2D, Vector3 &t);

int solvePose(
    const std::vector<std::tuple<int, int, int, Scalar>> &DepthCamInfo,
    const Vector3 &Root, const VectorX &KeyPoints,
    const Matrix3X &Measurements3D, Pose &pose);

int solveShape(
    const std::vector<std::tuple<std::array<int, 4>, Vector3>> &RelShapeInfo,
    const std::vector<std::tuple<std::array<int, 4>, Vector3>>
        &RelShapePairInfo,
    const MatrixX &ShapeSqrtCov, Scalar weight, const MatrixX &KeyPointDirs,
    const VectorX &KeyPoints, const Matrix3X &Measurements3D, VectorX &betas,
    Scalar &s);

int getKeyPointOrigin3D(const Pose &Root, const MatrixX &JDirs,
                        const VectorX &J, const MatrixX &KeyPointDirs,
                        const VectorX &KeyPoints, const VectorX &Param,
                        int OriginIndex, Vector3 &keypoint);
}  // namespace scope
