#include <scope/utils/Setup.h>

#include <fstream>
#include <nlohmann/json.hpp>

namespace scope {
int loadSMPLKeyPoints(const MatrixX &JDirs, const VectorX &J,
                      const MatrixX &VDirs, const VectorX &V,
                      MatrixX &KeyPointDirs, VectorX &KeyPoints) {
  const int NumPoses = 24;
  const int ParamSize = JDirs.cols();

  const auto &KeyPointInfo = KeyPointInfo::SMPL::KeyPointInfo;
  const auto &NumKeyPoints = KeyPointInfo.size();

  assert(JDirs.rows() == 3 * NumPoses);
  assert(J.rows() == 3 * NumPoses);
  assert(VDirs.rows() == 20670);
  assert(V.rows() == 20670);
  assert(JDirs.cols() == VDirs.cols());
  assert(JDirs.rightCols<1>().cwiseAbs().sum() < 1e-10);
  assert(VDirs.rightCols<1>().cwiseAbs().sum() < 1e-10);

  KeyPointDirs.setZero(3 * NumKeyPoints, ParamSize);
  KeyPoints.setZero(3 * NumKeyPoints);

  // setup vertex keypoints
  for (int i = 0; i < 11; i++) {
    assert(KeyPointInfo[i][1] < 6890);

    if (KeyPointInfo[i][1] >= 6890) {
      LOG(ERROR) << "Inconsistent keypoint information." << std::endl;
      exit(-1);
    }

    KeyPointDirs.middleRows<3>(3 * i) =
        VDirs.middleRows<3>(3 * KeyPointInfo[i][1]);
    KeyPoints.segment<3>(3 * i) = V.segment<3>(3 * KeyPointInfo[i][1]);
  }

  // setup head top
  const Scalar HeadTopRatio = 0.65;
  KeyPointDirs.middleRows<3>(3 * 11) =
      HeadTopRatio * VDirs.middleRows<3>(3 * 414) +
      (1 - HeadTopRatio) * JDirs.middleRows<3>(3 * 15);
  KeyPoints.segment<3>(3 * 11) = HeadTopRatio * V.segment<3>(3 * 414) +
                                 (1 - HeadTopRatio) * J.segment<3>(3 * 15);

  // setup chest keypoint
  const Scalar ChestRatio = 0.94;

  KeyPointDirs.middleRows<3>(3 * 12) = JDirs.middleRows<3>(3 * 6);
  KeyPointDirs.row(3 * 12 + 1) = ChestRatio * JDirs.row(3 * 6 + 1) +
                                 (1 - ChestRatio) * JDirs.row(3 * 3 + 1);

  KeyPoints.segment<3>(3 * 12) = J.segment<3>(3 * 6);
  KeyPoints[3 * 12 + 1] =
      ChestRatio * J[3 * 6 + 1] + (1 - ChestRatio) * J[3 * 3 + 1];

  // setup hip keypoints
  const Scalar HipRatio = 0.5;

  KeyPointDirs.middleRows<3>(3 * 13) = JDirs.middleRows<3>(0);
  KeyPointDirs.middleRows<3>(3 * 14) = JDirs.middleRows<3>(3);
  KeyPointDirs.middleRows<3>(3 * 15) = JDirs.middleRows<3>(6);

  Eigen::MatrixXd HDirs =
      JDirs.row(1) - 0.5 * JDirs.row(4) - 0.5 * JDirs.row(7);
  KeyPointDirs.row(3 * 13 + 1) -= (1 - HipRatio) * HDirs;
  KeyPointDirs.row(3 * 14 + 1) += HipRatio * HDirs;
  KeyPointDirs.row(3 * 15 + 1) += HipRatio * HDirs;

  KeyPointDirs.row(3 * 14) = JDirs.row(12);
  KeyPointDirs.row(3 * 15) = JDirs.row(15);

  KeyPointDirs.row(3 * 14 + 2) = JDirs.row(2);
  KeyPointDirs.row(3 * 15 + 2) = JDirs.row(2);

  KeyPoints.segment<3>(3 * 13) = J.segment<3>(0);
  KeyPoints.segment<3>(3 * 14) = J.segment<3>(3);
  KeyPoints.segment<3>(3 * 15) = J.segment<3>(6);

  Scalar H = J[1] - 0.5 * J[4] - 0.5 * J[7];
  KeyPoints[3 * 13 + 1] -= (1 - HipRatio) * H;
  KeyPoints[3 * 14 + 1] += HipRatio * H;
  KeyPoints[3 * 15 + 1] += HipRatio * H;

  KeyPoints[3 * 14] = J[12];
  KeyPoints[3 * 15] = J[15];

  KeyPoints[3 * 14 + 2] = J[2];
  KeyPoints[3 * 15 + 2] = J[2];

  KeyPointDirs(3 * 13 + 1, ParamSize - 1) = 10 * H;
  KeyPointDirs(3 * 14 + 1, ParamSize - 1) = 10 * H;
  KeyPointDirs(3 * 15 + 1, ParamSize - 1) = 10 * H;

  // setup upper arm keypoints
  const Scalar UpperArmRatio[] = {0.65, 0.5};
  KeyPointDirs.middleRows<3>(3 * 16) = JDirs.middleRows<3>(3 * 16);
  KeyPoints.segment<3>(3 * 16) = J.segment<3>(3 * 16);

  KeyPointDirs.middleRows<3>(3 * 17) = JDirs.middleRows<3>(3 * 17);
  KeyPoints.segment<3>(3 * 17) = J.segment<3>(3 * 17);

  KeyPointDirs.row(48) = (1 - UpperArmRatio[0]) * JDirs.row(48) +
                         UpperArmRatio[0] * KeyPointDirs.row(42);
  KeyPoints[48] =
      (1 - UpperArmRatio[0]) * J[48] + UpperArmRatio[0] * KeyPoints[42];

  KeyPointDirs.row(51) = (1 - UpperArmRatio[0]) * JDirs.row(51) +
                         UpperArmRatio[0] * KeyPointDirs.row(45);
  KeyPoints[51] =
      (1 - UpperArmRatio[0]) * J[51] + UpperArmRatio[0] * KeyPoints[45];

  KeyPointDirs.row(49) =
      (1 - UpperArmRatio[1]) * JDirs.row(49) + UpperArmRatio[1] * JDirs.row(37);
  KeyPoints[49] = (1 - UpperArmRatio[1]) * J[49] + UpperArmRatio[1] * J[37];

  KeyPointDirs.row(52) =
      (1 - UpperArmRatio[1]) * JDirs.row(52) + UpperArmRatio[1] * JDirs.row(37);
  KeyPoints[52] = (1 - UpperArmRatio[1]) * J[52] + UpperArmRatio[1] * J[37];

  // setup joint keypoints
  KeyPointDirs.bottomRows<3 * NumPoses>() = JDirs;
  KeyPoints.tail<3 * NumPoses>() = J;

  return 0;
}

int load3DPWAnnotation(const std::string &file, int &gender, VectorX &betas,
                       VectorX &poses, VectorX &cam) {
  cnpy::npz_t annot = cnpy::npz_load(file);

  cnpy::NpyArray subdata;

  subdata = annot.at("gender");
  assert(subdata.num_vals == 1);

  if (subdata.num_vals != 1) {
    LOG(ERROR) << "Bad input file format." << std::endl;

    exit(-1);
  }

  gender = *subdata.data<double>();

  subdata = annot.at("shape");
  assert(subdata.shape[0] == subdata.num_vals);

  if (subdata.shape[0] != subdata.num_vals) {
    LOG(ERROR) << "Bad input file format." << std::endl;

    exit(-1);
  }

  betas = Eigen::Map<VectorX>(subdata.data<double>(), subdata.num_vals);

  subdata = annot.at("pose");
  assert(subdata.shape[0] == subdata.num_vals);

  if (subdata.shape[0] != subdata.num_vals) {
    LOG(ERROR) << "Bad input file format." << std::endl;

    exit(-1);
  }

  poses = Eigen::Map<VectorX>(subdata.data<double>(), subdata.num_vals);

  subdata = annot.at("cam_params");
  assert(subdata.shape[0] == subdata.num_vals && subdata.num_vals == 4);

  if (subdata.num_vals != 4) {
    LOG(ERROR) << "Bad input file format." << std::endl;

    exit(-1);
  }

  cam = Eigen::Map<VectorX>(subdata.data<double>(), subdata.num_vals);

  return 0;
}

int loadKeyPoint2DMeasurements(const std::string &file,
                               Matrix3X &Measurements) {
  cnpy::NpyArray measurements = cnpy::npy_load(file);

  assert(measurements.shape[1] == 3);

  if (measurements.shape[1] != 3) {
    LOG(ERROR) << "Inconsistent measurement size." << std::endl;
    exit(-1);
  }

  Measurements = Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>>(
                     measurements.data<double>(), measurements.shape[1],
                     measurements.shape[0])
                     .cast<Scalar>();

  Measurements.row(2) = Measurements.row(2).cwiseMin(1.0);

  return 0;
}

int loadKeyPoint3DMeasurements(const std::string &file,
                               Matrix3X &Measurements) {
  cnpy::NpyArray measurements = cnpy::npy_load(file);

  assert(measurements.shape[1] == 3);

  if (measurements.shape[1] != 3) {
    LOG(ERROR) << "Inconsistent measurement size." << std::endl;
    exit(-1);
  }

  Measurements = Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>>(
                     measurements.data<double>(), measurements.shape[1],
                     measurements.shape[0])
                     .cast<Scalar>();

  Measurements.col(0).setZero();

  return 0;
}

int loadKeyPoint2Dand3DMeasurements(const std::string &file, Vector4 &CamParam,
                                    Matrix3X &Measurements2D,
                                    Matrix3X &Measurements3D) {
  std::ifstream input(file);

  if (input.fail()) {
    LOG(ERROR) << "Fail to open " << file << std::endl;
    exit(-1);
  }

  nlohmann::json measurements;

  input >> measurements;

  CamParam.setZero();
  Measurements2D.resize(3, 0);
  Measurements3D.resize(3, 0);

  assert(measurements.count("camera"));

  if (measurements.count("camera")) {
    const auto &camera = measurements["camera"];

    assert(camera.size() == 4);

    if (camera.size() != 4) {
      LOG(ERROR) << "Inconsistent camera parameter size." << std::endl;
      exit(-1);
    }

    for (int i = 0; i < 4; i++) {
      CamParam[i] = camera[i];
    }
  } else {
    LOG(ERROR) << "No 2D keypoints." << std::endl;
    exit(-1);
  }

  assert(measurements.count("keypoints2D"));

  if (measurements.count("keypoints2D")) {
    const auto &kpts2D = measurements["keypoints2D"];

    Measurements2D.resize(3, kpts2D.size());

    for (int i = 0; i < kpts2D.size(); i++) {
      const auto &kpt2D = kpts2D[i];

      assert(kpt2D.size() == 3);

      if (kpt2D.size() != 3) {
        LOG(ERROR) << "Inconsistent measurement size." << std::endl;
        exit(-1);
      }

      for (int j = 0; j < 3; j++) {
        Measurements2D(j, i) = kpt2D[j];
      }
    }
  } else {
    LOG(ERROR) << "No 2D keypoints." << std::endl;
    exit(-1);
  }

  assert(measurements.count("keypoints3D"));

  if (measurements.count("keypoints3D")) {
    const auto &kpts3D = measurements["keypoints3D"];

    Measurements3D.resize(3, kpts3D.size());

    for (int i = 0; i < kpts3D.size(); i++) {
      const auto &kpt3D = kpts3D[i];

      assert(kpt3D.size() == 3);

      if (kpt3D.size() != 3) {
        LOG(ERROR) << "Inconsistent measurement size." << std::endl;
        exit(-1);
      }

      for (int j = 0; j < 3; j++) {
        Measurements3D(j, i) = kpt3D[j];
      }
    }
  } else {
    LOG(ERROR) << "No 3D keypoints." << std::endl;
    exit(-1);
  }

  return 0;
}

int loadPOFMeasurements(const Matrix3X &KeyPoints3D, Matrix3X &Measurements) {
  const auto &POFInfo = KeyPointInfo::SMPL::POFInfo;

  Measurements.setZero(3, POFInfo.size());

  for (int i = 0; i < POFInfo.size(); i++) {
    assert(POFInfo[i][0] < KeyPoints3D.cols());
    assert(POFInfo[i][1] < KeyPoints3D.cols());

    if (POFInfo[i][0] >= KeyPoints3D.cols() ||
        POFInfo[i][1] >= KeyPoints3D.cols()) {
      LOG(ERROR) << "Inconsistent measurement size." << std::endl;
      exit(-1);
    }

    Measurements.col(i) =
        KeyPoints3D.col(POFInfo[i][1]) - KeyPoints3D.col(POFInfo[i][0]);
  }

  Measurements.colwise().normalize();

  return 0;
}

int loadJointLimitNeuralNetwork(const std::string &file,
                                AlignedVector<Matrix6> &weight,
                                AlignedVector<Vector6> &bias, VectorX &PRelua,
                                VectorX &PRelub) {
  cnpy::npz_t nn = cnpy::npz_load(file);

  const int size = nn.size();

  assert(size % 4 == 2);

  if (size % 4 != 2) {
    LOG(ERROR) << "Bad input file format." << std::endl;

    weight.clear();
    bias.clear();
    PRelua.resize(0);
    PRelub.resize(0);

    exit(-1);
  }

  const int num_layers = size / 4;

  weight.resize(num_layers + 1);
  bias.resize(num_layers + 1);
  PRelua.resize(num_layers);
  PRelub.resize(num_layers);

  cnpy::NpyArray w, b, a, s;

  for (int n = 0; n < num_layers; n++) {
    w = nn.at("weight" + std::to_string(n));
    b = nn.at("bias" + std::to_string(n));
    a = nn.at("alpha" + std::to_string(n));
    s = nn.at("beta" + std::to_string(n));

    weight[n] = Eigen::Map<Eigen::Matrix<double, 6, 6>>(w.data<double>(),
                                                        w.shape[1], w.shape[0])
                    .transpose()
                    .cast<Scalar>();
    bias[n] =
        Eigen::Map<Eigen::Matrix<double, 6, 1>>(b.data<double>(), b.num_vals)
            .cast<Scalar>();
    PRelua[n] = a.data<double>()[0];
    PRelub[n] = s.data<double>()[0];
  }

  w = nn.at("weight" + std::to_string(num_layers));
  b = nn.at("bias" + std::to_string(num_layers));

  weight[num_layers] = Eigen::Map<Eigen::Matrix<double, 6, 6>>(
                           w.data<double>(), w.shape[1], w.shape[0])
                           .transpose()
                           .cast<Scalar>();
  bias[num_layers] =
      Eigen::Map<Eigen::Matrix<double, 6, 1>>(b.data<double>(), b.num_vals)
          .cast<Scalar>();

  return 0;
}

int loadJointLimitPrior(const std::string &file,
                        AlignedVector<AlignedVector<Matrix6>> &weight,
                        AlignedVector<AlignedVector<Vector6>> &bias,
                        AlignedVector<VectorX> &PRelua,
                        AlignedVector<VectorX> &PRelub) {
  std::ifstream input(file);

  if (input.fail()) {
    LOG(ERROR) << "Fail to open " << file << std::endl;
    exit(-1);
  }

  nlohmann::json joints;

  input >> joints;

  int K = joints.size();

  weight.resize(K);
  bias.resize(K);
  PRelua.resize(K);
  PRelub.resize(K);

  for (int k = 0; k < K; k++) {
    const auto &joint = joints[k];

    int num_layers = joint.size() / 4;

    weight[k].resize(num_layers + 1);
    bias[k].resize(num_layers + 1);
    PRelua[k].resize(num_layers);
    PRelub[k].resize(num_layers);

    for (int n = 0; n < num_layers; n++) {
      const auto &w = joint["weight" + std::to_string(n)];
      const auto &b = joint["bias" + std::to_string(n)];
      const auto &a = joint["alpha" + std::to_string(n)];
      const auto &s = joint["beta" + std::to_string(n)];

      weight[k][n].setZero();
      bias[k][n].setZero();

      for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
          weight[k][n](i, j) = w[i][j].get<double>();
        }

        bias[k][n][i] = b[i].get<double>();
      }

      PRelua[k][n] = a.get<double>();
      PRelub[k][n] = s.get<double>();
    }

    const auto &w = joint["weight" + std::to_string(num_layers)];
    const auto &b = joint["bias" + std::to_string(num_layers)];

    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        weight[k][num_layers](i, j) = w[i][j].get<double>();
      }

      bias[k][num_layers][i] = b[i].get<double>();
    }
  }

  return 0;
}

int createInitialSMPLSkeleton(const Matrix3X &Measurements3D, VectorX &J) {
  J.setZero(Measurements3D.size());

  Eigen::Map<Matrix3X> Joints(J.data(), 3, Measurements3D.cols());

  // hip setup
  {
    Scalar L =
        0.5 * (Measurements3D.col(1) - Measurements3D.col(0)).stableNorm() +
        0.5 * (Measurements3D.col(4) - Measurements3D.col(0)).stableNorm();
    Joints.col(1) = Joints.col(0);
    Joints(0, 1) += L;
    Joints.col(4) = Joints.col(0);
    Joints(0, 4) -= L;
  }

  // knee setup
  {
    Scalar L =
        0.5 * (Measurements3D.col(2) - Measurements3D.col(1)).stableNorm() +
        0.5 * (Measurements3D.col(5) - Measurements3D.col(4)).stableNorm();
    Joints.col(2) = Joints.col(1);
    Joints(1, 2) -= L;
    Joints.col(5) = Joints.col(4);
    Joints(1, 5) -= L;
  }

  // ankle setup
  {
    Scalar L =
        0.5 * (Measurements3D.col(3) - Measurements3D.col(2)).stableNorm() +
        0.5 * (Measurements3D.col(6) - Measurements3D.col(5)).stableNorm();
    Joints.col(3) = Joints.col(2);
    Joints(1, 3) -= L;
    Joints.col(6) = Joints.col(5);
    Joints(1, 6) -= L;
  }

  // chest setup
  {
    Scalar L = (Measurements3D.col(7) - Measurements3D.col(0)).stableNorm();
    Joints.col(7) = Joints.col(0);
    Joints(1, 7) += L;
  }

  // thorax setup
  {
    Scalar L = (Measurements3D.col(8) - Measurements3D.col(7)).stableNorm();
    Joints.col(8) = Joints.col(7);
    Joints(1, 8) += L;
  }

  // neck setup
  {
    Vector3 Thorax = Measurements3D.col(8) - Measurements3D.col(7);
    Vector3 Neck = Measurements3D.col(9) - Measurements3D.col(8);

    Scalar LThorax = Thorax.stableNorm();
    Scalar LNeck = Neck.stableNorm();

    Scalar theta = M_PI / 4;

    Joints.col(9) = Joints.col(8);
    Joints(1, 9) += LNeck * sin(theta);
    Joints(2, 9) += LNeck * cos(theta);
  }

  // head top setup
  {
    Vector3 Neck = Measurements3D.col(9) - Measurements3D.col(8);
    Vector3 Head = Measurements3D.col(10) - Measurements3D.col(9);

    Scalar LNeck = Neck.stableNorm();
    Scalar LHead = Head.stableNorm();

    Scalar theta = acos(Neck.dot(Head) / (LNeck * LHead)) + M_PI / 4;

    Joints.col(10) = Joints.col(9);
    Joints(1, 10) += LHead * sin(theta);
    Joints(2, 10) += LHead * cos(theta);
  }

  // shoulder setup
  {
    Vector3 LShoulder = Measurements3D.col(11) - Measurements3D.col(8);
    Vector3 RShoulder = Measurements3D.col(14) - Measurements3D.col(8);
    Scalar LLShoulder = LShoulder.stableNorm();
    Scalar LRShoulder = RShoulder.stableNorm();

    Scalar theta =
        0.5 * acos(LShoulder.dot(RShoulder) / (LLShoulder * LRShoulder));

    Scalar L = 0.5 * LLShoulder + 0.5 * LRShoulder;
    Scalar X = L * sin(theta);
    Scalar Y = L * cos(theta);

    Joints.col(11) = Joints.col(8);
    Joints(0, 11) += X;
    Joints(1, 11) -= Y;
    Joints.col(14) = Joints.col(8);
    Joints(0, 14) -= X;
    Joints(1, 14) -= Y;
  }

  // elbow setup
  {
    Scalar L =
        0.5 * (Measurements3D.col(12) - Measurements3D.col(11)).stableNorm() +
        0.5 * (Measurements3D.col(15) - Measurements3D.col(14)).stableNorm();

    Joints.col(12) = Joints.col(11);
    Joints(0, 12) += L;
    Joints.col(15) = Joints.col(14);
    Joints(0, 15) -= L;
  }

  // wrist setup
  {
    Scalar L =
        0.5 * (Measurements3D.col(13) - Measurements3D.col(12)).stableNorm() +
        0.5 * (Measurements3D.col(16) - Measurements3D.col(15)).stableNorm();

    Joints.col(13) = Joints.col(12);
    Joints(0, 13) += L;
    Joints.col(16) = Joints.col(15);
    Joints(0, 16) -= L;
  }

  return 0;
}

int createInitialPinholeCameraFactors(
    const std::vector<std::array<int, 3>> &Info,
    const std::vector<std::array<int, 2>> &Index, const VectorX &KeyPoints,
    Scalar gm_est_sigma, Scalar gm_est_eps, const Matrix3X &Measurements,
    const VectorX &ConfidenceThreshold2D,
    std::vector<std::shared_ptr<Initializer::PinholeCameraFactor>> &factors) {
  factors.clear();

  assert(KeyPoints.size() % 3 == 0);
  assert(Measurements.cols() == ConfidenceThreshold2D.size());

  if (KeyPoints.size() % 3 ||
      Measurements.cols() != ConfidenceThreshold2D.size()) {
    LOG(ERROR) << "Inconsistent keypoints or joint locations." << std::endl;

    exit(-1);
  }

  const int M = Measurements.cols();
  const int I = Index.size();
  const int K = KeyPoints.size() / 3;

  factors.reserve(Info.size());

  for (const auto &info : Info) {
    const auto &m = info[0];
    const auto &i = info[1];
    const auto &v = info[2];

    assert(m < M);
    assert(i < I);
    assert(v < K);

    const int &n = Index[i][1];

    assert(n < K);

    if (m >= M || i >= I || v >= K || n >= K) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    const auto &measurement = Measurements.col(m);
    const auto &confidence_threshold = ConfidenceThreshold2D(m);

    if (measurement[2] < confidence_threshold) {
      continue;
    }

    Vector3 kpt = KeyPoints.segment<3>(3 * v) - KeyPoints.segment<3>(3 * n);
    factors.push_back(std::make_shared<Initializer::PinholeCameraFactor>(
        i, kpt, gm_est_sigma, gm_est_eps, measurement.head<2>(),
        measurement[2]));
  }

  return 0;
}

int createInitialPOFFactors(
    const AlignedVector<std::tuple<int, int, scope::Vector3, Scalar>> &Info,
    Scalar gm_est_sigma, Scalar gm_est_eps, const Matrix3X &Measurements,
    std::vector<std::shared_ptr<Initializer::POFFactor>> &factors) {
  factors.clear();

  const int M = Measurements.cols();

  factors.reserve(Info.size());

  for (const auto &info : Info) {
    const auto &m = std::get<0>(info);
    const auto &i = std::get<1>(info);
    const auto &d = std::get<2>(info);
    const auto &confidence = std::get<3>(info);

    assert(m < M);

    if (m >= M) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    const auto &measurement = Measurements.col(m);

    if (measurement.stableNorm() < 1e-8) {
      continue;
    }

    factors.push_back(std::make_shared<Initializer::POFFactor>(
        i, d, gm_est_sigma, gm_est_eps, measurement, confidence));
  }

  return 0;
}

int createInitialDepthCameraFactors(
    const std::vector<std::tuple<int, int, int, Scalar>> &Info,
    const std::vector<std::array<int, 2>> &Index, const VectorX &KeyPoints,
    Scalar gm_est_sigma, Scalar gm_est_eps, const Matrix3X &Measurements,
    const Vector3 &root,
    std::vector<std::shared_ptr<Initializer::DepthCameraFactor>> &factors) {
  factors.clear();

  assert(KeyPoints.size() % 3 == 0);

  if (KeyPoints.size() % 3) {
    LOG(ERROR) << "Inconsistent keypoints or joint locations." << std::endl;

    exit(-1);
  }

  const int M = Measurements.cols();
  const int I = Index.size();
  const int K = KeyPoints.size() / 3;

  factors.reserve(Info.size());

  for (const auto &info : Info) {
    const auto &m = std::get<0>(info);
    const auto &i = std::get<1>(info);
    const auto &v = std::get<2>(info);
    const auto &confidence = std::get<3>(info);

    assert(m < M);
    assert(i < I);
    assert(v < K);

    const int &n = Index[i][1];

    assert(n < K);

    if (m >= M || i >= I || v >= K || n >= K) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    const Vector3 measurement = Measurements.col(m) + root;

    Vector3 kpt = KeyPoints.segment<3>(3 * v) - KeyPoints.segment<3>(3 * n);

    factors.push_back(std::make_shared<Initializer::DepthCameraFactor>(
        i, kpt, gm_est_sigma, gm_est_eps, measurement, confidence));
  }

  return 0;
}

int createInitialJointLimitFactors(
    const std::vector<std::array<int, 2>> &Index,
    const AlignedVector<AlignedVector<scope::Matrix6>> &weight,
    const AlignedVector<AlignedVector<Vector6>> &bias,
    const AlignedVector<VectorX> &PRelua, const AlignedVector<VectorX> &PRelub,
    const VectorX &scale,
    std::vector<std::shared_ptr<Initializer::JointLimitFactor>> &factors) {
  factors.clear();

  const int N = weight.size();

  assert(bias.size() == N);
  assert(PRelua.size() == N);
  assert(PRelub.size() == N);
  assert(scale.size() == N);

  if (bias.size() != N || PRelua.size() != N || PRelub.size() != N ||
      scale.size() != N) {
    LOG(ERROR) << "Inconsistent factor information or measurements."
               << std::endl;

    exit(-1);
  }

  for (int n = 1; n < Index.size(); n++) {
    const int &j = Index[n][0] - 1;

    assert(j < N && j >= 0);

    if (j >= N || j < 0) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    factors.push_back(std::make_shared<Initializer::JointLimitFactor>(
        n - 1, weight[j], bias[j], PRelua[j], PRelub[j], scale[j]));
  }

  return 0;
}

int createInitialPoseConstFactors(
    const AlignedVector<std::tuple<int, Vector6, Vector6, Vector6>>
        &PoseConstInfo,
    const AlignedVector<Pose> &PoseRef,
    std::vector<std::shared_ptr<Initializer::PoseConstFactor>> &factors) {
  factors.clear();

  const int N = PoseRef.size();
  assert(PoseConstInfo.size() == N);

  if (PoseConstInfo.size() != N) {
    LOG(ERROR) << "Inconsistent factor information." << std::endl;

    exit(-1);
  }

  for (int n = 0; n < N; n++) {
    const auto &info = PoseConstInfo[n];
    const auto &pose = std::get<0>(info);

    assert(pose >= 0);

    if (pose < 0) {
      LOG(ERROR) << "Inconsistent factor information." << std::endl;

      exit(-1);
    }

    const auto &weight = std::get<1>(info);
    const auto &lbnd = std::get<2>(info);
    const auto &ubnd = std::get<3>(info);

    factors.push_back(std::make_shared<Initializer::PoseConstFactor>(
        pose, weight, PoseRef[n], lbnd, ubnd));
  }

  return 0;
}

int createInitialJointConstFactors(
    const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
        &JointConstInfo,
    std::vector<std::shared_ptr<Initializer::JointConstFactor>> &factors) {
  factors.clear();

  const int N = JointConstInfo.size();

  for (int n = 0; n < N; n++) {
    const auto &info = JointConstInfo[n];
    const auto &joint = std::get<0>(info);

    assert(joint >= 0);

    if (joint < 0) {
      LOG(ERROR) << "Inconsistent factor information." << std::endl;

      exit(-1);
    }

    const auto &weight = std::get<1>(info);
    const auto &ref = std::get<2>(info);
    const auto &lbnd = std::get<3>(info);
    const auto &ubnd = std::get<4>(info);

    factors.push_back(std::make_shared<Initializer::JointConstFactor>(
        joint, weight, ref, lbnd, ubnd));
  }

  return 0;
}

int createInitialEulerAngleConstFactors(
    const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
        &JointConstInfo,
    std::vector<std::shared_ptr<Initializer::EulerAngleConstFactor>> &factors) {
  factors.clear();

  const int N = JointConstInfo.size();

  for (int n = 0; n < N; n++) {
    const auto &info = JointConstInfo[n];
    const auto &joint = std::get<0>(info);

    assert(joint >= 0);

    if (joint < 0) {
      LOG(ERROR) << "Inconsistent factor information." << std::endl;

      exit(-1);
    }

    const auto &weight = std::get<1>(info);
    const auto &ref = std::get<2>(info);
    const auto &lbnd = std::get<3>(info);
    const auto &ubnd = std::get<4>(info);

    factors.push_back(std::make_shared<Initializer::EulerAngleConstFactor>(
        joint, weight, ref, lbnd, ubnd));
  }

  return 0;
}

int createInitialRootPinholeCameraFactors(
    const std::vector<std::array<int, 2>> &JointPinholeInfo,
    const std::vector<std::array<int, 3>> &VertexPinholeInfo,
    const AlignedVector<Pose> &poses, const VectorX &Param,
    const MatrixX &JDirs, const VectorX &J, const MatrixX &KeyPointDirs,
    const VectorX &KeyPoints, Scalar gm_est_sigma, Scalar gm_est_eps,
    const Matrix3X &Measurements, const VectorX &ConfidenceThreshold2D,
    std::vector<std::shared_ptr<Initializer::PinholeCameraFactor>> &factors) {
  assert(KeyPoints.size() % 3 == 0);
  assert(J.size() % 3 == 0);
  assert(KeyPointDirs.rows() == KeyPoints.size());
  assert(JDirs.rows() == J.size());
  assert(Measurements.cols() == ConfidenceThreshold2D.size());

  factors.clear();

  if (KeyPoints.size() % 3 || J.size() % 3 ||
      KeyPointDirs.rows() != KeyPoints.size() || JDirs.rows() != J.size() ||
      Measurements.cols() != ConfidenceThreshold2D.size()) {
    LOG(ERROR) << "Inconsistent keypoints or joint locations." << std::endl;

    exit(-1);
  }

  const int M = Measurements.cols();
  const int N = poses.size();
  const int K = KeyPoints.size() / 3;

  for (const auto &info : JointPinholeInfo) {
    const auto &m = info[0];
    const auto &n = info[1];

    assert(m < M);
    assert(n < N);

    if (m >= M || n >= N) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    const auto &measurement = Measurements.col(m);
    const auto &confidence_threshold = ConfidenceThreshold2D(m);

    if (measurement[2] < confidence_threshold) {
      continue;
    }

    factors.push_back(std::make_shared<Initializer::PinholeCameraFactor>(
        0, poses[n].t, gm_est_sigma, gm_est_eps, measurement.head<2>(),
        measurement[2]));
  }

  Matrix3X KptDirs;
  Vector3 vertex;
  Vector3 p;

  for (const auto &info : VertexPinholeInfo) {
    const auto &m = info[0];
    const auto &n = info[1];
    const auto &k = info[2];

    assert(m < M);
    assert(n < N);
    assert(k < K);

    if (m >= M || n >= N || k >= K) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    const auto &measurement = Measurements.col(m);
    const auto &confidence_threshold = ConfidenceThreshold2D(m);

    if (measurement[2] < confidence_threshold) {
      continue;
    }

    KptDirs = KeyPointDirs.middleRows<3>(3 * k) - JDirs.middleRows<3>(3 * n);
    vertex = KeyPoints.segment<3>(3 * k) - J.segment<3>(3 * n);
    vertex.noalias() += KptDirs * Param;

    p = poses[n].t;
    p.noalias() += poses[n].R * vertex;

    factors.push_back(std::make_shared<Initializer::PinholeCameraFactor>(
        0, p, gm_est_sigma, gm_est_eps, measurement.head<2>(), measurement[2]));
  }

  return 0;
}

int createInitialRootPOFFactors(
    const AlignedVector<std::tuple<int, int, scope::Vector3, Scalar>>
        &UnitPOFInfo,
    const std::vector<std::tuple<int, int, int, int, Scalar>> &ScaledPOFInfo,
    const std::vector<std::tuple<int, int, int, int, int, Scalar>> &RelPOFInfo,
    const AlignedVector<Pose> &poses, const VectorX &Param,
    const MatrixX &JDirs, const VectorX &J, const MatrixX &KeyPointDirs,
    const VectorX &KeyPoints, const std::vector<int> &kintree,
    Scalar gm_est_sigma, Scalar gm_est_eps, const Matrix3X &Measurements,
    std::vector<std::shared_ptr<Initializer::POFFactor>> &factors) {
  assert(KeyPoints.size() % 3 == 0);
  assert(J.size() % 3 == 0);
  assert(KeyPointDirs.rows() == KeyPoints.size());
  assert(JDirs.rows() == J.size());

  factors.clear();

  if (KeyPoints.size() % 3 || J.size() % 3 ||
      KeyPointDirs.rows() != KeyPoints.size() || JDirs.rows() != J.size()) {
    LOG(ERROR) << "Inconsistent keypoints or joint locations." << std::endl;

    exit(-1);
  }

  const int M = Measurements.cols();
  const int N = poses.size();
  const int K = KeyPoints.size() / 3;

  Vector3 d;

  for (const auto &info : UnitPOFInfo) {
    const auto &m = std::get<0>(info);
    const auto &n = std::get<1>(info);
    const auto &S = std::get<2>(info);
    const auto &confidence = std::get<3>(info);

    assert(m < M);
    assert(n < N);

    if (m >= M || n >= N) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    const auto &measurement = Measurements.col(m);

    if (measurement.stableNorm() < 1e-8) {
      continue;
    }

    d.noalias() = poses[n].R * S;

    factors.push_back(std::make_shared<Initializer::POFFactor>(
        0, d, gm_est_sigma, gm_est_eps, measurement, confidence));
  }

  Matrix3X Dirs;
  Vector3 vertex;

  for (const auto &info : ScaledPOFInfo) {
    const auto &m = std::get<0>(info);
    const auto &n = std::get<1>(info);
    const auto &v0 = std::get<2>(info);
    const auto &v1 = std::get<3>(info);
    const auto &confidence = std::get<4>(info);

    assert(m < M);
    assert(n < N);
    assert(v0 < K);
    assert(v1 < K);

    if (m >= M || n >= N || v0 >= K || v1 >= K) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    const auto &measurement = Measurements.col(m);

    if (measurement.stableNorm() < 1e-8) {
      continue;
    }

    Dirs =
        KeyPointDirs.middleRows<3>(3 * v1) - KeyPointDirs.middleRows<3>(3 * v0);
    vertex = KeyPoints.segment<3>(3 * v1) - KeyPoints.segment<3>(3 * v0);
    vertex.noalias() += Dirs * Param;

    d.noalias() = poses[n].R * vertex;

    factors.push_back(std::make_shared<Initializer::POFFactor>(
        0, d, gm_est_sigma, gm_est_eps, measurement, confidence));
  }

  for (const auto &info : RelPOFInfo) {
    const auto &m = std::get<0>(info);
    const auto &n = std::get<1>(info);
    const auto &v0 = std::get<2>(info);
    const auto &v1 = std::get<3>(info);
    const auto &confidence = std::get<5>(info);

    assert(m < M);
    assert(n < N);
    assert(kintree[n] < n);
    assert(v0 < K);
    assert(v1 < K);

    if (m >= M || n > N || kintree[n] >= n || v0 >= K || v1 >= K) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    Vector3 measurement = Measurements.col(m) * std::get<4>(info);

    if (measurement.stableNorm() < 1e-8) {
      continue;
    }

    Dirs = KeyPointDirs.middleRows<3>(3 * v1) - JDirs.middleRows<3>(3 * n);
    vertex = KeyPoints.segment<3>(3 * v1) - J.segment<3>(3 * n);
    vertex.noalias() += Dirs * Param;
    d.noalias() = poses[n].R * vertex;

    Dirs = KeyPointDirs.middleRows<3>(3 * v0) - JDirs.middleRows<3>(3 * n);
    vertex = KeyPoints.segment<3>(3 * v0) - J.segment<3>(3 * n);
    vertex.noalias() += Dirs * Param;
    d.noalias() -= poses[kintree[n]].R * vertex;

    if (d.stableNorm() < 1e-8) {
      continue;
    }

    factors.push_back(std::make_shared<Initializer::POFFactor>(
        0, d, gm_est_sigma, gm_est_eps, measurement, confidence));
  }

  return 0;
}

int createJointPinholeCameraFactors(
    const std::vector<std::array<int, 2>> &Info, Scalar gm_est_sigma,
    Scalar gm_est_eps, const Matrix3X &Measurements,
    const VectorX &ConfidenceThreshold2D,
    std::vector<std::shared_ptr<JointPinholeCameraFactor>> &factors) {
  factors.clear();

  assert(Measurements.cols() == ConfidenceThreshold2D.size());

  if (Measurements.cols() != ConfidenceThreshold2D.size()) {
    LOG(ERROR) << "Inconsistent keypoints or joint locations." << std::endl;

    exit(-1);
  }

  const int M = Measurements.cols();

  factors.reserve(Info.size());

  for (const auto &info : Info) {
    assert(info[0] < M);

    if (info[0] >= M) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    const auto &measurement = Measurements.col(info[0]);
    const auto &confidence_threshold = ConfidenceThreshold2D(info[0]);

    if (measurement[2] < confidence_threshold) {
      continue;
    }

    factors.push_back(std::make_shared<JointPinholeCameraFactor>(
        info[1], gm_est_sigma, gm_est_eps, measurement.head<2>(),
        measurement[2]));
  }

  return 0;
}

int createVertexPinholeCameraFactors(
    const std::vector<std::array<int, 3>> &Info, int VertexParam,
    const MatrixX &JDirs, const VectorX &J, const MatrixX &KeyPointDirs,
    const VectorX &KeyPoints, Scalar gm_est_sigma, Scalar gm_est_eps,
    const Matrix3X &Measurements, const VectorX &ConfidenceThreshold2D,
    std::vector<std::shared_ptr<VertexPinholeCameraFactor>> &factors) {
  factors.clear();

  assert(KeyPoints.size() % 3 == 0);
  assert(J.size() % 3 == 0);
  assert(KeyPointDirs.rows() == KeyPoints.size());
  assert(JDirs.rows() == J.size());
  assert(Measurements.cols() == ConfidenceThreshold2D.size());

  if (KeyPoints.size() % 3 || J.size() % 3 ||
      KeyPointDirs.rows() != KeyPoints.size() || JDirs.rows() != J.size() ||
      Measurements.cols() != ConfidenceThreshold2D.size()) {
    LOG(ERROR) << "Inconsistent keypoints or joint locations." << std::endl;

    exit(-1);
  }

  const int M = Measurements.cols();
  const int N = J.size() / 3;
  const int K = KeyPoints.size() / 3;

  factors.reserve(Info.size());

  Matrix3X KptDirs;
  Vector3 Kpt;

  for (const auto &info : Info) {
    assert(info[0] < M);
    assert(info[1] < N);
    assert(info[2] < K);

    if (info[0] >= M || info[1] >= N || info[2] >= K) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    const auto &measurement = Measurements.col(info[0]);
    const auto &confidence_threshold = ConfidenceThreshold2D(info[0]);

    if (measurement[2] < confidence_threshold) {
      continue;
    }

    KptDirs = KeyPointDirs.middleRows<3>(3 * info[2]) -
              JDirs.middleRows<3>(3 * info[1]);
    Kpt = KeyPoints.segment<3>(3 * info[2]) - J.segment<3>(3 * info[1]);

    factors.push_back(std::make_shared<VertexPinholeCameraFactor>(
        info[1], VertexParam, KptDirs, Kpt, gm_est_sigma, gm_est_eps,
        measurement.head<2>(), measurement[2]));
  }

  return 0;
}

int createUnitPOFFactors(
    const AlignedVector<std::tuple<int, int, scope::Vector3, Scalar>> &Info,
    Scalar gm_est_sigma, Scalar gm_est_eps, const Matrix3X &Measurements,
    std::vector<std::shared_ptr<UnitPOFFactor>> &factors) {
  factors.clear();

  const int M = Measurements.cols();

  for (const auto &info : Info) {
    const auto &m = std::get<0>(info);

    assert(m < M);

    if (m >= M) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    const auto &measurement = Measurements.col(m);

    if (measurement.stableNorm() < 1e-8) {
      continue;
    }

    const auto &i = std::get<1>(info);
    const auto &S = std::get<2>(info);
    const auto &confidence = std::get<3>(info);

    factors.push_back(std::make_shared<UnitPOFFactor>(
        i, S, gm_est_sigma, gm_est_eps, measurement, confidence));
  }

  return 0;
}

int createScaledPOFFactors(
    const std::vector<std::tuple<int, int, int, int, Scalar>> &Info,
    int VertexParam, const MatrixX &KeyPointDirs, const VectorX &KeyPoints,
    Scalar gm_est_sigma, Scalar gm_est_eps, const Matrix3X &Measurements,
    std::vector<std::shared_ptr<ScaledPOFFactor>> &factors) {
  factors.clear();

  const int M = Measurements.cols();
  const int K = KeyPoints.size() / 3;

  assert(KeyPoints.size() % 3 == 0);
  assert(KeyPointDirs.rows() == KeyPoints.size());

  if (KeyPoints.size() % 3 || KeyPointDirs.rows() != KeyPoints.size()) {
    LOG(ERROR) << "Inconsistent keypoints." << std::endl;

    exit(-1);
  }

  std::array<Matrix3X, 2> UDirs;
  std::array<Vector3, 2> U;
  Vector3 measurement;

  for (const auto &info : Info) {
    const auto &m = std::get<0>(info);
    const auto &i = std::get<1>(info);
    const auto &v0 = std::get<2>(info);
    const auto &v1 = std::get<3>(info);
    const auto &confidence = std::get<4>(info);

    assert(m < M);
    assert(i >= 0);
    assert(v0 < K);
    assert(v1 < K);

    if (m >= M || i < 0 || v0 >= K || v1 >= K) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    measurement = Measurements.col(m);

    if (measurement.stableNorm() < 1e-8) {
      continue;
    }

    UDirs[0] = KeyPointDirs.middleRows<3>(3 * v0);
    UDirs[1] = KeyPointDirs.middleRows<3>(3 * v1);
    U[0] = KeyPoints.segment<3>(3 * v0);
    U[1] = KeyPoints.segment<3>(3 * v1);

    factors.push_back(std::make_shared<ScaledPOFFactor>(
        i, VertexParam, UDirs, U, gm_est_sigma, gm_est_eps, measurement,
        confidence));
  }

  return 0;
}

int createRelPOFFactors(
    const std::vector<std::tuple<int, int, int, int, int, Scalar>> &Info,
    int VertexParam, const MatrixX &JDirs, const VectorX &J,
    const MatrixX &KeyPointDirs, const VectorX &KeyPoints,
    const std::vector<int> &kintree, Scalar gm_est_sigma, Scalar gm_est_eps,
    const Matrix3X &Measurements,
    std::vector<std::shared_ptr<RelPOFFactor>> &factors) {
  factors.clear();

  const int M = Measurements.cols();
  const int K = KeyPoints.size() / 3;
  const int N = J.size() / 3;

  assert(KeyPoints.size() % 3 == 0);
  assert(J.size() % 3 == 0);
  assert(KeyPointDirs.rows() == KeyPoints.size());
  assert(JDirs.rows() == J.size());
  assert(kintree.size() == N);

  if (KeyPoints.size() % 3 || J.size() % 3 ||
      KeyPointDirs.rows() != KeyPoints.size() || JDirs.rows() != J.size() ||
      kintree.size() != N) {
    LOG(ERROR) << "Inconsistent keypoints or joint locations." << std::endl;

    exit(-1);
  }

  std::array<Matrix3X, 2> UDirs;
  std::array<Vector3, 2> U;
  Vector3 measurement;

  for (const auto &info : Info) {
    const auto &m = std::get<0>(info);
    const auto &i = std::get<1>(info);
    const auto &v0 = std::get<2>(info);
    const auto &v1 = std::get<3>(info);
    const auto &confidence = std::get<5>(info);

    assert(m < M);
    assert(i < N);
    assert(kintree[i] < i);
    assert(i > 0);
    assert(v0 < K);
    assert(v1 < K);

    if (m >= M || i > N || kintree[i] >= i || i <= 0 || v0 >= K || v1 >= K) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    measurement = Measurements.col(m) * std::get<4>(info);

    if (measurement.stableNorm() < 1e-8) {
      continue;
    }

    UDirs[0] = KeyPointDirs.middleRows<3>(3 * v0) - JDirs.middleRows<3>(3 * i);
    UDirs[1] = KeyPointDirs.middleRows<3>(3 * v1) - JDirs.middleRows<3>(3 * i);
    U[0] = KeyPoints.segment<3>(3 * v0) - J.segment<3>(3 * i);
    U[1] = KeyPoints.segment<3>(3 * v1) - J.segment<3>(3 * i);

    factors.push_back(std::make_shared<RelPOFFactor>(
        i, VertexParam, kintree, UDirs, U, gm_est_sigma, gm_est_eps,
        measurement, confidence));
  }

  return 0;
}

int createJointDepthCameraFactors(
    const std::vector<std::tuple<int, int, Scalar>> &Info, Scalar gm_est_sigma,
    Scalar gm_est_eps, const Matrix3X &Measurements, const Vector3 &root,
    std::vector<std::shared_ptr<JointDepthCameraFactor>> &factors) {
  factors.clear();

  const int M = Measurements.cols();

  factors.reserve(Info.size());

  for (const auto &info : Info) {
    const auto &m = std::get<0>(info);
    const auto &i = std::get<1>(info);
    const auto &confidence = std::get<2>(info);

    assert(m < M);

    if (m >= M) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    const Vector3 measurement = Measurements.col(m) + root;

    factors.push_back(std::make_shared<JointDepthCameraFactor>(
        i, gm_est_sigma, gm_est_eps, measurement, confidence));
  }

  return 0;
}

int createVertexDepthCameraFactors(
    const std::vector<std::tuple<int, int, int, Scalar>> &Info, int VertexParam,
    const MatrixX &JDirs, const VectorX &J, const MatrixX &KeyPointDirs,
    const VectorX &KeyPoints, Scalar gm_est_sigma, Scalar gm_est_eps,
    const Matrix3X &Measurements, const Vector3 &root,
    std::vector<std::shared_ptr<VertexDepthCameraFactor>> &factors) {
  factors.clear();

  assert(KeyPoints.size() % 3 == 0);
  assert(J.size() % 3 == 0);
  assert(KeyPointDirs.rows() == KeyPoints.size());
  assert(JDirs.rows() == J.size());

  if (KeyPoints.size() % 3 || J.size() % 3 ||
      KeyPointDirs.rows() != KeyPoints.size() || JDirs.rows() != J.size()) {
    LOG(ERROR) << "Inconsistent keypoints or joint locations." << std::endl;

    exit(-1);
  }

  const int M = Measurements.cols();
  const int N = J.size() / 3;
  const int K = KeyPoints.size() / 3;

  factors.reserve(Info.size());

  Matrix3X KptDirs;
  Vector3 Kpt;

  for (const auto &info : Info) {
    const auto &m = std::get<0>(info);
    const auto &i = std::get<1>(info);
    const auto &v = std::get<2>(info);
    const auto &confidence = std::get<3>(info);

    if (m >= M || i >= N || v >= K) {
      LOG(ERROR) << "Inconsistent factor information or measurements."
                 << std::endl;

      factors.clear();

      exit(-1);
    }

    const Vector3 measurement = Measurements.col(m) + root;

    KptDirs = KeyPointDirs.middleRows<3>(3 * v) - JDirs.middleRows<3>(3 * i);
    Kpt = KeyPoints.segment<3>(3 * v) - J.segment<3>(3 * i);

    factors.push_back(std::make_shared<VertexDepthCameraFactor>(
        i, VertexParam, KptDirs, Kpt, gm_est_sigma, gm_est_eps, measurement,
        confidence));
  }

  return 0;
}

int createJointConstFactors(
    const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
        &JointConstInfo,
    std::vector<std::shared_ptr<JointConstFactor>> &factors) {
  factors.clear();

  const int N = JointConstInfo.size();

  for (int n = 0; n < N; n++) {
    const auto &info = JointConstInfo[n];
    const auto &joint = std::get<0>(info);

    assert(joint >= 0);

    if (joint < 0) {
      LOG(ERROR) << "Inconsistent factor information." << std::endl;

      exit(-1);
    }

    const auto &weight = std::get<1>(info);
    const auto &ref = std::get<2>(info);
    const auto &lbnd = std::get<3>(info);
    const auto &ubnd = std::get<4>(info);

    factors.push_back(
        std::make_shared<JointConstFactor>(joint, weight, ref, lbnd, ubnd));
  }

  return 0;
}

int createEulerAngleConstFactors(
    const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
        &EulerAngleConstInfo,
    std::vector<std::shared_ptr<EulerAngleConstFactor>> &factors) {
  factors.clear();

  const int N = EulerAngleConstInfo.size();

  for (int n = 0; n < N; n++) {
    const auto &info = EulerAngleConstInfo[n];
    const auto &joint = std::get<0>(info);

    assert(joint >= 0);

    if (joint < 0) {
      LOG(ERROR) << "Inconsistent factor information." << std::endl;

      exit(-1);
    }

    const auto &weight = std::get<1>(info);
    const auto &ref = std::get<2>(info);
    const auto &lbnd = std::get<3>(info);
    const auto &ubnd = std::get<4>(info);

    factors.push_back(std::make_shared<EulerAngleConstFactor>(joint, weight,
                                                              ref, lbnd, ubnd));
  }

  return 0;
}

int createJointLimitFactors(
    const AlignedVector<AlignedVector<scope::Matrix6>> &weight,
    const AlignedVector<AlignedVector<Vector6>> &bias,
    const AlignedVector<VectorX> &PRelua, const AlignedVector<VectorX> &PRelub,
    const VectorX &scale,
    std::vector<std::shared_ptr<JointLimitFactor>> &factors) {
  factors.clear();

  const int N = weight.size();

  assert(bias.size() == N);
  assert(PRelua.size() == N);
  assert(PRelub.size() == N);
  assert(scale.size() == N);

  if (bias.size() != N || PRelua.size() != N || PRelub.size() != N ||
      scale.size() != N) {
    LOG(ERROR) << "Inconsistent factor information or measurements."
               << std::endl;

    exit(-1);
  }

  for (int n = 0; n < N; n++) {
    factors.push_back(std::make_shared<JointLimitFactor>(
        n, weight[n], bias[n], PRelua[n], PRelub[n], scale[n]));
  }

  return 0;
}

int solveReducedEPnP(const std::vector<int> &KeyPoint3Dto2D,
                     const Matrix3X &Measurements3D,
                     const Matrix3X &Measurements2D,
                     const VectorX &ConfidenceThreshold2D, Vector3 &t) {
  assert(Measurements3D.cols() == KeyPoint3Dto2D.size());
  assert(Measurements2D.cols() == ConfidenceThreshold2D.size());

  if (Measurements3D.cols() != KeyPoint3Dto2D.size() ||
      Measurements2D.cols() != ConfidenceThreshold2D.size()) {
    LOG(ERROR) << "Inconsistent keypoint information." << std::endl;

    exit(-1);
  }

  const auto &N = KeyPoint3Dto2D.size();
  const auto &M = ConfidenceThreshold2D.size();

  std::vector<int> sel;

  for (int n = 0; n < N; n++) {
    const auto &m = KeyPoint3Dto2D[n];

    assert(m < M);

    if (m >= M) {
      LOG(ERROR) << "Inconsistent keypoint information." << std::endl;

      exit(-1);
    }

    const auto &measurement2D = Measurements2D.col(m);

    if (measurement2D[2] < ConfidenceThreshold2D[m]) {
      continue;
    }

    sel.push_back(n);
  }

  if (sel.size() >= 2) {
    MatrixX3 A = MatrixX3::Zero(2 * sel.size(), 3);
    VectorX a = VectorX::Zero(2 * sel.size());

    for (int i = 0; i < sel.size(); i++) {
      const auto &n = sel[i];
      const auto &m = KeyPoint3Dto2D[n];

      const auto &measurement3D = Measurements3D.col(n);
      const auto &measurement2D = Measurements2D.col(m);

      A.block<2, 2>(2 * i, 0).setIdentity();
      A.block<2, 1>(2 * i, 2) = -measurement2D.head<2>();

      a.segment<2>(2 * i) =
          measurement2D.head<2>() * measurement3D[2] - measurement3D.head<2>();

      A.middleRows<2>(2 * i) *= measurement2D[2];
      a.segment<2>(2 * i) *= measurement2D[2];
    }

    t.noalias() = A.colPivHouseholderQr().solve(a);

    return sel.size();
  } else {
    t.head<2>().setZero();

    for (const auto &n : sel) {
      const auto &m = KeyPoint3Dto2D[n];
      t.head<2>() += Measurements2D.col(m).head<2>();
    }

    t.head<2>() /= std::max(1, (int)sel.size());

    t.head<2>() *= 2;
    t[2] = 2;

    return -1;
  }

  return 0;
}

int solvePose(
    const std::vector<std::tuple<int, int, int, Scalar>> &DepthCamInfo,
    const Vector3 &Root, const VectorX &KeyPoints,
    const Matrix3X &Measurements3D, Pose &pose) {
  Matrix3X P(3, DepthCamInfo.size());
  Matrix3X Q(3, DepthCamInfo.size());

  for (int n = 0; n < DepthCamInfo.size(); n++) {
    const auto &info = DepthCamInfo[n];

    const auto &m = std::get<0>(info);
    const auto &v = std::get<2>(info);

    assert(m < Measurements3D.cols());
    assert(3 * v + 2 < KeyPoints.size());

    if (m >= Measurements3D.cols() || 3 * v + 2 >= KeyPoints.size()) {
      LOG(ERROR) << "Inconsistent keypoint or measurement information"
                 << std::endl;

      exit(-1);
    }

    P.col(n) = KeyPoints.segment<3>(3 * v) - Root;
    Q.col(n) = Measurements3D.col(m);
  }

  Vector3 Pmean = P.rowwise().mean();
  Vector3 Qmean = Q.rowwise().mean();

  P.colwise() -= Pmean;
  Q.colwise() -= Qmean;

  Matrix3 M;

  M.noalias() = Q * P.transpose();

  projectToSO3(M, pose.R);
  pose.t = Qmean;
  pose.t.noalias() -= pose.R * Pmean;

  return 0;
}

int solveShape(
    const std::vector<std::tuple<std::array<int, 4>, Vector3>> &RelShapeInfo,
    const std::vector<std::tuple<std::array<int, 4>, Vector3>>
        &RelShapePairInfo,
    const MatrixX &ShapeSqrtCov, Scalar weight, const MatrixX &KeyPointDirs,
    const VectorX &KeyPoints, const Matrix3X &Measurements3D, VectorX &betas,
    Scalar &s) {
  assert(RelShapePairInfo.size() % 2 == 0);
  assert(KeyPointDirs.cols() == ShapeSqrtCov.cols());

  if (RelShapePairInfo.size() % 2 != 0 ||
      KeyPointDirs.cols() != ShapeSqrtCov.cols()) {
    LOG(ERROR) << "Inconsistent relative shape information" << std::endl;

    exit(-1);
  }

  const int N = RelShapeInfo.size() + RelShapePairInfo.size();
  const int P = KeyPointDirs.cols();

  MatrixX A(3 * N, P);
  VectorX a(3 * N);
  VectorX b(3 * N);

  Scalar L0, L1;

  int n = 0;
  for (; n < RelShapePairInfo.size(); n += 2) {
    const auto &index0 = std::get<0>(RelShapePairInfo[n]);
    const auto &index1 = std::get<0>(RelShapePairInfo[n + 1]);

    Scalar L = (Measurements3D.col(index0[3]) - Measurements3D.col(index0[2]))
                   .stableNorm() +
               (Measurements3D.col(index1[3]) - Measurements3D.col(index1[2]))
                   .stableNorm();
    L *= 0.5;

    A.middleRows<3>(3 * n) = KeyPointDirs.middleRows<3>(3 * index0[1]) -
                             KeyPointDirs.middleRows<3>(3 * index0[0]);
    a.segment<3>(3 * n) = KeyPoints.segment<3>(3 * index0[1]) -
                          KeyPoints.segment<3>(3 * index0[0]);
    b.segment<3>(3 * n) = L * std::get<1>(RelShapePairInfo[n]);
    L0 += a.segment<3>(3 * n).stableNorm();
    L1 += L;

    A.middleRows<3>(3 * n + 3) = KeyPointDirs.middleRows<3>(3 * index1[1]) -
                                 KeyPointDirs.middleRows<3>(3 * index1[0]);
    a.segment<3>(3 * n + 3) = KeyPoints.segment<3>(3 * index1[1]) -
                              KeyPoints.segment<3>(3 * index1[0]);
    b.segment<3>(3 * n + 3) = L * std::get<1>(RelShapePairInfo[n + 1]);
    L0 += a.segment<3>(3 * n + 3).stableNorm();
    L1 += L;
  }

  for (const auto &info : RelShapeInfo) {
    const auto &index = std::get<0>(info);

    Scalar L = (Measurements3D.col(index[3]) - Measurements3D.col(index[2]))
                   .stableNorm();

    A.middleRows<3>(3 * n) = KeyPointDirs.middleRows<3>(3 * index[1]) -
                             KeyPointDirs.middleRows<3>(3 * index[0]);
    a.segment<3>(3 * n) =
        KeyPoints.segment<3>(3 * index[1]) - KeyPoints.segment<3>(3 * index[0]);
    b.segment<3>(3 * n) = L * std::get<1>(info);
    L0 += a.segment<3>(3 * n).stableNorm();
    L1 += L;

    n++;
  }

  s = L1 / L0;

  VectorX c = b - s * a;

  for (int n = 2; n < A.rows(); n += 3) {
    A.row(n) *= 0.4;
    c[n] *= 0.4;
  }

  MatrixX M;
  VectorX m;

  M.noalias() = weight * ShapeSqrtCov.transpose() * ShapeSqrtCov;
  M.noalias() += A.transpose() * A;
  M.diagonal().array() += 1e-12;
  m.noalias() = A.transpose() * c;

  betas.noalias() = M.ldlt().solve(m);

  return 0;
}

int getKeyPointOrigin3D(const Pose &Root, const MatrixX &JDirs,
                        const VectorX &J, const MatrixX &KeyPointDirs,
                        const VectorX &KeyPoints, const VectorX &Param,
                        int OriginIndex, Vector3 &keypoint) {
  assert(JDirs.cols() == Param.size());
  assert(KeyPointDirs.cols() == Param.size());
  assert(JDirs.cols() == KeyPointDirs.cols());

  Matrix3X vDirs =
      KeyPointDirs.middleRows<3>(3 * OriginIndex) - JDirs.middleRows<3>(0);
  Vector3 vertex = KeyPoints.segment<3>(3 * OriginIndex) - J.segment<3>(0);
  vertex.noalias() += vDirs * Param;

  keypoint = Root.t;
  keypoint += Root.R * vertex;

  return 0;
}
}  // namespace scope
