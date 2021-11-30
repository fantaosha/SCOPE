#pragma once

#include <scope/base/Pose.h>
#include <scope/model/Link.h>

namespace scope {
template <int K, int P, int N>
class Model {
 public:
  // number of joints
  const static int NumJoints = K;
  // number of implicit parameters
  const static int NumParams = P;
  // number of vertices
  const static int NumVertices = N;

 protected:
  // human model
  HumanModel mHuman;

  const MatrixX mRawRelJDirs;
  const VectorX mRawRelJ;
  const MatrixX mRawVDirs;
  const VectorX mRawV;

  // linear map from shape parameters to joint locations
  const Eigen::Map<const Matrix<3 * K, P>> mRelJDirs;
  const Eigen::Map<const Vector<3 * K>> mRelJ;

  AlignedVector<Matrix<3, P>> mvRelJDirs;
  AlignedVector<Vector3> mvRelJ;

  // linear map from shape parameters to mesh vertices
  const Eigen::Map<const Matrix<3 * N, P>> mVDirs;
  const Eigen::Map<const Vector<3 * N>> mV;

  AlignedVector<Matrix<3, P>> mvVDirs;
  AlignedVector<Vector3> mvV;

  // body parts
  std::vector<Link> mvLinks;

  // pairs of body parts to check collisions
  std::vector<std::array<int, 2>> mvCollisionPairs;

 protected:
  Model(const HumanModel& human, const MatrixX& RelJDirs, const VectorX& RelJ,
        const MatrixX& VDirs, const VectorX& v);

  Model(const Model<K, P, N>& model);
  Model(Model<K, P, N>&& model);
  Model& operator=(const Model<K, P, N>& model) = delete;
  Model& operator=(Model<K, P, N>&& model) = delete;

  virtual int setKinematicsTree() = 0;

  virtual int FKIterate(AlignedVector<Pose>& T, int i, const Pose& T0,
                        const AlignedVector<Matrix3>& Omega,
                        const Eigen::Map<Vector<3 * K>>& mu) const;

  virtual int DFKIterate(AlignedVector<Matrix<3, P>>& Bp,
                         AlignedVector<Matrix63>& Bu, int i,
                         const AlignedVector<Pose>& T) const;

  virtual int DFKIterate(AlignedVector<Eigen::Map<Matrix<6, P + 3>>>& B, int i,
                         const AlignedVector<Pose>& T) const;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  const HumanModel& getHumanModel() const;

  const Eigen::Map<const Matrix<3 * K, P>>& getRelJDirs() const;
  const Eigen::Map<const Vector<3 * K>>& getRelJ() const;

  const Eigen::Map<const Matrix<3 * N, P>>& getVDirs() const;
  const Eigen::Map<const Vector<3 * N>>& getV() const;

  const std::vector<Link>& getLinks() const;

  const std::vector<std::array<int, 2>>& getCollisionsPairs() const;

  virtual const std::vector<int> & getKinematicsTree() const = 0;

  // ---------------------------------------------------------
  // forward kinematics
  // ---------------------------------------------------------
  // T: body part poses
  // T0: the root pose
  // Omega: joint states
  // mu: joint locations
  virtual int FK(AlignedVector<Pose>& T, Eigen::Map<Vector<3 * K>>& mu,
                 const Pose& T0, const AlignedVector<Matrix3>& Omega,
                 const VectorX& beta) const;

  // ---------------------------------------------------------
  // linearization of forward kinematics
  // ---------------------------------------------------------
  virtual int DFK(AlignedVector<Matrix<3, P>>& Bp, AlignedVector<Matrix63>& Bu,
                  const AlignedVector<Pose>& T) const;

  virtual int DFK(AlignedVector<Eigen::Map<Matrix<6, P + 3>>>& B,
                  const AlignedVector<Pose>& T) const;
};

extern template class Model<2, 2, 20>;
extern template class Model<23, 10, 6890>;
extern template class Model<51, 10, 6890>;
}  // namespace scope
