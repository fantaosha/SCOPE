#include <scope/model/Model.h>

namespace scope {
template <int K, int P, int N>
Model<K, P, N>::Model(const HumanModel& human, const MatrixX& RelJDirs,
                      const VectorX& RelJ, const MatrixX& VDirs,
                      const VectorX& V)
    : mHuman(human),
      mRawRelJDirs(RelJDirs),
      mRawRelJ(RelJ),
      mRawVDirs(VDirs),
      mRawV(V),
      mRelJDirs(mRawRelJDirs.data()),
      mRelJ(mRawRelJ.data()),
      mVDirs(mRawVDirs.data()),
      mV(mRawV.data()) {
  assert(mRelJDirs.rows() == 3 * K && mRelJDirs.cols() == P);
  assert(mVDirs.rows() == 3 * N && mVDirs.cols() == P);
  assert(mRelJ.rows() == 3 * K);
  assert(mV.rows() == 3 * N);

  mvRelJDirs.resize(K);
  mvRelJ.resize(K);

  for (int i = 0; i < K; i++) {
    mvRelJDirs[i] = mRelJDirs.template middleRows<3>(3 * i);
    mvRelJ[i] = mRelJ.template segment<3>(3 * i);
  }

  mvVDirs.resize(N);
  mvV.resize(N);

  for (int i = 0; i < N; i++) {
    mvVDirs[i] = mVDirs.middleRows(3 * i, 3);
    mvV[i] = mV.segment(3 * i, 3);
  }
}

template <int K, int P, int N>
Model<K, P, N>::Model(const Model<K, P, N>& model)
    : mHuman(model.mHuman),
      mRawRelJDirs(model.mRawRelJDirs),
      mRawRelJ(model.mRawRelJ),
      mRawVDirs(model.mRawVDirs),
      mRawV(model.mRawV),
      mRelJDirs(mRawRelJDirs.data()),
      mRelJ(mRawRelJ.data()),
      mVDirs(mRawVDirs.data()),
      mV(mRawV.data()),
      mvRelJDirs(model.mvRelJDirs),
      mvRelJ(model.mvRelJ),
      mvVDirs(model.mvVDirs),
      mvV(model.mvV),
      mvLinks(model.mvLinks),
      mvCollisionPairs(model.mvCollisionPairs) {}

template <int K, int P, int N>
Model<K, P, N>::Model(Model<K, P, N>&& model)
    : mHuman(std::move(model.mHuman)),
      mRawRelJDirs(std::move(model.mRawRelJDirs)),
      mRawRelJ(std::move(model.mRawRelJ)),
      mRawVDirs(std::move(model.mRawVDirs)),
      mRawV(std::move(model.mRawV)),
      mRelJDirs(std::move(model.mRelJDirs)),
      mRelJ(std::move(model.mRelJ)),
      mVDirs(std::move(model.mVDirs)),
      mV(std::move(model.mV)),
      mvRelJDirs(std::move(model.mvRelJDirs)),
      mvRelJ(std::move(model.mvRelJ)),
      mvVDirs(std::move(model.mvVDirs)),
      mvV(std::move(model.mvV)),
      mvLinks(std::move(model.mvLinks)),
      mvCollisionPairs(std::move(model.mvCollisionPairs)) {}

template <int K, int P, int N>
const HumanModel& Model<K, P, N>::getHumanModel() const {
  return mHuman;
}

template <int K, int P, int N>
const Eigen::Map<const Matrix<3 * K, P>>& Model<K, P, N>::getRelJDirs() const {
  return mRelJDirs;
}

template <int K, int P, int N>
const Eigen::Map<const Vector<3 * K>>& Model<K, P, N>::getRelJ() const {
  return mRelJ;
}

template <int K, int P, int N>
const Eigen::Map<const Matrix<3 * N, P>>& Model<K, P, N>::getVDirs() const {
  return mVDirs;
}

template <int K, int P, int N>
const Eigen::Map<const Vector<3 * N>>& Model<K, P, N>::getV() const {
  return mV;
}

template <int K, int P, int N>
const std::vector<Link>& Model<K, P, N>::getLinks() const {
  return mvLinks;
}

template <int K, int P, int N>
const std::vector<std::array<int, 2>>& Model<K, P, N>::getCollisionsPairs()
    const {
  return mvCollisionPairs;
}

template <int K, int P, int N>
int Model<K, P, N>::FKIterate(AlignedVector<Pose>& T, int i, const Pose& T0,
                              const AlignedVector<Matrix3>& Omega,
                              const Eigen::Map<Vector<3 * K>>& mu) const {
  assert(T.size() == K + 1);
  assert(i >= 1 && i <= K);
  assert(Omega.size() == K);
  assert(mu.size() == 3 * K);

  const auto& link = mvLinks[i];

  assert(i == link.id());

  const auto& ii = link.joint();
  const auto& parent = link.parent();

  const auto& Tpar = T[parent];
  auto& Ti = T[i];

  Ti.R.noalias() = Tpar.R * Omega[ii];
  Ti.t = Tpar.t;
  Ti.t.noalias() += Tpar.R * mu.template segment<3>(3 * ii);

  return 0;
}

template <int K, int P, int N>
int Model<K, P, N>::DFKIterate(AlignedVector<Matrix<3, P>>& Bp,
                               AlignedVector<Matrix63>& Bu, int i,
                               const AlignedVector<Pose>& T) const {
  assert(Bp.size() == K);
  assert(Bu.size() == K);
  assert(i >= 1 && i <= K);
  assert(T.size() == K + 1);

  const auto& link = mvLinks[i];

  assert(i == link.id());

  const auto& ii = link.joint();
  const auto& parent = link.parent();

  const auto& Tpar = T[parent];
  const auto& Ti = T[i];

  Bp[ii].noalias() = Tpar.R * mvRelJDirs[ii];

  Bu[ii].template topRows<3>() = Tpar.R;
  Bu[ii].template block<3, 1>(3, 0).noalias() = Ti.t.cross(Tpar.R.col(0));
  Bu[ii].template block<3, 1>(3, 1).noalias() = Ti.t.cross(Tpar.R.col(1));
  Bu[ii].template block<3, 1>(3, 2).noalias() = Ti.t.cross(Tpar.R.col(2));

  return 0;
}

template <int K, int P, int N>
int Model<K, P, N>::DFKIterate(AlignedVector<Eigen::Map<Matrix<6, P + 3>>>& B,
                               int i, const AlignedVector<Pose>& T) const {
  assert(B.size() == K);
  assert(B.size() == K);
  assert(i >= 1 && i <= K);
  assert(T.size() == K + 1);

  const auto& link = mvLinks[i];

  assert(i == link.id());

  const auto& ii = link.joint();
  const auto& parent = link.parent();

  const auto& Tpar = T[parent];
  const auto& Ti = T[i];

  B[ii].template topLeftCorner<3, P>().setZero();
  B[ii].template bottomLeftCorner<3, P>().noalias() = Tpar.R * mvRelJDirs[ii];

  B[ii].template topRightCorner<3, 3>() = Tpar.R;
  B[ii].template block<3, 1>(3, P + 0).noalias() = Ti.t.cross(Tpar.R.col(0));
  B[ii].template block<3, 1>(3, P + 1).noalias() = Ti.t.cross(Tpar.R.col(1));
  B[ii].template block<3, 1>(3, P + 2).noalias() = Ti.t.cross(Tpar.R.col(2));

  return 0;
}

template <int K, int P, int N>
int Model<K, P, N>::FK(AlignedVector<Pose>& T, Eigen::Map<Vector<3 * K>>& mu,
                       const Pose& T0, const AlignedVector<Matrix3>& Omega,
                       const VectorX& beta) const {
  assert(Omega.size() == K);
  assert(beta.size() == P);

  T.resize(K + 1);

  mu = mRelJ;
  mu.noalias() += mRelJDirs * beta.head<P>();

  T[0] = T0;

  for (int i = 1; i <= K; i++) {
    FKIterate(T, i, T0, Omega, mu);
  }

  return 0;
}

template <int K, int P, int N>
int Model<K, P, N>::DFK(AlignedVector<Matrix<3, P>>& Bp,
                        AlignedVector<Matrix63>& Bu,
                        const AlignedVector<Pose>& T) const {
  assert(T.size() == K + 1);

  Bp.resize(K);
  Bu.resize(K);

  for (int i = 1; i <= K; i++) {
    DFKIterate(Bp, Bu, i, T);
  }

  return 0;
}

template <int K, int P, int N>
int Model<K, P, N>::DFK(AlignedVector<Eigen::Map<Matrix<6, P + 3>>>& B,
                        const AlignedVector<Pose>& T) const {
  assert(T.size() == K + 1);

  assert(B.size() == K);

  for (int i = 1; i <= K; i++) {
    DFKIterate(B, i, T);
  }

  return 0;
}


template <int K, int P, int N> const int Model<K, P, N>::NumJoints;
template <int K, int P, int N> const int Model<K, P, N>::NumParams;
template <int K, int P, int N> const int Model<K, P, N>::NumVertices;

template class Model<2, 2, 20>;
template class Model<23, 10, 6890>;
template class Model<51, 10, 6890>;
template class Model<23, 11, 6890>;
template class Model<51, 11, 6890>;
}  // namespace scope
