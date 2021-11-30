#include <glog/logging.h>

#include <scope/factor/JointLimitFactor.h>
#include <scope/math/skew3.h>

namespace scope {
int JointLimitFactor::Evaluation::clear() {
  Factor::Evaluation::clear();

  outputs.clear();
  fin.clear();
  fexp[0].clear();
  fexp[1].clear();
  flog.clear();
  selected.clear();

  return 0;
}

int JointLimitFactor::Linearization::clear() {
  Factor::Linearization::clear();

  derivatives.clear();
  slopes.clear();

  return 0;
}

JointLimitFactor::JointLimitFactor(int joint, const AlignedVector<Matrix6> &W,
                                   const AlignedVector<Vector6> &w,
                                   const VectorX &a, const VectorX &b,
                                   Scalar scale, const std::string &name,
                                   int index, bool active)
    : Factor({}, {}, {joint}, {}, name, index, active), mnLayers(W.size()),
      mnInnerLayers(W.size() - 1), NNWeight(W), NNBias(w), NNPReLURate(b) {
  assert(NNWeight.size() == NNBias.size());
  assert(mnLayers > 0);
  assert(a.size() == mnInnerLayers);
  assert(b.size() == mnInnerLayers);

  NNPReLUScale[0] = (1 - a.array()) / b.array();
  NNPReLUScale[1] = a;

  JointLimitFactor::Evaluation eval;
  eval.outputs.resize(mnInnerLayers);
  eval.fin.resize(mnInnerLayers);
  eval.fexp[0].resize(mnInnerLayers);
  eval.fexp[1].resize(mnInnerLayers);
  eval.flog.resize(mnInnerLayers);
  eval.selected.resize(mnInnerLayers);

  Matrix3 Omega = Matrix3::Identity();
  evaluate(Omega, eval);
  Scalar fsqrt = std::max(std::sqrt(eval.f), 1e-4);

  NNBias[mnInnerLayers] *= scale;
  NNWeight[mnInnerLayers] *= scale;
  NNBias[mnInnerLayers] /= fsqrt;
  NNWeight[mnInnerLayers] /= fsqrt;
}

int JointLimitFactor::evaluate(const AlignedVector<Pose> &poses,
                               const AlignedVector<VectorX> &shapes,
                               const AlignedVector<Matrix3> &joints,
                               const AlignedVector<VectorX> &params,
                               Factor::Evaluation &base_eval) const {
  auto &eval = dynamic_cast<Evaluation &>(base_eval);
  eval.clear();

  const auto &i = mvJoints[0];
  assert(i >= 0 && i < joints.size());

  if (i >= joints.size()) {
    LOG(ERROR) << "The joint must be valid." << std::endl;

    exit(-1);
  }

  assert(mnLayers >= 0);

  eval.outputs.resize(mnInnerLayers);
  eval.fin.resize(mnInnerLayers);
  eval.fexp[0].resize(mnInnerLayers);
  eval.fexp[1].resize(mnInnerLayers);
  eval.flog.resize(mnInnerLayers);
  eval.selected.resize(mnInnerLayers);

  evaluate(joints[i], eval);

  eval.status = Status::VALID;

  return 0;
}

int JointLimitFactor::linearize(const AlignedVector<Pose> &poses,
                                const AlignedVector<VectorX> &shapes,
                                const AlignedVector<Matrix3> &joints,
                                const AlignedVector<VectorX> &params,
                                const Factor::Evaluation &base_eval,
                                Factor::Linearization &base_lin) const {
  auto &eval = dynamic_cast<const Evaluation &>(base_eval);
  auto &lin = dynamic_cast<Linearization &>(base_lin);

  lin.clear();

  assert(eval.status == Status::VALID);

  if (eval.status != Status::VALID) {
    LOG(ERROR) << "The evaluation must be valid." << std::endl;

    exit(-1);
  }

  const auto &i = mvJoints[0];
  assert(i >= 0 && i < joints.size());

  if (i < 0 || i >= joints.size()) {
    LOG(ERROR) << "The joint must be valid." << std::endl;

    exit(-1);
  }

  assert(mnLayers > 0);

  lin.derivatives.resize(mnLayers);
  lin.slopes.resize(mnInnerLayers);

  lin.jacobians[2].resize(1);

  linearize(joints[i], eval, lin);

  lin.status = Status::VALID;

  return 0;
}

int JointLimitFactor::evaluate(const Matrix3 &Omega, Evaluation &eval) const {
  Eigen::Map<const Vector6> input(Omega.data(), 6);

  eval.outputs[0] = NNBias[0];
  eval.outputs[0].matrix().noalias() += NNWeight[0] * input;

  eval.fin[0] = eval.outputs[0] * NNPReLURate[0];
  eval.selected[0] = eval.fin[0] >= 10;

  eval.fexp[0][0] = eval.selected[0].select(1e15, eval.fin[0].exp());
  eval.fexp[1][0] = eval.fexp[0][0] + 1;
  eval.fexp[0][0] += NNPReLUScale[1][0];

  eval.flog[0] = eval.selected[0].select(eval.fin[0], eval.fexp[1][0].log());

  eval.outputs[0] =
      NNPReLUScale[0][0] * eval.flog[0] + NNPReLUScale[1][0] * eval.outputs[0];

  for (int i = 1, ii = 0; i < mnInnerLayers; i++, ii++) {
    eval.outputs[i] = NNBias[i];
    eval.outputs[i].matrix().noalias() +=
        NNWeight[i] * eval.outputs[ii].matrix();

    eval.fin[i] = eval.outputs[i] * NNPReLURate[i];
    eval.selected[i] = eval.fin[i] >= 10;

    eval.fexp[0][i] = eval.selected[i].select(1e15, eval.fin[i].exp());
    eval.fexp[1][i] = eval.fexp[0][i] + 1;
    eval.fexp[0][i] += NNPReLUScale[1][i];

    eval.flog[i] = eval.selected[i].select(eval.fin[i], eval.fexp[1][i].log());

    eval.outputs[i] = NNPReLUScale[0][i] * eval.flog[i] +
                      NNPReLUScale[1][i] * eval.outputs[i];
  }

  eval.error.resize(6);
  Eigen::Map<Vector6> error(eval.error.data());

  error = NNBias[mnInnerLayers];
  error.noalias() += NNWeight[mnInnerLayers] * eval.outputs.back().matrix();

  eval.f = eval.error.squaredNorm();

  return 0;
}

int JointLimitFactor::linearize(const Matrix3 &Omega, const Evaluation &eval,
                                Linearization &lin) const {
  lin.jacobians[2][0].setZero(6, 3);

  Eigen::Map<scope::Matrix<6, 3>> J(lin.jacobians[2][0].data());

  lin.derivatives[mnInnerLayers] = NNWeight[mnInnerLayers];

  for (int i = mnInnerLayers, ii = mnInnerLayers - 1; i > 0; i--, ii--) {
    lin.slopes[ii] = eval.fexp[0][ii] / eval.fexp[1][ii];
    lin.derivatives[i] *= lin.slopes[ii].asDiagonal();
    lin.derivatives[ii].noalias() = lin.derivatives[i] * NNWeight[ii];
  }

  math::skew3::multR<math::OPS::SUB>(Omega.col(0),
                                     lin.derivatives[0].leftCols<3>(), J);
  math::skew3::multR<math::OPS::SUB>(Omega.col(1),
                                     lin.derivatives[0].rightCols<3>(), J);

  return 0;
}
} // namespace scope
