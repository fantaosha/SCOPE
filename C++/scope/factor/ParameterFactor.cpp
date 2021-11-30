#include <glog/logging.h>

#include <scope/factor/ParameterFactor.h>

namespace scope {
ParameterFactor::ParameterFactor(int param, const MatrixX &sigma,
                                 const VectorX &mean, const std::string &name,
                                 int index, bool active)
    : Factor({}, {}, {}, {param}, name, index, active), mSqrtCov(sigma),
      mMean(mean) {}

int ParameterFactor::evaluate(const AlignedVector<Pose> &poses,
                              const AlignedVector<VectorX> &shapes,
                              const AlignedVector<Matrix3> &joints,
                              const AlignedVector<VectorX> &params,
                              Factor::Evaluation &base_eval) const {
  auto &eval = dynamic_cast<Evaluation &>(base_eval);
  eval.clear();

  const auto &s = mvParams[0];
  assert(s >= 0 && s < params.size());

  if (s < 0 || s >= params.size()) {
    LOG(ERROR) << "The parameter must be valid." << std::endl;

    exit(-1);
  }

  eval.unscaledError = params[s] - mMean;

  eval.error.noalias() = mSqrtCov * eval.unscaledError;
  eval.f = eval.error.squaredNorm();

  eval.status = Status::VALID;

  return 0;
}

int ParameterFactor::linearize(const AlignedVector<Pose> &poses,
                               const AlignedVector<VectorX> &shapes,
                               const AlignedVector<Matrix3> &joints,
                               const AlignedVector<VectorX> &params,
                               const Factor::Evaluation &base_eval,
                               Factor::Linearization &base_lin) const {
  assert(base_eval.status == Status::VALID);

  auto &eval = base_eval;
  auto &lin = base_lin;

  lin.clear();

  if (eval.status != Status::VALID) {
    LOG(ERROR) << "The evaluation must be valid." << std::endl;

    exit(-1);
  }

  const auto &s = mvParams[0];
  assert(s >= 0 && s < params.size());

  if (s >= params.size()) {
    LOG(ERROR) << "The parameter must be valid." << std::endl;

    exit(-1);
  }

  lin.jacobians[3].resize(mvParams.size());

  lin.jacobians[3][0] = mSqrtCov;

  lin.status = Status::VALID;

  return 0;
}

} // namespace scope
