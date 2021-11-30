#include <scope/factor/Factor.h>

namespace scope {
Factor::Evaluation::Evaluation() : status(Status::INVALID) {}

int Factor::Evaluation::reset() {
  status = Status::INVALID;

  return 0;
}

int Factor::Evaluation::clear() {
  reset();

  error.resize(0);

  return 0;
}

Factor::Linearization::Linearization() : status(Status::INVALID) {}

int Factor::Linearization::reset() {
  status = Status::INVALID;

  jacobians[0].clear();
  jacobians[1].clear();
  jacobians[2].clear();
  jacobians[3].clear();

  return 0;
}

int Factor::Linearization::clear() {
  reset();

  jacobians[0].clear();
  jacobians[1].clear();
  jacobians[2].clear();
  jacobians[3].clear();

  return 0;
}

Factor::Factor(const std::vector<int> &poses, const std::vector<int> &shapes,
               const std::vector<int> &joints, const std::vector<int> &params,
               const std::string &name, int index, bool active)
    : mvPoses(poses), mvShapes(shapes), mvJoints(joints), mvParams(params),
      mIndex(index), mName(name), mActive(active) {}

} // namespace scope
