#include <scope/initializer/factor/Factor.h>

namespace scope {
namespace Initializer {
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

  return 0;
}

int Factor::Linearization::clear() {
  reset();

  return 0;
}

Factor::Factor(const std::string &name, int index)
    : mIndex(index), mName(name) {}
} // namespace Initializer
} // namespace scope
