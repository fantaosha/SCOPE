#pragma once

#include <memory>
#include <scope/base/Pose.h>
#include <vector>

#define SCOAT_INITIAL_EVALUATION_NEW                                           \
  virtual std::shared_ptr<Initializer::Factor::Evaluation> newEvaluation()     \
      const override {                                                         \
    return std::make_shared<Evaluation>();                                     \
  }

#define SCOAT_INITIAL_LINEARIZATION_NEW                                        \
  virtual std::shared_ptr<Initializer::Factor::Linearization>                  \
  newLinearization() const override {                                          \
    return std::make_shared<Linearization>();                                  \
  }

namespace scope {
namespace Initializer {
class Factor : public std::enable_shared_from_this<Factor> {
protected:
  std::string mName;

  int mIndex;

public:
  enum class Status { INVALID = -1, VALID = 0 };

  struct Evaluation : std::enable_shared_from_this<Evaluation> {
    Status status;

    VectorX error;

    Scalar f;

    Evaluation();

    virtual int reset();

    virtual int clear();
  };

  struct Linearization : std::enable_shared_from_this<Linearization> {
    Status status;

    Linearization();

    virtual int reset();

    virtual int clear();
  };

public:
  Factor(const std::string &name = "", int index = -1);

  virtual int evaluate(const AlignedVector<Pose> &poses,
                       const AlignedVector<Matrix3> &joints,
                       Evaluation &eval) const = 0;

  virtual int linearize(const AlignedVector<Pose> &poses,
                        const AlignedVector<Matrix3> &joints,
                        const Evaluation &eval, Linearization &lin) const = 0;

  virtual std::shared_ptr<Evaluation> newEvaluation() const {
    return std::make_shared<Evaluation>();
  }

  virtual std::shared_ptr<Linearization> newLinearization() const {
    return std::make_shared<Linearization>();
  }

  void setID(int index) { mIndex = index; }

  int getID() const { return mIndex; }

  const std::string &getName() const { return mName; }
};
} // namespace Initializer
} // namespace scope
