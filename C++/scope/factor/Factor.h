#pragma once

#include <memory>
#include <scope/base/Pose.h>
#include <vector>

#define SCOAT_FACTOR_EVALUATION_NEW                                            \
  virtual std::shared_ptr<Factor::Evaluation> newEvaluation() const override { \
    return std::make_shared<Evaluation>();                                     \
  }

#define SCOAT_FACTOR_LINEARIZATION_NEW                                         \
  virtual std::shared_ptr<Factor::Linearization> newLinearization()            \
      const override {                                                         \
    return std::make_shared<Linearization>();                                  \
  }

namespace scope {
class Factor : public std::enable_shared_from_this<Factor> {
protected:
  std::vector<int> mvPoses;
  std::vector<int> mvShapes;
  std::vector<int> mvJoints;
  std::vector<int> mvParams;

  std::string mName;

  int mIndex;

  bool mActive;

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

    // Jacobians[0]: linearization w.r.t. poses
    // Jacobians[1]: linearization w.r.t. shapes
    // Jacobians[2]: linearization w.r.t. joint states
    // Jacobians[3]: linearization w.r.t. implicit parameters
    AlignedVector<MatrixX> jacobians[4];

    Linearization();

    virtual int reset();

    virtual int clear();
  };

public:
  Factor(const std::vector<int> &poses, const std::vector<int> &shapes,
         const std::vector<int> &joints, const std::vector<int> &params,
         const std::string &name = "", int index = -1, bool active = true);

  virtual int evaluate(const AlignedVector<Pose> &poses,
                       const AlignedVector<VectorX> &shapes,
                       const AlignedVector<Matrix3> &joints,
                       const AlignedVector<VectorX> &params,
                       Evaluation &eval) const = 0;

  virtual int linearize(const AlignedVector<Pose> &poses,
                        const AlignedVector<VectorX> &shapes,
                        const AlignedVector<Matrix3> &joints,
                        const AlignedVector<VectorX> &params,
                        const Evaluation &eval, Linearization &lin) const = 0;

  virtual std::shared_ptr<Evaluation> newEvaluation() const {
    return std::make_shared<Evaluation>();
  }

  virtual std::shared_ptr<Linearization> newLinearization() const {
    return std::make_shared<Linearization>();
  }

  const std::vector<int> &getPoses() const { return mvPoses; }

  const std::vector<int> &getShapes() const { return mvShapes; }

  const std::vector<int> &getJoints() const { return mvJoints; }

  const std::vector<int> &getParams() const { return mvParams; }

  void setID(int index) { mIndex = index; }

  int getID() const { return mIndex; }

  const std::string &getName() const { return mName; }

  bool isActive() const { return mActive; }

  void activate() { mActive = true; }

  void deactivate() { mActive = false; }
};
} // namespace scope
