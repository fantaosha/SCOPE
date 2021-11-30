#pragma once

#include <scope/initializer/factor/Factor.h>

namespace scope {
namespace Initializer {
class JointLimitFactor : public Factor {
public:
  struct Evaluation : public Factor::Evaluation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    AlignedVector<Eigen::Array<Scalar, 6, 1>> outputs;
    AlignedVector<Eigen::Array<Scalar, 6, 1>> fin;
    AlignedVector<Eigen::Array<Scalar, 6, 1>> fexp[2];
    AlignedVector<Eigen::Array<Scalar, 6, 1>> flog;
    AlignedVector<Eigen::Array<bool, 6, 1>> selected;

    virtual int clear() override;
  };

  struct Linearization : public Factor::Linearization {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Matrix63 jacobian;

    AlignedVector<Matrix6> derivatives;
    AlignedVector<Vector6> slopes;

    virtual int clear() override;
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SCOAT_INITIAL_EVALUATION_NEW
  SCOAT_INITIAL_LINEARIZATION_NEW

  JointLimitFactor(int joint, const AlignedVector<Matrix6> &NNWeight,
                   const AlignedVector<Vector6> &NNBias, const VectorX &a,
                   const VectorX &b, Scalar scale = 1.0,
                   const std::string &name = "", int index = -1);

  virtual int evaluate(const AlignedVector<Pose> &poses,
                       const AlignedVector<Matrix3> &joints,
                       Factor::Evaluation &base_eval) const override;

  virtual int linearize(const AlignedVector<Pose> &poses,
                        const AlignedVector<Matrix3> &joints,
                        const Factor::Evaluation &base_eval,
                        Factor::Linearization &base_lin) const override;

  int getJoint() const { return mJoint; }

protected:
  int mJoint;

  int mnLayers;
  int mnInnerLayers;

  // weights for linear transformation
  AlignedVector<Matrix6> NNWeight;
  // biases for linear transformation
  AlignedVector<Vector6> NNBias;
  // parameters for PRelu
  VectorX NNPReLUScale[2];
  VectorX NNPReLURate;

protected:
  int evaluate(const Matrix3 &Omega, Evaluation &eval) const;

  int linearize(const Matrix3 &Omega, const Evaluation &eval,
                Linearization &lin) const;
};
} // namespace Initializer
} // namespace scope
