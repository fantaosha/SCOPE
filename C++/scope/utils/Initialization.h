#pragma once

#include <scope/initializer/Initializers.h>
#include <scope/model/SMPL.h>
#include <scope/utils/Setup.h>

namespace scope {
namespace Initialization {
namespace SMPL {
int initialize(const MatrixX &ShapeSqrtCov, Scalar FocalLength,
               // KeyPoint2D setup
               const Matrix3X &Measurements2D, Scalar KeyPoint2DGMSigma,
               Scalar KeyPoint2DGMEps,
               const VectorX &KeyPoint2DConfidenceThreshold,
               // POF setup
               const Matrix3X &POFMeasurements, Scalar POFGMSigma,
               Scalar POFGMEps,
               // KeyPoint3D setup
               const Matrix3X &Measurements3D, Scalar KeyPoint3DGMSigma,
               Scalar KeyPoint3DGMEps,
               // joint limit
               const AlignedVector<AlignedVector<Matrix6>> &JointLimitWeight,
               const AlignedVector<AlignedVector<Vector6>> &JointLimitBias,
               const AlignedVector<VectorX> &JointLimitPRelua,
               const AlignedVector<VectorX> &JointLimitPRelub,
               const Vector<23> &JointLimitScale,
               // initial estimates
               Pose &TorsoPose,      // torso pose
               Matrix3 &SpineJoint,  // spine joint
               AlignedVector<Matrix3> &LeftArmJoint,
               AlignedVector<Matrix3> &RightArmJoint,
               AlignedVector<Matrix3> &LeftLegJoint,
               AlignedVector<Matrix3> &RightLegJoint, VectorX &betas);

int initializeRootPose(const AlignedVector<Pose> &Poses, const VectorX &Param,
                       // SMPL model info
                       const std::vector<int> &kintree, const MatrixX &JDirs,
                       const VectorX &J, const MatrixX &KeyPointDirs,
                       const VectorX &KeyPoints,
                       // KeyPoint2D setup
                       Scalar FocalLength, const Matrix3X &Measurements2D,
                       Scalar KeyPoint2DGMSigma, Scalar KeyPoint2DGMEps,
                       const VectorX &KeyPoint2DConfidenceThreshold,
                       // POF setup
                       const Matrix3X &POFMeasurements, Scalar POFGMSigma,
                       Scalar POFGMEps,
                       // initial estimates
                       Pose &RootPose);
}  // namespace SMPL
}  // namespace Initialization
}  // namespace scope
