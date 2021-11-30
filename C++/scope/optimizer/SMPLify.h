#pragma once

#include <scope/model/SMPL.h>
#include <scope/optimizer/Optimizer.h>

namespace scope {
namespace Optimizer {
template <int P, bool CamOpt>
class SMPLify
    : public Optimizer<SMPL<P>::NumJoints, P, SMPL<P>::NumVertices, CamOpt> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Model = SMPL<P>;
  using Optimization =
      Optimizer<Model::NumJoints, P, Model::NumVertices, CamOpt>;

  using Optimization::computeGradient;
  using Optimization::linearize;
  using Optimization::solveGaussNewton;

  SMPLify(const Model &smpl, const Options &options);

  virtual int solveGaussNewton() const override;

protected:
  std::shared_ptr<const Model> mSMPL;

protected:
  virtual int setupParameterInfo() override;

private:
  using Optimization::CamParamIndex;
  using Optimization::CamParamOffset;
  using Optimization::CamParamSize;
  using Optimization::DCamParamOffset;
  using Optimization::DFaceParamOffset;
  using Optimization::DJointOffset;
  using Optimization::DJointSize;
  using Optimization::DParamOffset;
  using Optimization::DParamSize;
  using Optimization::DPoseOffset;
  using Optimization::DPoseSize;
  using Optimization::DSize;
  using Optimization::DVertexParamOffset;
  using Optimization::FaceParamIndex;
  using Optimization::FaceParamOffset;
  using Optimization::FaceParamSize;
  using Optimization::MaxDSize;
  using Optimization::MaxNumCollisions;
  using Optimization::NumJoints;
  using Optimization::NumParams;
  using Optimization::NumPoses;
  using Optimization::NumShapes;
  using Optimization::ParamSize;
  using Optimization::Size;
  using Optimization::VertexParamIndex;
  using Optimization::VertexParamOffset;
  using Optimization::VertexParamSize;

  using Optimization::mvB;
  using Optimization::mvBp;
  using Optimization::mvBu;
  using Optimization::mvH;
  using Optimization::mvHuuInv;
  using Optimization::mvHxB;
  using Optimization::mvKuxp;
  using Optimization::mvM;

  using Optimization::mvE;
  using Optimization::mvhp;
  using Optimization::mvhu;
  using Optimization::mvhx;
  using Optimization::mvku;
  using Optimization::mvmp;
  using Optimization::mvmu;
  using Optimization::mvmx;

  using Optimization::mParamGN;
  using Optimization::mvJointGN;
  using Optimization::mvPoseGN;

  using Optimization::mDLambda;
  using Optimization::mLambda;
  using Optimization::mOptions;
};
} // namespace Optimizer
} // namespace scope
