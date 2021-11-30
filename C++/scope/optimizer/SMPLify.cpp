#include <memory>

#include <glog/logging.h>

#include <scope/factor/Factors.h>
#include <scope/optimizer/SMPLify.h>

#define FACTORIZE(i)                                                           \
  {                                                                            \
    const auto &link = links[i];                                               \
                                                                               \
    assert(i == link.id());                                                    \
                                                                               \
    const int ii = i - 1;                                                      \
                                                                               \
    assert(ii == link.joint());                                                \
                                                                               \
    const auto &children = link.children();                                    \
                                                                               \
    mvH[ii] = mvM[i];                                                          \
                                                                               \
    for (const auto &j : children) {                                           \
      const auto &jj = links[j].joint();                                       \
                                                                               \
      mvH[ii]                                                                  \
          .template topLeftCorner<DPoseSize + DParamSize,                      \
                                  DPoseSize + DParamSize>() +=                 \
          mvH[jj]                                                              \
              .template topLeftCorner<DPoseSize + DParamSize,                  \
                                      DPoseSize + DParamSize>();               \
    }                                                                          \
                                                                               \
    mvHxB[ii].noalias() = mvH[ii].template leftCols<6>() * mvB[ii];            \
                                                                               \
    mvH[ii].template block<VertexParamSize, DPoseSize + DParamSize>(           \
        DVertexParamOffset, DPoseOffset) +=                                    \
        mvHxB[ii]                                                              \
            .template block<DPoseSize + DParamSize, VertexParamSize>(          \
                DPoseOffset, 0)                                                \
            .transpose();                                                      \
                                                                               \
    mvH[ii].template bottomRows<DJointSize>() +=                               \
        mvHxB[ii].template rightCols<DJointSize>().transpose();                \
                                                                               \
    mvH[ii].template block<DParamSize + DJointSize, VertexParamSize>(          \
        DParamOffset, DVertexParamOffset) +=                                   \
        mvHxB[ii].template block<DParamSize + DJointSize, VertexParamSize>(    \
            DParamOffset, 0);                                                  \
                                                                               \
    mvH[ii]                                                                    \
        .template block<VertexParamSize + 3, VertexParamSize>(                 \
            DVertexParamOffset, DVertexParamOffset)                            \
        .noalias() +=                                                          \
        mvHxB[ii].template middleRows<3>(3).transpose() * mvBp[ii];            \
                                                                               \
    mvH[ii].template block<DJointSize, DJointSize>(DPoseSize + DParamSize,     \
                                                   DPoseSize + DParamSize) +=  \
        mvHxB[ii].template block<DJointSize, DJointSize>(DJointOffset,         \
                                                         VertexParamSize);     \
    mvH[ii]                                                                    \
        .template block<DJointSize, DJointSize>(DJointOffset, DJointOffset)    \
        .noalias() += mvBu[ii].transpose() *                                   \
                      mvHxB[ii].template block<DPoseSize, DJointSize>(         \
                          DPoseOffset, VertexParamSize);                       \
                                                                               \
    if (mOptions.method == Method::LM) {                                       \
      mvH[ii].diagonal().template segment<DJointSize>(DJointOffset).array() *= \
          mDLambda;                                                            \
      mLambda.template segment<DJointSize>(6 + DParamSize + 3 * ii) =          \
          mvH[ii].diagonal().template segment<DJointSize>(DJointOffset);       \
    }                                                                          \
                                                                               \
    mvH[ii].diagonal().template segment<DJointSize>(DJointOffset).array() +=   \
        mOptions.delta;                                                        \
                                                                               \
    mvHuuInv[ii] = mvH[ii]                                                     \
                       .template bottomRightCorner<DJointSize, DJointSize>()   \
                       .inverse();                                             \
                                                                               \
    mvKuxp[ii].template leftCols<DPoseSize + DParamSize>().noalias() =         \
        -mvHuuInv[ii] *                                                        \
        mvH[ii]                                                                \
            .template bottomLeftCorner<DJointSize, DPoseSize + DParamSize>();  \
                                                                               \
    mvH[ii]                                                                    \
        .template topLeftCorner<DPoseSize + DParamSize,                        \
                                DPoseSize + DParamSize>()                      \
        .noalias() +=                                                          \
        mvH[ii]                                                                \
            .template bottomLeftCorner<DJointSize, DPoseSize + DParamSize>()   \
            .transpose() *                                                     \
        mvKuxp[ii];                                                            \
  }

#define BACKWARD(i)                                                            \
  {                                                                            \
    const auto &link = links[i];                                               \
                                                                               \
    assert(i == link.id());                                                    \
                                                                               \
    const int ii = i - 1;                                                      \
                                                                               \
    assert(ii == link.joint());                                                \
                                                                               \
    const auto &children = link.children();                                    \
                                                                               \
    mvhx[ii] = mvmx[i];                                                        \
    mvhp[ii] = mvmp[i];                                                        \
    mvhu[ii] = mvmu[i];                                                        \
    mvE[ii] = 0;                                                               \
                                                                               \
    for (const auto &j : children) {                                           \
      const auto &jj = links[j].joint();                                       \
                                                                               \
      mvhx[ii] += mvhx[jj];                                                    \
      mvhp[ii] += mvhp[jj];                                                    \
      mvE[ii] += mvE[jj];                                                      \
    }                                                                          \
                                                                               \
    mvhp[ii].template segment<VertexParamSize>(VertexParamOffset).noalias() += \
        mvBp[ii].transpose() * mvhx[ii].template bottomRows<3>();              \
    mvhu[ii].noalias() += mvBu[ii].transpose() * mvhx[ii];                     \
                                                                               \
    mvku[ii].noalias() = -mvHuuInv[ii] * mvhu[ii];                             \
                                                                               \
    mvE[ii] += 0.5 * mvhu[ii].dot(mvku[ii]);                                   \
                                                                               \
    mvhx[ii].noalias() +=                                                      \
        mvH[ii]                                                                \
            .template block<DJointSize, DPoseSize>(DJointOffset, DPoseOffset)  \
            .transpose() *                                                     \
        mvku[ii];                                                              \
    mvhp[ii].noalias() += mvH[ii]                                              \
                              .template block<DJointSize, DParamSize>(         \
                                  DJointOffset, DParamOffset)                  \
                              .transpose() *                                   \
                          mvku[ii];                                            \
  }

#define BACKWARD_PASS(i)                                                       \
  FACTORIZE(i)                                                                 \
  BACKWARD(i)

#define FORWARD_PASS(i)                                                        \
  {                                                                            \
    const auto &link = links[i];                                               \
                                                                               \
    const auto &parent = link.parent();                                        \
    const auto &ii = i - 1;                                                    \
                                                                               \
    mvJointGN[ii] = mvku[ii];                                                  \
                                                                               \
    mvJointGN[ii].noalias() +=                                                 \
        mvKuxp[ii].template leftCols<6>() * mvPoseGN[parent];                  \
                                                                               \
    mvPoseGN[i] = mvPoseGN[parent];                                            \
    mvPoseGN[i].noalias() += mvBu[ii] * mvJointGN[ii];                         \
    mvPoseGN[i].template tail<3>().noalias() +=                                \
        mvBp[ii] * mParamGN.template segment<VertexParamSize>(                 \
                       DVertexParamOffset - DParamOffset);                     \
  }

namespace scope {
namespace Optimizer {
template <int P, bool CamOpt>
SMPLify<P, CamOpt>::SMPLify(const Model &smpl, const Options &options)
    : Optimization(std::make_shared<Model>(smpl), options) {
  mSMPL = std::dynamic_pointer_cast<const Model>(this->mModel);

  this->setupModelInfo();

  assert(Optimization::NumPoses == Optimization::NumJoints + 1);

  setupParameterInfo();
  this->setupFactors();
  this->setupOptimization();
}

template <int P, bool CamOpt> int SMPLify<P, CamOpt>::setupParameterInfo() {
  return 0;
}

template <int P, bool CamOpt> int SMPLify<P, CamOpt>::solveGaussNewton() const {
#if 0
#pragma omp parallel
  {
#pragma omp single nowait
    {
#pragma omp task
      { this->backwardPass(mSMPL->mvLeftArm); }

#pragma omp task
      { this->backwardPass(mSMPL->mvRightArm); }

#pragma omp task
      { this->backwardPass(mSMPL->mvHead); }

#pragma omp task
      { this->backwardPass(mSMPL->mvLeftLeg); }

#pragma omp task
      { this->backwardPass(mSMPL->mvRightLeg); }
    }
#pragma omp taskwait
  }

  this->backwardPass(mSMPL->mvBack);
  factorizeN();
  this->backwardN();

  this->forwardPass(mSMPL->mvBack);
#pragma omp parallel
  {
#pragma omp single nowait
    {
#pragma omp task
      { this->forwardPass(mSMPL->mvLeftArm); }

#pragma omp task
      { this->forwardPass(mSMPL->mvRightArm); }

#pragma omp task
      { this->forwardPass(mSMPL->mvHead); }

#pragma omp task
      { this->forwardPass(mSMPL->mvLeftLeg); }

#pragma omp task
      { this->forwardPass(mSMPL->mvRightLeg); }
    }
#pragma omp taskwait
  }
#else
#if 0
  const auto& links = mSMPL->getLinks();

#pragma omp parallel sections
  {
#pragma omp section
    {
      // head
      BACKWARD_PASS(15);
      BACKWARD_PASS(12);
    }

#pragma omp section
    {
      // left arm
      BACKWARD_PASS(22);
      BACKWARD_PASS(20);
      BACKWARD_PASS(18);
      BACKWARD_PASS(16);
      BACKWARD_PASS(13);
    }

#pragma omp section
    {
      // right arm
      BACKWARD_PASS(23);
      BACKWARD_PASS(21);
      BACKWARD_PASS(19);
      BACKWARD_PASS(17);
      BACKWARD_PASS(14);
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    {  // left leg
      BACKWARD_PASS(10);
      BACKWARD_PASS(7);
      BACKWARD_PASS(4);
      BACKWARD_PASS(1);
    }

#pragma omp section
    {  // right leg
      BACKWARD_PASS(11);
      BACKWARD_PASS(8);
      BACKWARD_PASS(5);
      BACKWARD_PASS(2);
    }

#pragma omp section
    {
      // torso
      BACKWARD_PASS(9);
      BACKWARD_PASS(6);
      BACKWARD_PASS(3);
    }
  }

  //this->linearize(0);
  this->factorizeN();
  this->backwardN();

#pragma omp parallel sections
  {
#pragma omp section
    {
      // torso
      FORWARD_PASS(3);
      FORWARD_PASS(6);
      FORWARD_PASS(9);
    }

#pragma omp section
    {  // left leg
      FORWARD_PASS(1);
      FORWARD_PASS(4);
      FORWARD_PASS(7);
      FORWARD_PASS(10);
    }

#pragma omp section
    {  // right leg
      FORWARD_PASS(2);
      FORWARD_PASS(5);
      FORWARD_PASS(8);
      FORWARD_PASS(11);
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    {
      // head
      FORWARD_PASS(12);
      FORWARD_PASS(15);
    }

#pragma omp section
    {
      // left arm
      FORWARD_PASS(13);
      FORWARD_PASS(16);
      FORWARD_PASS(18);
      FORWARD_PASS(20);
      FORWARD_PASS(22);
    }

#pragma omp section
    {
      // right arm
      FORWARD_PASS(14);
      FORWARD_PASS(17);
      FORWARD_PASS(19);
      FORWARD_PASS(21);
      FORWARD_PASS(23);
    }
  }
#else
#pragma omp parallel sections
  {
#pragma omp section
    { this->backwardPass(mSMPL->mvHead); }

#pragma omp section
    { this->backwardPass(mSMPL->mvLeftArm); }

#pragma omp section
    { this->backwardPass(mSMPL->mvRightArm); }
  }

#pragma omp parallel sections
  {
#pragma omp section
    { this->backwardPass(mSMPL->mvLeftLeg); }

#pragma omp section
    { this->backwardPass(mSMPL->mvRightLeg); }

#pragma omp section
    { this->backwardPass(mSMPL->mvBack); }
  }

  this->factorizeN();
  this->backwardN();

#pragma omp parallel sections
  {
#pragma omp section
    { this->forwardPass(mSMPL->mvLeftLeg); }

#pragma omp section
    { this->forwardPass(mSMPL->mvRightLeg); }

#pragma omp section
    { this->forwardPass(mSMPL->mvBack); }
  }

#pragma omp parallel sections
  {
#pragma omp section
    { this->forwardPass(mSMPL->mvHead); }

#pragma omp section
    { this->forwardPass(mSMPL->mvLeftArm); }

#pragma omp section
    { this->forwardPass(mSMPL->mvRightArm); }
  }
#endif
#endif

  return 0;
}

template class SMPLify<10, false>;
template class SMPLify<11, false>;
} // namespace Optimizer
} // namespace scope
