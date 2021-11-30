#pragma once

#include <scope/model/Model.h>

namespace scope {
template <int P>
class SMPL : public Model<23, P, 6890> {
 public:
  const static std::vector<int> mvLeftLeg, mvRightLeg, mvLeftArm, mvRightArm,
      mvHead, mvBack;

  SMPL(const MatrixX& RelJDirs, const VectorX& RelJ,
       const MatrixX& VDirs, const VectorX& V);

  virtual const std::vector<int> & getKinematicsTree() const override;

 protected:
  virtual int setKinematicsTree() override;
};

extern template class SMPL<10>;
extern template class SMPL<11>;
}  // namespace scope
