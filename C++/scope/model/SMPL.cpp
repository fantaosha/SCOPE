#include <scope/model/SMPL.h>

namespace scope {
template <int P>
SMPL<P>::SMPL(const MatrixX& RelJDirs, const VectorX& RelJ,
              const MatrixX& vDirs, const VectorX& v)
    : Model<23, P, 6890>(HumanModel::SMPL, RelJDirs, RelJ, vDirs, v) {
  setKinematicsTree();
}

template <int P>
const std::vector<int>& SMPL<P>::getKinematicsTree() const {
  static const std::vector<int> kintree = {
      -1,  // 0
      0,   // 1
      0,   // 2
      0,   // 3
      1,   // 4
      2,   // 5
      3,   // 6
      4,   // 7
      5,   // 8
      6,   // 9
      7,   // 10
      8,   // 11
      9,   // 12
      9,   // 13
      9,   // 14
      12,  // 15
      13,  // 16
      14,  // 17
      16,  // 18
      17,  // 19
      18,  // 20
      19,  // 21
      20,  // 22
      21   // 23
  };

  return kintree;
}

template <int P>
int SMPL<P>::setKinematicsTree() {
  auto& links = this->mvLinks;

  links.resize(this->NumJoints + 1);

  static const std::string names[] = {
      "pelvis",         // 0
      "leftThigh",      // 1
      "rightThigh",     // 2
      "spine",          // 3
      "leftCalf",       // 4
      "rightCalf",      // 5
      "spine1",         // 6
      "leftFoot",       // 7
      "rightFoot",      // 8
      "spine2",         // 9
      "leftToes",       // 10
      "rightToes",      // 11
      "neck",           // 12
      "leftShoulder",   // 13
      "rightShoulder",  // 14
      "head",           // 15
      "leftUpperArm",   // 16
      "rightUpperArm",  // 17
      "leftForeArm",    // 18
      "rightForeArm",   // 19
      "leftHand",       // 20
      "rightHand",      // 21
      "leftFingers",    // 22
      "rightFingers"    // 23
  };

  const auto& parents = getKinematicsTree();

  for (int i = 0; i <= this->NumJoints; i++) {
    assert(parents[i] < i);

    links[i].mId = i;
    links[i].mName = names[i];
    links[i].mParent = parents[i];
    links[i].mJoint = i - 1;
  }

  for (int i = 1; i <= this->NumJoints; i++) {
    links[links[i].mParent].mvChildren.push_back(i);
  }

  return 0;
}

template <int P>
const std::vector<int> SMPL<P>::mvLeftLeg = std::vector<int>{1, 4, 7, 10};
template <int P>
const std::vector<int> SMPL<P>::mvRightLeg = std::vector<int>{2, 5, 8, 11};
template <int P>
const std::vector<int> SMPL<P>::mvHead = std::vector<int>{12, 15};
template <int P>
const std::vector<int> SMPL<P>::mvLeftArm =
    std::vector<int>{13, 16, 18, 20, 22};
template <int P>
const std::vector<int> SMPL<P>::mvRightArm =
    std::vector<int>{14, 17, 19, 21, 23};
template <int P>
const std::vector<int> SMPL<P>::mvBack = std::vector<int>{3, 6, 9};

// const std::vector<int> SMPL::mvLeftLeg = std::vector<int>{10, 7, 4, 1};
// const std::vector<int> SMPL::mvRightLeg = std::vector<int>{11, 8, 5, 2};
// const std::vector<int> SMPL::mvHead = std::vector<int>{15, 12};
// const std::vector<int> SMPL::mvLeftArm = std::vector<int>{22, 20, 18, 16,
// 13}; const std::vector<int> SMPL::mvRightArm = std::vector<int>{23, 21, 19,
// 17, 14}; const std::vector<int> SMPL::mvBack = std::vector<int>{9, 6, 3};

template class SMPL<10>;
template class SMPL<11>;
}  // namespace scope
