#pragma once

#include <array>
#include <vector>

#include <scope/base/Types.h>

namespace scope {
namespace InitialInfo {
namespace SMPL {
//----------------------------------------------------
// Shape
//----------------------------------------------------
const std::vector<std::tuple<std::array<int, 4>, Vector3>> RelShapeInfo{
    //[0][0]: 2D keypoint tail
    //[0][1]: 2D keypoint head
    //[0][2]: 3D keypoint tail
    //[0][3]: 3D keypoint head
    //[1]: 3D direction
    {{13, 12, 0, 7}, {0, 1, 0}} // middle hip -> chest
};

const std::vector<std::tuple<std::array<int, 4>, Vector3>> RelShapePairInfo{
    //[0][0]: 2D keypoint tail
    //[0][1]: 2D keypoint head
    //[0][2]: 3D keypoint tail
    //[0][3]: 3D keypoint head
    //[1]: 3D direction
    {{13, 14, 0, 1}, {1, 0, 0}},    // middle hip -> left hip
    {{13, 15, 0, 4}, {-1, 0, 0}},   // middle hip -> right hip
    {{22, 25, 2, 3}, {0, -1, 0}},   // left knee -> left ankle
    {{23, 26, 5, 6}, {0, -1, 0}},   // right knee -> right ankle
    {{16, 36, 11, 12}, {1, 0, 0}},  // left shoulder ->  left elbow
    {{17, 37, 14, 15}, {-1, 0, 0}}, // right shoulder ->  right elbow
    {{36, 38, 12, 13}, {1, 0, 0}},  // left elbow ->  left wrist
    {{37, 39, 15, 16}, {-1, 0, 0}}  // right elbow ->  right wrist
};

//----------------------------------------------------
// Torso
//----------------------------------------------------
const std::vector<std::array<int, 2>> TorsoIndex[2] = {
    //[0]: body part index on SMPL/SMPLH
    //[1]: 3D keypoint index as joint origin
    {
        {0, 0} // 0: (torso, middle hip)
    },
    {
        {0, 0} // 0: (torso, middle hip)
    }};

const std::vector<std::array<int, 2>> TorsoKinematicsTree{
    //[0]: parent
    //[1]: child
};

const std::vector<std::array<int, 3>> TorsoPinholeCameraFactorInfo[2] = {
    {
        //[0]: 2D keypoint measurement index
        //[2]: body part index on SMPL/SMPLH
        //[3]: keypoint index
        {0, 0, 0},        // nose
        {1, 0, 1},        // left eye
        {2, 0, 2},        // right eye
        {3, 0, 3},        // left ear
        {4, 0, 4},        // right ear
        {5, 0, 16},       // left upper arm
        {6, 0, 17},       // right upper arm
        {11, 0, 14},      // left hip
        {12, 0, 15},      // right hip
        {17, 0, 11},      // head top
        {18, 0, 12 + 18}, // thorax
        {19, 0, 13},      // middle hip
        {26, 0, 12},      // chest
        {27, 0, 15 + 18}  // neck
    },
    {
        //[0]: 2D keypoint measurement index
        //[2]: body part index on initial skeleton
        //[3]: joint index on initial skeleton
        {5, 0, 11},  // left upper arm
        {6, 0, 14},  // right upper arm
        {11, 0, 1},  // left hip
        {12, 0, 4},  // right hip
        {17, 0, 10}, // head top
        {18, 0, 8},  // thorax
        {19, 0, 0},  // middle hip
        {26, 0, 7},  // chest
        {27, 0, 9}   // neck
    }};

const AlignedVector<std::tuple<int, int, Vector3, Scalar>>
    TorsoPOFFactorInfo[2] = {
        {
            //[0]: POF measurement index
            //[1]: body part index
            //[2]: unit directional vector
            //[3]: confidence
            {0, 0, {1, 0, 0}, 1.0},  // middle hip -> left hip
            {3, 0, {-1, 0, 0}, 1.0}, // middle hip -> right hip
            {6, 0, {0, 1, 0}, 0.4},  // middle hip -> chest
            {7, 0, {0, 1, 0}, 0.4},  // chest -> thorax
            {10, 0, {1, 0, 0}, 1.0}, // thorax -> left upper arm
            {13, 0, {-1, 0, 0}, 1.0} // thorax -> right upper arm
        },
        {
            //[0]: POF measurement index
            //[1]: body part index
            //[2]: unit directional vector
            //[3]: confidence
            {0, 0, {1, 0, 0}, 1.0},  // middle hip -> left hip
            {3, 0, {-1, 0, 0}, 1.0}, // middle hip -> right hip
            {6, 0, {0, 1, 0}, 0.4},  // middle hip -> chest
            {7, 0, {0, 1, 0}, 0.4},  // chest -> thorax
            {10, 0, {1, 0, 0}, 1.0}, // thorax -> left upper arm
            {13, 0, {-1, 0, 0}, 1.0} // thorax -> right upper arm
        }};

const std::vector<std::tuple<int, int, int, Scalar>>
    TorsoDepthCameraFactorInfo[2] = {
        {
            //[0]: 3D keypoint measurement index
            //[1]: rigid body index
            //[2]: keypoint index
            {0, 0, 13, 1.0},      // middle hip
            {1, 0, 14, 1.0},      // left hip
            {4, 0, 15, 1.0},      // right hip
            {7, 0, 12, 0.2},      // chest
            {8, 0, 12 + 18, 1.0}, // thorax
            {9, 0, 15 + 18, 0.2}, // neck
            {11, 0, 16, 1.0},     // left upper arm
            {14, 0, 17, 1.0}      // right upper arm
        },

        {
            //[0]: 3D keypoint measurement index
            //[1]: rigid body index
            //[2]: joint index on initial skeleton
            {0, 0, 0, 1.0},   // middle hip
            {1, 0, 1, 1.0},   // left hip
            {4, 0, 4, 1.0},   // right hip
            {7, 0, 7, 0.2},   // chest
            {8, 0, 8, 1.0},   // thorax
                              //{9, 0, 9, 0.2},    // neck
            {11, 0, 11, 1.0}, // left upper arm
            {14, 0, 14, 1.0}  // right upper arm
        }};

//----------------------------------------------------
// extended torso
//----------------------------------------------------
const std::vector<std::array<int, 2>> ExtTorsoIndex[2] = {
    //[0]: body part index on SMPL/SMPLH
    //[1]: 3D keypoint index as joint origin
    {
        {0, 13}, // 0: (torso, middle hip)
        {6, 12}  // 1: (spine, chest)
    },
    {
        {0, 0}, // 0: (torso, middle hip)
        {6, 7}  // 1: (spine, chest)
    }};

const std::vector<std::array<int, 2>> ExtTorsoKinematicsTree{
    //[0]: parent
    //[1]: child
    {0, 1}, // middle hip -> chest
};

const std::vector<std::array<int, 3>> ExtTorsoPinholeCameraFactorInfo[2] = {
    {
        //[0]: 2D keypoint measurement index
        //[2]: body part index on SMPL/SMPLH
        //[3]: keypoint index
        {0, 1, 0},        // nose
        {1, 1, 1},        // left eye
        {2, 1, 2},        // right eye
        {3, 1, 3},        // left ear
        {4, 1, 4},        // right ear
        {5, 1, 16},       // left upper arm
        {6, 1, 17},       // right upper arm
        {17, 1, 11},      // head top
        {18, 1, 12 + 18}, // thorax
        {27, 1, 15 + 18}, // neck
        {11, 0, 14},      // left hip
        {12, 0, 15},      // right hip
        {19, 0, 13},      // middle hip
        {26, 0, 12}       // chest
    },
    {
        //[0]: 2D keypoint measurement index
        //[2]: body part index on SMPL/SMPLH
        //[3]: keypoint index
        {5, 1, 11}, // left upper arm
        {6, 1, 14}, // right upper arm
        {18, 1, 8}, // thorax
        {11, 0, 1}, // left hip
        {12, 0, 4}, // right hip
        {19, 0, 0}, // middle hip
        {26, 0, 7}  // chest
    }};

const AlignedVector<std::tuple<int, int, Vector3, Scalar>>
    ExtTorsoPOFFactorInfo[2] = {
        {
            //[0]: POF measurement index
            //[1]: body part index
            //[2]: unit directional vector
            //[3]: confidence
            {0, 0, {1, 0, 0}, 1.0},      // middle hip -> left hip
            {3, 0, {-1, 0, 0}, 1.0},     // middle hip -> right hip
            {6, 0, {0, 1, 0}, 0.4},      // middle hip -> chest
            {7, 1, {0, 1, 0}, 0.4},      // chest -> thorax
            {10, 1, {1, -0.25, 0}, 1.0}, // thorax -> left upper arm
            {13, 1, {-1, -0.25, 0}, 1.0} // thorax -> right upper arm
        },
        {
            //[0]: POF measurement index
            //[1]: body part index
            //[2]: unit directional vector
            //[3]: confidence
            {0, 0, {1, 0, 0}, 1.0},  // middle hip -> left hip
            {3, 0, {-1, 0, 0}, 1.0}, // middle hip -> right hip
            {6, 0, {0, 1, 0}, 0.4},  // middle hip -> chest
            {7, 1, {0, 1, 0}, 0.4},  // chest -> thorax
            {10, 1, {1, 0, 0}, 1.0}, // thorax -> left upper arm
            {13, 1, {-1, 0, 0}, 1.0} // thorax -> right upper arm
        }};

const std::vector<std::tuple<int, int, int, Scalar>>
    ExtTorsoDepthCameraFactorInfo[2] = {
        {
            //[0]: 3D keypoint measurement index
            //[1]: rigid body index
            //[2]: keypoint index
            {0, 0, 13, 1.0},      // middle hip
            {1, 0, 14, 1.0},      // left hip
            {4, 0, 15, 1.0},      // right hip
            {7, 0, 12, 0.2},      // chest
            {8, 1, 12 + 18, 1.0}, // thorax
            {9, 1, 15 + 18, 0.2}, // neck
            {11, 1, 16, 1.0},     // left upper arm
            {14, 1, 17, 1.0}      // right upper arm
        },
        {
            //[0]: 3D keypoint measurement index
            //[1]: rigid body index
            //[2]: joint index on initial skeleton
            {0, 0, 0, 1.0},   // middle hip
            {1, 0, 1, 1.0},   // left hip
            {4, 0, 4, 1.0},   // right hip
            {7, 0, 7, 0.2},   // chest
            {8, 1, 8, 1.0},   // thorax
            {11, 1, 11, 1.0}, // left upper arm
            {14, 1, 14, 1.0}  // right upper arm
        }};

//----------------------------------------------------
// Head
//----------------------------------------------------
const std::vector<std::array<int, 2>> HeadIndex[2] = {
    //[0]: body part index on SMPL/SMPLH
    //[1]: 3D keypoint index as joint origin
    {
        {15, 12 + 18} // 0: (head, thorax)
    },
    {
        {15, 8} // 0: (head, thorax)
    }};

const std::vector<std::array<int, 2>> HeadKinematicsTree{
    //[0]: parent
    //[1]: child
};

const std::vector<std::array<int, 3>> HeadPinholeCameraFactorInfo[2] = {
    {
        //[0]: 2D keypoint measurement index
        //[2]: body part index on SMPL/SMPLH
        //[3]: keypoint index
        {0, 0, 0},        // nose
        {1, 0, 1},        // left eye
        {2, 0, 2},        // right eye
        {3, 0, 3},        // left ear
        {4, 0, 4},        // right ear
        {17, 0, 11},      // head top
        {18, 0, 12 + 18}, // thorax
    },
    {
        //[0]: 2D keypoint measurement index
        //[2]: body part index on initial skeleton
        //[3]: joint index on initial skeleton
        {17, 0, 10}, // head top
        {18, 0, 8},  // thorax
        {27, 0, 9}   // neck
    }};

const std::vector<std::tuple<int, int, int, Scalar>>
    HeadDepthCameraFactorInfo[2] = {{
                                        //[0]: 3D keypoint measurement index
                                        //[1]: rigid body index
                                        //[2]: keypoint index
                                        {8, 0, 12 + 18, 1.0}, // thorax
                                        {9, 0, 15 + 18, 0.2}, // neck
                                        {10, 0, 11, 0.5},     // head top
                                    },

                                    {
                                        //[0]: 3D keypoint measurement index
                                        //[1]: rigid body index
                                        //[2]: joint index on initial skeleton
                                        {8, 0, 8, 1.0},   // thorax
                                        {9, 0, 9, 0.5},   // neck
                                        {10, 0, 10, 0.5}, // head top
                                    }};

//----------------------------------------------------
// left arm
//----------------------------------------------------
const std::vector<std::array<int, 2>> LeftArmIndex[2] = {
    //[0]: body part index on SMPL/SMPLH
    //[1]: 2D keypoint index as joint origin on SMPL/SMPLH
    //[2]: 3D keypoint index as joint origin on SMPL/SMPLH
    {
        {0, 13},      // 0: (torso, middle hip)
        {16, 16},     // 1: (left upper arm, left shoulder)
        {18, 18 + 18} // 2: (left forearm, left elbow)
    },
    {
        {3, 7},   // 0: (torso, middle hip)
        {16, 11}, // 1: (left upper arm, left upper arm)
        {18, 12}  // 2: (left forearm, left elbow)
    }};

const std::vector<std::array<int, 2>> LeftArmKinematicsTree{
    //[0]: parent
    //[1]: child
    {0, 1}, // torso -> left upper arm
    {1, 2}  // left upper arm -> left forearm
};

const std::vector<std::array<int, 3>> LeftArmPinholeCameraFactorInfo[2] = {
    {
        //[0]: 2D keypoint measurement index
        //[1]: body part index on initial skeleton
        //[2]: joint index on initial skeleton
        {7, 1, 18 + 18}, // left elbow
        {9, 2, 20 + 18}  // left wrist
    },
    {
        //[0]: 2D keypoint measurement index
        //[1]: rigid body index
        //[2]: joint index on initial skeleton
        {7, 1, 12}, // left elbow
        {9, 2, 13}  // left wrist
    }};

const AlignedVector<std::tuple<int, int, Vector3, Scalar>>
    LeftArmPOFFactorInfo[2] = {
        {
            //[0]: POF measurement index
            //[1]: body part index
            //[2]: unit directional vector
            //[3]: confidence
            {11, 1, {1, 0, 0}, 5.0}, // left upper arm -> left elbow
            {12, 2, {1, 0, 0}, 5.0}  // left elbow -> left wrist
        },
        {
            //[0]: POF measurement index
            //[1]: body part index
            //[2]: unit directional vector
            //[3]: confidence
            {11, 1, {1, 0, 0}, 5.0}, // left upper arm -> left elbow
            {12, 2, {1, 0, 0}, 5.0}  // left elbow -> left wrist
        }};

const std::vector<std::tuple<int, int, int, Scalar>>
    LeftArmDepthCameraFactorInfo[2] = {
        {
            //[0]: 3D keypoint measurement index
            //[1]: rigid body index
            //[2]: joint index on initial skeleton
            {12, 1, 18 + 18, 1.0}, // left elbow
            {13, 2, 20 + 18, 1.0}  // left wrist
        },
        {
            //[0]: 3D keypoint measurement index
            //[1]: rigid body index
            //[2]: joint index on initial skeleton
            {12, 1, 12, 1.0}, // left elbow
            {13, 2, 13, 1.0}  // left wrist
        }};

const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
    LeftArmJointConstInfo{
        //[0]: joint index
        //[1]: weight
        //[2]: reference rotation
        //[3]: lower bound
        //[4]: upper bound
        {1,
         {3, 3, 3},
         (Matrix3() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished(),
         {-0.50 * M_PI, -0.75 * M_PI, -0.25 * M_PI},
         {+0.25 * M_PI, +0.00 * M_PI, +0.25 * M_PI}}};

const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
    LeftArmEulerAngleConstInfo{
        //[0]: joint index
        //[1]: weight
        //[2]: reference rotation
        //[3]: lower bound
        //[4]: upper bound
        {0,
         {3, 3, 3},
         (Matrix3() << 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0).finished(),
         {-0.67 * M_PI, -0.33 * M_PI, -0.60 * M_PI},
         {+0.46 * M_PI, +0.48 * M_PI, +0.40 * M_PI}}};

//----------------------------------------------------
// right arm
//----------------------------------------------------
const std::vector<std::array<int, 2>> RightArmIndex[2] = {
    //[0]: body part index on SMPL/SMPLH
    //[1]: 2D keypoint index as joint origin on SMPL/SMPLH
    //[2]: 3D keypoint index as joint origin on SMPL/SMPLH
    {
        {0, 13},      // 0: (torso, middle hip)
        {17, 17},     // 1: (right upper arm, right shoulder)
        {19, 19 + 18} // 2: (right forearm, right elbow)
    },
    {
        {3, 7},   // 0: (torso, middle hip)
        {17, 14}, // 1: (right upper arm, right upper arm)
        {19, 15}  // 2: (right forearm, right elbow)
    }};

const std::vector<std::array<int, 2>> RightArmKinematicsTree{
    //[0]: parent
    //[1]: child
    {0, 1}, // torso -> right upper arm
    {1, 2}  // right upper arm -> right forearm
};

const std::vector<std::array<int, 3>> RightArmPinholeCameraFactorInfo[2] = {
    {
        //[0]: 2D keypoint measurement index
        //[1]: rigid body index
        //[2]: keypoint index
        {8, 1, 19 + 18}, // right elbow
        {10, 2, 21 + 18} // right wrist
    },
    {
        //[0]: 2D keypoint measurement index
        //[1]: rigid body index
        //[2]: joint index on initial skeleton
        {8, 1, 15}, // right elbow
        {10, 2, 16} // right wrist
    }};

const AlignedVector<std::tuple<int, int, Vector3, Scalar>>
    RightArmPOFFactorInfo[2] = {
        {
            //[0]: POF measurement index
            //[1]: body part index
            //[2]: unit directional vector
            //[3]: confidence
            {14, 1, {-1, 0, 0}, 5.0}, // right upper arm -> right elbow
            {15, 2, {-1, 0, 0}, 5.0}  // right elbow -> right wrist
        },
        {
            //[0]: POF measurement index
            //[1]: body part index
            //[2]: unit directional vector
            //[3]: confidence
            {14, 1, {-1, 0, 0}, 5.0}, // right upper arm -> right elbow
            {15, 2, {-1, 0, 0}, 5.0}  // right elbow -> right wrist
        }};

const std::vector<std::tuple<int, int, int, Scalar>>
    RightArmDepthCameraFactorInfo[2] = {
        {
            //[0]: 3D keypoint measurement index
            //[1]: rigid body index
            //[2]: keypoint index
            {15, 1, 19 + 18, 1.0}, // right elbow
            {16, 2, 21 + 18, 1.0}  // right wrist
        },
        {
            //[0]: 3D keypoint measurement index
            //[1]: rigid body index
            //[2]: joint index on initial skeleton
            {15, 1, 15, 1.0}, // right elbow
            {16, 2, 16, 1.0}  // right wrist
        }};

const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
    RightArmJointConstInfo{
        //[0]: joint index
        //[1]: weight
        //[2]: reference rotation
        //[3]: lower bound
        //[4]: upper bound
        {1,
         {3, 3, 3},
         (Matrix3() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished(),
         {-0.50 * M_PI, -0.00 * M_PI, -0.25 * M_PI},
         {+0.25 * M_PI, +0.75 * M_PI, +0.25 * M_PI}}};

const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
    RightArmEulerAngleConstInfo{
        //[0]: joint index
        //[1]: weight
        //[2]: reference rotation
        //[3]: lower bound
        //[4]: upper bound
        {0,
         {3, 3, 3},
         (Matrix3() << 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0).finished(),
         {-0.46 * M_PI, -0.33 * M_PI, -0.40 * M_PI},
         {+0.67 * M_PI, +0.48 * M_PI, +0.60 * M_PI}}};

//----------------------------------------------------
// left leg
//----------------------------------------------------
const std::vector<std::array<int, 2>> LeftLegIndex[2] = {
    // body part index on SMPL/SMPLH
    {
        {0, 13},    // 0: (torso, middle hip, middle hip)
        {1, 14},    // 1: (left tigh, left hip, left hip)
        {4, 4 + 18} // 2: (left calf, left knee, left knee)
    },
    {
        {0, 0}, // 0: (torso, middle hip, middle hip)
        {1, 1}, // 1: (left tigh, left hip, left hip)
        {4, 2}  // 2: (left calf, left knee, left knee)
    }};

const std::vector<std::array<int, 2>> LeftLegKinematicsTree{
    //[0]: parent
    //[1]: child
    {0, 1}, // torso -> left tigh
    {1, 2}  // left tigh -> left calf
};

const std::vector<std::array<int, 3>> LeftLegPinholeCameraFactorInfo[2] = {
    {
        //[0]: 2D keypoint measurement index
        //[1]: rigid body index
        //[2]: keypoint index
        //------------------------------------
        {0, 0, 0},        // noise
        {1, 0, 1},        // left eye
        {2, 0, 2},        // right eye
        {3, 0, 3},        // left ear
        {4, 0, 4},        // right ear
        {5, 0, 16},       // left upper arm
        {6, 0, 17},       // right upper arm
        {11, 0, 14},      // left hip
        {12, 0, 15},      // right hip
        {17, 0, 11},      // head top
        {18, 0, 12 + 18}, // thorax
        {19, 0, 13},      // middle hip
        {26, 0, 12},      // chest
        {27, 0, 15 + 18}, // neck
                          //------------------------------------
        {13, 1, 4 + 18},  // left knee
        {15, 2, 7 + 18}   // left ankle
    },
    {
        //[0]: 2D keypoint measurement index
        //[1]: rigid body index
        //[2]: joint index on initial skeleton
        //------------------------------------
        //{5, 0, 11},   // left upper arm
        //{6, 0, 14},   // right upper arm
        //{17, 0, 10},  // head top
        //{18, 0, 8},   // thorax
        //{27, 0, 9},   // neck
        {11, 0, 1}, // left hip
        {12, 0, 4}, // right hip
        {19, 0, 0}, // middle hip
        {26, 0, 7}, // chest
                    //------------------------------------
        {13, 1, 2}, // left knee
        {15, 2, 3}  // left ankle
    }};

const AlignedVector<std::tuple<int, int, Vector3, Scalar>>
    LeftLegPOFFactorInfo[2] = {
        {
            //[0]: POF measurement index
            //[1]: body part index
            //[2]: unit directional vector
            //[3]: confidence
            //------------------------------------
            {10, 0, {1, 0, 0}, 1.0},  // thorax -> left upper arm
            {13, 0, {-1, 0, 0}, 1.0}, // thorax -> right upper arm
            {0, 0, {1, 0, 0}, 1.0},   // middle hip -> left hip
            {3, 0, {-1, 0, 0}, 1.0},  // middle hip -> right hip
                                      //------------------------------------
            {1, 1, {0, -1, 0}, 1},    // left hip -> left knee
            {2, 2, {0, -1, 0}, 1},    // left knee -> left ankle
        },
        {
            //[0]: POF measurement index
            //[1]: body part index
            //[2]: unit directional vector
            //[3]: confidence
            //------------------------------------
            //{10, 0, {1, 0, 0}, 1.0},   // thorax -> left upper arm
            //{13, 0, {-1, 0, 0}, 1.0},  // thorax -> right upper arm
            {0, 0, {1, 0, 0}, 1.0},  // middle hip -> left hip
            {3, 0, {-1, 0, 0}, 1.0}, // middle hip -> right hip
                                     //------------------------------------
            {1, 1, {0, -1, 0}, 1},   // left hip -> left knee
            {2, 2, {0, -1, 0}, 1},   // left knee -> left ankle
        }};

const std::vector<std::tuple<int, int, int, Scalar>>
    LeftLegDepthCameraFactorInfo[2] = {
        {
            //[0]: 3D keypoint measurement index
            //[1]: rigid body index
            //[2]: keypoint index
            //------------------------------------
            {0, 0, 13, 1.0},      // middle hip
            {1, 0, 14, 1.0},      // left hip
            {4, 0, 15, 1.0},      // right hip
            {7, 0, 12, 0.2},      // chest
            {8, 0, 12 + 18, 1.0}, // thorax
            {9, 0, 15 + 18, 0.2}, // neck
            {11, 0, 16, 1.0},     // left upper arm
            {14, 0, 17, 1.0},     // right upper arm
                                  //------------------------------------
            {2, 1, 4 + 18, 1.0},  // left knee
            {3, 2, 7 + 18, 1.0}   // left ankle
        },
        {
            //[0]: 3D keypoint measurement index
            //[1]: rigid body index
            //[2]: keypoint index
            //------------------------------------
            //{8, 0, 8, 1.0},    // thorax
            //{9, 0, 9, 0.2},    // neck
            //{11, 0, 11, 1.0},  // left upper arm
            //{14, 0, 14, 1.0},  // right upper arm
            {0, 0, 0, 1.0}, // middle hip
            {1, 0, 1, 1.0}, // left hip
            {4, 0, 4, 1.0}, // right hip
            {7, 0, 7, 0.2}, // chest
                            //------------------------------------
            {2, 1, 2, 1.0}, // left knee
            {3, 2, 3, 1.0}  // left ankle
        }};

const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
    LeftLegEulerAngleConstInfo{
        //[0]: joint index
        //[1]: weight
        //[2]: reference rotation
        //[3]: lower bound
        //[4]: upper bound
        {0,
         {10, 3, 3},
         (Matrix3() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished(),
         {-0.67 * M_PI, -0.25 * M_PI, -0.17 * M_PI},
         {+0.50 * M_PI, +0.48 * M_PI, +0.25 * M_PI}}};

//----------------------------------------------------
// right leg
//----------------------------------------------------
const std::vector<std::array<int, 2>> RightLegIndex[2] = {
    // body part index on SMPL/SMPLH
    {
        {0, 13},     // 0: (torso, middle hip, middle hip)
        {2, 15},     // 1: (right tigh, right hip, right hip)
        {5, 5 + 18}, // 2: (right calf, right knee, right knee)
    },
    {
        {0, 0}, // 0: (torso, middle hip, middle hip)
        {2, 4}, // 1: (right tigh, right hip, right hip)
        {5, 5}, // 2: (right calf, right knee, right knee)
    }};

const std::vector<std::array<int, 2>> RightLegKinematicsTree{
    //[0]: parent
    //[1]: child
    {0, 1}, // torso -> right tigh
    {1, 2}  // rightTigh -> right calf
};

const std::vector<std::array<int, 3>> RightLegPinholeCameraFactorInfo[2] = {
    {
        //[0]: 2D keypoint measurement index
        //[1]: rigid body index
        //[2]: keypoint index
        //------------------------------------
        {0, 0, 0},        // noise
        {1, 0, 1},        // left eye
        {2, 0, 2},        // right eye
        {3, 0, 3},        // left ear
        {4, 0, 4},        // right ear
        {5, 0, 16},       // left upper arm
        {6, 0, 17},       // right upper arm
        {11, 0, 14},      // left hip
        {12, 0, 15},      // right hip
        {17, 0, 11},      // head top
        {18, 0, 12 + 18}, // thorax
        {19, 0, 13},      // middle hip
        {26, 0, 12},      // chest
        {27, 0, 15 + 18}, // neck
                          //------------------------------------
        {14, 1, 5 + 18},  // right knee
        {16, 2, 8 + 18}   // right ankle
    },
    {
        //[0]: 2D keypoint measurement index
        //[1]: rigid body index
        //[2]: joint index on initial skeleton
        //------------------------------------
        //{5, 0, 11},   // left upper arm
        //{6, 0, 14},   // right upper arm
        //{17, 0, 10},  // head top
        //{18, 0, 8},   // thorax
        //{27, 0, 9},   // neck
        {11, 0, 1}, // left hip
        {12, 0, 4}, // right hip
        {19, 0, 0}, // middle hip
        {26, 0, 7}, // chest
                    //------------------------------------
        {14, 1, 5}, // right knee
        {16, 2, 6}  // right ankle
    }};

const AlignedVector<std::tuple<int, int, Vector3, Scalar>>
    RightLegPOFFactorInfo[2] = {
        {
            //[0]: POF measurement index
            //[1]: body part index
            //[2]: unit directional vector
            //[3]: confidence
            //------------------------------------
            {0, 0, {1, 0, 0}, 1.0},   // middle hip -> left hip
            {3, 0, {-1, 0, 0}, 1.0},  // middle hip -> right hip
            {10, 0, {1, 0, 0}, 1.0},  // thorax -> left upper arm
            {13, 0, {-1, 0, 0}, 1.0}, // thorax -> right upper arm
                                      //------------------------------------
            {4, 1, {0, -1, 0}, 1},    // right hip -> right knee
            {5, 2, {0, -1, 0}, 1},    // right knee -> right ankle
        },
        {
            //[0]: POF measurement index
            //[1]: body part index
            //[2]: unit directional vector
            //[3]: confidence
            //------------------------------------
            {0, 0, {1, 0, 0}, 1.0},  // middle hip -> left hip
            {3, 0, {-1, 0, 0}, 1.0}, // middle hip -> right hip
                                     //------------------------------------
            {4, 1, {0, -1, 0}, 1},   // right hip -> right knee
            {5, 2, {0, -1, 0}, 1},   // right knee -> right ankle
        }};

const std::vector<std::tuple<int, int, int, Scalar>>
    RightLegDepthCameraFactorInfo[2] = {
        {
            //[0]: 2D keypoint measurement index
            //[1]: rigid body index
            //[2]: keypoint index
            //------------------------------------
            {0, 0, 13, 1.0},      // middle hip
            {1, 0, 14, 1.0},      // left hip
            {4, 0, 15, 1.0},      // right hip
            {7, 0, 12, 0.2},      // chest
            {8, 0, 12 + 18, 1.0}, // thorax
            {9, 0, 15 + 18, 0.2}, // neck
            {11, 0, 16, 1.0},     // left upper arm
            {14, 0, 17, 1.0},     // right upper arm
                                  //------------------------------------
            {5, 1, 5 + 18, 1.0},  // right knee
            {6, 2, 8 + 18, 1.0}   // right ankle
        },
        {
            //[0]: 3D keypoint measurement index
            //[1]: rigid body index
            //[2]: keypoint index
            //------------------------------------
            //{8, 0, 8, 1.0},    // thorax
            //{9, 0, 9, 0.2},    // neck
            //{11, 0, 11, 1.0},  // left upper arm
            //{14, 0, 14, 1.0},  // right upper arm
            {0, 0, 0, 1.0}, // middle hip
            {1, 0, 1, 1.0}, // left hip
            {4, 0, 4, 1.0}, // right hip
            {7, 0, 7, 0.2}, // chest
                            //------------------------------------
            {5, 1, 5, 1.0}, // right knee
            {6, 2, 6, 1.0}  // right ankle
        }};

const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
    RightLegEulerAngleConstInfo{
        //[0]: joint index
        //[1]: weight
        //[2]: reference rotation
        //[3]: lower bound
        //[4]: upper bound
        {0,
         {10, 3, 3},
         (Matrix3() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished(),
         {-0.67 * M_PI, -0.48 * M_PI, -0.25 * M_PI},
         {+0.50 * M_PI, +0.25 * M_PI, +0.17 * M_PI}}};
} // namespace SMPL
} // namespace InitialInfo
} // namespace scope
