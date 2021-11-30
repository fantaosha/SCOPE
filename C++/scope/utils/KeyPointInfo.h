#pragma once

#include <array>
#include <scope/base/Types.h>

namespace scope {
namespace KeyPointInfo {
//----------------------------------------------
// 2D keypoint measurement index
//----------------------------------------------
// 0: nose
// 1: left eye
// 2: right eye
// 3: left ear
// 4: right ear
// 5: left upper arm
// 6: right upper arm
// 7: left elbow
// 8: right elow
// 9: left wrist
// 10: right wrist
// 11: left hip
// 12: right hip
// 13: left knee
// 14: right knee
// 15: left ankle
// 16: right ankle
// 17: head top
// 18: thorax
// 19: middle hip
// 20: left big toe
// 21: right big toe
// 22: left small toe
// 23: right small toe
// 24: left heel
// 25: right heel
// 26: chest
// 27: neck
//----------------------------------------------

//----------------------------------------------
// 3D keypoint measurement index
//----------------------------------------------
// 0: middile hip
// 1: left hip
// 2: left knee
// 3: left ankle
// 4: right hip
// 5: right knee
// 6: right ankle
// 7: chest
// 8: thorax
// 9: neck
// 10: head top
// 11: left upper arm
// 12: left elbow
// 13: left wrist
// 14: right upper arm
// 15: right elbow
// 16: right wrist
//----------------------------------------------

namespace SMPL {
const int KeyPointOriginIndex = 13;

const std::vector<std::array<int, 2>> KeyPointInfo{
    {15, 332},  // 0: nose
    {15, 2800}, // 1: left eye
    {15, 6260}, // 2: right eye
    {15, 583},  // 3: left ear
    {15, 4071}, // 4: right ear
    {10, 3216}, // 5: left big toe
    {11, 6617}, // 6: right big toe
    {10, 3226}, // 7: left small toe
    {11, 6624}, // 8: right small toe
    {7, 3387},  // 9: left heel
    {8, 6787},  // 10: right heel
    {15, 6890}, // 11: head top
    {3, 6891},  // 12: chest
    {0, 6892},  // 13: middile hip
    {0, 6893},  // 14: left hip
    {0, 6894},  // 15: right hip
    {9, 6911},  // 16: left upper arm
    {9, 6912},  // 17: right upper arm
    {0, 6895},  // 18: pelvis
    {1, 6896},  // 19: left hip
    {2, 6897},  // 20: right hip
    {3, 6898},  // 21: spine
    {4, 6899},  // 22: left knee
    {5, 6900},  // 23: right knee
    {6, 6901},  // 24: spine1
    {7, 6902},  // 25: left ankle
    {8, 6903},  // 26: right ankle
    {9, 6904},  // 27: spine2
    {10, 6905}, // 28: left toes
    {11, 6906}, // 29: right toes
    {12, 6907}, // 30: thorax
    {13, 6908}, // 31: left chest
    {14, 6909}, // 32: right chest
    {15, 6910}, // 33: neck
    {16, 6911}, // 34: left shoulder
    {17, 6912}, // 35: right shoulder
    {18, 6913}, // 36: left elbow
    {19, 6914}, // 37: right elbow
    {20, 6915}, // 38: left wrist
    {21, 6916}, // 39: right wrist
    {22, 6917}, // 40: left fingers
    {23, 6918}  // 41: right fingers
};

const std::vector<std::array<int, 3>> POFInfo{
    //[0]: 3D keypoint measurment index -- tail
    //[1]: 3D keypoint measurment index -- head
    {0, 1},   // 0: middle hip -> left hip
    {1, 2},   // 1: left hip -> left knee
    {2, 3},   // 2: left knee -> left ankle
    {0, 4},   // 3: middle hip -> right hip
    {4, 5},   // 4: right hip -> right knee
    {5, 6},   // 5: right knee -> right ankle
    {0, 7},   // 6: middle hip -> chest/spine1
    {7, 8},   // 7: chest -> thorax
    {8, 9},   // 8: thorax -> neck
    {9, 10},  // 9: neck -> head top
    {8, 11},  // 10: thorax -> left upper arm
    {11, 12}, // 11: left upper arm -> left elbow
    {12, 13}, // 12: left elbow -> left wrist
    {8, 14},  // 13: thorax -> right upper arm
    {14, 15}, // 14: right upper arm -> right elbow
    {15, 16}  // 15: right elbow -> right wrist
};

const std::vector<int> KeyPoint3Dto2D{
    // 2D keypoint measurement index
    19, // 0: middile hip
    11, // 1: left hip
    13, // 2: left knee
    15, // 3: left ankle
    12, // 4: right hip
    14, // 5: right knee
    16, // 6: right ankle
    26, // 7: chest
    18, // 8: thorax
    27, // 9: neck
    17, // 10: head top
    5,  // 11: left upper arm
    7,  // 12: left elbow
    9,  // 13: left wrist
    6,  // 14: right upper arm
    8,  // 15: right elbow
    10  // 16: right wrist
};

const std::vector<std::array<int, 2>> JointPinholeCameraFactorInfo{
    //[0]: 2D keypoint measurement index
    //[1]: SMPL body part index
    {7, 18},  // left elbow
    {8, 19},  // right elbow
    {9, 20},  // left wrist
    {10, 21}, // right wrist
    {13, 4},  // left knee
    {14, 5},  // right knee
    {15, 7},  // left ankle
    {16, 8},  // right ankle
    {18, 12}, // thorax
    {27, 15}  // neck
};

const std::vector<std::array<int, 3>> VertexPinholeCameraFactorInfo{
    //[0]: 2D keypoint measurement index
    //[1]: SMPL body part index
    //[2]: SMPL vertex index
    {0, 15, 0},   // nose
    {1, 15, 1},   // left eye
    {2, 15, 2},   // right eye
    {3, 15, 3},   // left ear
    {4, 15, 4},   // right ear
    {5, 9, 16},   // left upper arm
    {6, 9, 17},   // right upper arm
    {11, 0, 14},  // left hip
    {12, 0, 15},  // right hip
    {17, 15, 11}, // head top
    {19, 0, 13},  // middle hip
    {20, 10, 5},  // left big toe
    {21, 11, 6},  // right big toe
    {22, 10, 7},  // left small toe
    {23, 11, 8},  // right small toe
    {24, 7, 9},   // left heel
    {25, 8, 10},  // right heel
    {26, 3, 12}   // chest
};

const std::vector<std::array<int, 2>> UpperBodyJointPinholeCameraFactorInfo{
    //[0]: 2D keypoint measurement index
    //[1]: SMPL body part index
    {7, 18},  // left elbow
    {8, 19},  // right elbow
    {9, 20},  // left wrist
    {10, 21}, // right wrist
    {18, 12}, // thorax
    {27, 15}, // neck
};

const std::vector<std::array<int, 3>> UpperBodyVertexPinholeCameraFactorInfo{
    //[0]: 2D keypoint measurement index
    //[1]: SMPL body part index
    //[2]: SMPL vertex index
    {0, 15, 0},   // nose
    {1, 15, 1},   // left eye
    {2, 15, 2},   // right eye
    {3, 15, 3},   // left ear
    {4, 15, 4},   // right ear
    {5, 9, 15},   // left upper arm
    {6, 9, 16},   // right upper arm
    {11, 0, 13},  // left hip
    {12, 0, 14},  // right hip
    {17, 15, 11}, // head top
    {19, 0, 13},  // middle hip
    {26, 3, 12}   // chest
};

const std::vector<std::array<int, 2>> LowerBodyJointPinholeCameraFactorInfo{
    //[0]: 2D keypoint measurement index
    //[1]: SMPL body part index
    {13, 4}, // left knee
    {14, 5}, // right knee
    {15, 7}, // left ankle
    {16, 8}  // right ankle
};

const std::vector<std::array<int, 3>> LowerBodyVertexPinholeCameraFactorInfo{
    //[0]: 2D keypoint measurement index
    //[1]: SMPL body part index
    //[2]: SMPL vertex index
    {20, 10, 5}, // left big toe
    {21, 11, 6}, // right big toe
    {22, 10, 7}, // left small toe
    {23, 11, 8}, // right small toe
    {24, 7, 9},  // left heel
    {25, 8, 10}  // right heel
};

const std::vector<std::tuple<int, int, Scalar>> JointDepthCameraFactorInfo{
    //[0]: 3D keypoint measurment index
    //[1]: SMPL body part index
    //[2]: confidence
    {2, 4, 1.0},   // left knee
    {3, 7, 1.0},   // left ankle
    {5, 5, 1.0},   // right knee
    {6, 8, 1.0},   // right ankle
    {8, 12, 1.0},  // thorax
    {9, 15, 1.0},  // neck
    {12, 18, 1.0}, // left elbow
    {13, 20, 1.0}, // left wrist
    {15, 19, 1.0}, // right elbow
    {16, 21, 1.0}  // right wrist
};

const std::vector<std::tuple<int, int, int, Scalar>>
    VertexDepthCameraFactorInfo{
        //[0]: 3D keypoint measurment index
        //[1]: SMPL body part index
        //[2]: SMPL vertex index
        //[3]: confidence
        {0, 0, 13, 1.0},   // middle hip
        {1, 0, 14, 1.0},   // left hip
        {4, 0, 15, 1.0},   // right hip
        {7, 3, 12, 1.0},   // chest
        {10, 15, 11, 1.0}, // head top
        {11, 9, 16, 1.0},  // left upper arm
        {14, 9, 17, 1.0}   // right upper arm
    };

const std::vector<std::tuple<int, int, int, Scalar>>
    OriginDepthCameraFactorInfo{
        //[0]: 3D keypoint measurment index
        //[1]: SMPL body part index
        //[2]: SMPL vertex index
        //[3]: confidence
        {0, 0, 13, 50.0}, // middle hip
    };

const AlignedVector<std::tuple<int, int, scope::Vector3, Scalar>>
    UnitPOFFactorInfo{
        //[0]: POF measurement index
        //[1]: body part index
        //[2]: unit directional vector
        //[3]: confidence
        {0, 0, {1, 0, 0}, 1.0}, // middle hip -> left hip
        //{1, 1, {0, -1, 0}, 1.0},    // left hip -> left knee
        {2, 4, {0, -1, 0}, 1.0}, // left knee -> left ankle
        {3, 0, {-1, 0, 0}, 1.0}, // middle hip -> right hip
        //{4, 2, {0, -1, 0}, 1.0},    // right hip -> right knee
        {5, 5, {0, -1, 0}, 1.0},   // right knee -> right ankle
        {7, 9, {0, 1, 0}, 0.5},    // chest -> thorax
        {11, 16, {1, 0, 0}, 1.0},  // left upper arm -> left elbow
        {12, 18, {1, 0, 0}, 1.0},  // left elbow -> left wrist
        {14, 17, {-1, 0, 0}, 1.0}, // right upper arm -> right elbow
        {15, 19, {-1, 0, 0}, 1.0}  // right elbow -> right wrist
    };

const std::vector<std::tuple<int, int, int, int, Scalar>> ScaledPOFFactorInfo{
    //[0]: POF measurement index
    //[1]: body part index
    //[2]: vertex index -- tail
    //[3]: vertex index -- head
    //[4]: confidence
    {8, 12, 30, 33, 0.25}, // thorax -> neck
    {9, 15, 33, 11, 0.25}, // neck -> head top
    {10, 9, 30, 16, 0.5},  // thorax -> left upper arm
    {13, 9, 30, 17, 0.5}   // thorax -> right upper arm
};

const std::vector<std::tuple<int, int, int, int, int, Scalar>> RelPOFFactorInfo{
    //[0]: POF measurement index
    //[1]: body part index -- child
    //[2]: vertex index -- tail
    //[3]: vertex index -- head
    //[4]: inverse the POF measurement or not
    //[5]: confidence
    {6, 3, 13, 12, 1, 0.1}, // middle hip -> chest
    {1, 1, 14, 22, 1, 2.0}, // left hip -> left knee
    {4, 2, 15, 23, 1, 2.0}, // right hip -> right knee
};
} // namespace SMPL
} // namespace KeyPointInfo
} // namespace scope
