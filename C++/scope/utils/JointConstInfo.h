#pragma once

#include <array>
#include <vector>

#include <scope/base/Types.h>

namespace scope {
namespace JointConstInfo {
namespace SMPL {
const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
    JointConstInfo{
        //[0]: joint index
        //[1]: weight
        //[2]: lower bound
        //[3]: upper bound
        {3, // left knee
         {5, 3, 3},
         (Matrix3() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished(),
         {-0.00 * M_PI, -0.05 * M_PI, -0.03 * M_PI},
         {+0.85 * M_PI, +0.05 * M_PI, +0.03 * M_PI}},
        {4, // right knee
         {5, 3, 3},
         (Matrix3() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished(),
         {-0.00 * M_PI, -0.05 * M_PI, -0.03 * M_PI},
         {+0.85 * M_PI, +0.05 * M_PI, +0.03 * M_PI}},
        {15,
         {3, 3, 3},
         (Matrix3() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished(),
         {-0.50 * M_PI, -0.75 * M_PI, -0.25 * M_PI},
         {+0.25 * M_PI, +0.00 * M_PI, +0.25 * M_PI}},
        {16,
         {3, 3, 3},
         (Matrix3() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished(),
         {-0.50 * M_PI, -0.00 * M_PI, -0.25 * M_PI},
         {+0.25 * M_PI, +0.75 * M_PI, +0.25 * M_PI}}};

const AlignedVector<std::tuple<int, Vector3, Matrix3, Vector3, Vector3>>
    EulerAngleConstInfo{
        //[0]: joint index
        //[1]: weight
        //[2]: lower bound
        //[3]: upper bound
        {0, // left hip
         {10, 3, 3},
         (Matrix3() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished(),
         {-0.67 * M_PI, -0.25 * M_PI, -0.17 * M_PI},
         {+0.50 * M_PI, +0.48 * M_PI, +0.25 * M_PI}},
        {1, // right hip
         {10, 3, 3},
         (Matrix3() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished(),
         {-0.67 * M_PI, -0.48 * M_PI, -0.25 * M_PI},
         {+0.50 * M_PI, +0.25 * M_PI, +0.17 * M_PI}},
        //{15,  // left upper arm
        //{3, 3, 3},
        //(Matrix3() << 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0).finished(),
        //{-0.50 * M_PI, -0.33 * M_PI, -0.60 * M_PI},
        //{+0.46 * M_PI, +0.48 * M_PI, +0.40 * M_PI}},
        //{16,  // right upper arm
        //{3, 3, 3},
        //(Matrix3() << 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0).finished(),
        //{-0.46 * M_PI, -0.33 * M_PI, -0.40 * M_PI},
        //{+0.50 * M_PI, +0.48 * M_PI, +0.60 * M_PI}},
        //{17,  // left upper arm
        //{3, 3, 3},
        //(Matrix3() << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0,
        // 0.0, 1.0).finished(),
        //{-0.00 * M_PI, -0.33 * M_PI, -0.15 * M_PI},
        //{+0.75 * M_PI, +0.33 * M_PI, +0.15 * M_PI}},
        //{18,  // right upper arm
        //{3, 3, 3},
        //(Matrix3() << 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0).finished(),
        //{-0.75 * M_PI, -0.33 * M_PI, -0.15 * M_PI},
        //{+0.00 * M_PI, +0.33 * M_PI, +0.15 * M_PI}}
    };
} // namespace SMPL

namespace SMPLH {
const AlignedVector<std::tuple<int, Vector3, Vector3, Vector3>> JointConstInfo{
    //[0]: joint index
    //[1]: weight
    //[2]: lower bound
    //[3]: upper bound
    {0, // left hip
     {10, 3, 3},
     {-0.67 * M_PI, -0.10 * M_PI, -0.20 * M_PI},
     {+0.50 * M_PI, +0.33 * M_PI, +0.33 * M_PI}},
    {1, // right hip
     {10, 3, 3},
     {-0.67 * M_PI, -0.33 * M_PI, -0.33 * M_PI},
     {+0.50 * M_PI, +0.10 * M_PI, +0.20 * M_PI}},
    {3, // left knee
     {5, 5, 5},
     {-0.00 * M_PI, -0.05 * M_PI, -0.03 * M_PI},
     {+0.80 * M_PI, +0.05 * M_PI, +0.03 * M_PI}},
    {4, // right knee
     {5, 5, 5},
     {-0.00 * M_PI, -0.05 * M_PI, -0.03 * M_PI},
     {+0.80 * M_PI, +0.05 * M_PI, +0.03 * M_PI}},
    {15,
     {3, 3, 3},
     {-0.50 * M_PI, -0.75 * M_PI, -0.25 * M_PI},
     {+0.25 * M_PI, +0.05 * M_PI, +0.25 * M_PI}},
    {16,
     {3, 3, 3},
     {-0.50 * M_PI, -0.05 * M_PI, -0.25 * M_PI},
     {+0.25 * M_PI, +0.75 * M_PI, +0.25 * M_PI}}};
}
} // namespace JointConstInfo
} // namespace scope
