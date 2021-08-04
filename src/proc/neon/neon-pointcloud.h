// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#pragma once
#include "../pointcloud.h"

namespace librealsense
{
    class pointcloud_neon : public pointcloud
    {
    public:
        pointcloud_neon();
    private:
        const float3* depth_to_points(
            rs2::points output,
            const rs2_intrinsics& depth_intrinsics,
            const rs2::depth_frame& depth_frame,
            float depth_scale) override;
        void get_texture_map(
            rs2::points output,
            const float3* points,
            const unsigned int width,
            const unsigned int height,
            const rs2_intrinsics& other_intrinsics,
            const rs2_extrinsics& extr,
            float2* pixels_ptr) override;

        void pre_compute_x_y_map();
    };
}
