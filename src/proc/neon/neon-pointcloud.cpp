// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//
//https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics

#include "../include/librealsense2/rs.hpp"
#include "../include/librealsense2/rsutil.h"

#include "proc/synthetic-stream.h"
#include "environment.h"
#include "proc/occlusion-filter.h"
#include "proc/neon/neon-pointcloud.h"
#include "option.h"
#include "environment.h"
#include "context.h"

#include <iostream>

#define __ARM_NEON 1
#ifdef __ARM_NEON

#include <arm_neon.h>

#define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))

#define _mm_shuffle_ps(a, b, imm8)                                   \
   __extension__({                                                        \
      float32x4_t ret;                                                   \
      ret = vmovq_n_f32(                                                 \
          vgetq_lane_f32(a, (imm8) & (0x3)));     \
      ret = vsetq_lane_f32(                                              \
          vgetq_lane_f32(a, ((imm8) >> 2) & 0x3), \
          ret, 1);                                                       \
      ret = vsetq_lane_f32(                                              \
          vgetq_lane_f32(b, ((imm8) >> 4) & 0x3), \
          ret, 2);                                                       \
      ret = vsetq_lane_f32(                                              \
          vgetq_lane_f32(b, ((imm8) >> 6) & 0x3), \
          ret, 3);                                                                    \
  })
#endif

/* "__has_builtin" can be used to query support for built-in functions
 * provided by gcc/clang and other compilers that support it.
 */
#ifndef __has_builtin /* GCC prior to 10 or non-clang compilers */
 /* Compatibility with gcc <= 9 */
#if __GNUC__ <= 9
#define __has_builtin(x) HAS##x
#define HAS__builtin_popcount 1
#define HAS__builtin_popcountll 1
#else
#define __has_builtin(x) 0
#endif
#endif

namespace librealsense
{
    pointcloud_neon::pointcloud_neon() : pointcloud("Pointcloud (NEON)") {}

#ifdef __ARM_NEON
    
    float32x4_t divide(float32x4_t a, float32x4_t b)
    {
#if defined(__aarch64__) && !SSE2NEON_PRECISE_DIV
        return vdivq_f32(a, b);
#else
        float32x4_t recip = vrecpeq_f32(b);
        recip = vmulq_f32(recip, vrecpsq_f32(recip, b));
#if SSE2NEON_PRECISE_DIV
        // Additional Netwon-Raphson iteration for accuracy
        recip = vmulq_f32(recip, vrecpsq_f32(recip, b));
#endif
        return vmulq_f32(a, recip);
#endif
    }

    int64x2_t shuffleMask(int64x2_t a, int64x2_t b)
    {
        int8x16_t tbl = vreinterpretq_s8_s64(a);   // input a
        uint8x16_t idx = vreinterpretq_u8_s64(b);  // input b
        uint8x16_t idx_masked = vandq_u8(idx, vdupq_n_u8(0x8F));  // avoid using meaningless bits
#if defined(__aarch64__)
        return vreinterpretq_s32_s8(vqtbl1q_s8(tbl, idx_masked));
#elif defined(__GNUC__)
        int8x16_t ret;
        // %e and %f represent the even and odd D registers
        // respectively.
        __asm__ __volatile__(
            "vtbl.8  %e[ret], {%e[tbl], %f[tbl]}, %e[idx]\n"
            "vtbl.8  %f[ret], {%e[tbl], %f[tbl]}, %f[idx]\n"
            : [ret] "=&w"(ret)
            : [tbl] "w"(tbl), [idx] "w"(idx_masked));
        return vreinterpretq_s64_s8(ret);
#else
    // use this line if testing on aarch64
        int8x8x2_t a_split = { vget_low_s8(tbl), vget_high_s8(tbl) };
        return vreinterpretq_s64_s8(
            vcombine_s8(vtbl2_s8(a_split, vget_low_u8(idx_masked)),
                vtbl2_s8(a_split, vget_high_u8(idx_masked))));
#endif
    }
#endif

    const float3* pointcloud_neon::depth_to_points(rs2::points output,
        const rs2_intrinsics& depth_intrinsics,
        const rs2::depth_frame& depth_frame,
        float depth_scale)
    {
#ifdef __ARM_NEON

        auto depth_image = (const uint16_t*)depth_frame.get_data();

        float* pre_compute_x = _pre_compute_map_x.data();
        float* pre_compute_y = _pre_compute_map_y.data();

        uint32_t size = depth_intrinsics.height * depth_intrinsics.width;

        auto point = (float*)output.get_vertices();

        //mask for shuffle
        const int8_t __attribute__((aligned(16))) data1[16] = { (int8_t)0, (int8_t)1, (int8_t)0xff, (int8_t)0xff,
                                                                (int8_t)2, (int8_t)3, (int8_t)0xff, (int8_t)0xff,
                                                                (int8_t)4, (int8_t)5, (int8_t)0xff, (int8_t)0xff,
                                                                (int8_t)6, (int8_t)7, (int8_t)0xff, (int8_t)0xff };

        const int8_t __attribute__((aligned(16))) data2[16] = { (int8_t)8, (int8_t)9, (int8_t)0xff, (int8_t)0xff,
                                                                (int8_t)10, (int8_t)11, (int8_t)0xff, (int8_t)0xff,
                                                                (int8_t)12, (int8_t)13, (int8_t)0xff, (int8_t)0xff,
                                                                (int8_t)14, (int8_t)15, (int8_t)0xff, (int8_t)0xff };

        const int64x2_t mask0 = vreinterpretq_s64_s8(vld1q_s8(data1));
        const int64x2_t mask1 = vreinterpretq_s64_s8(vld1q_s8(data2));

        auto scale = vdupq_n_f32(depth_scale);

        auto mapx = pre_compute_x;
        auto mapy = pre_compute_y;

        for (unsigned int i = 0; i < size; i += 8)
        {
            auto x0 = vld1q_f32(mapx + i);
            auto x1 = vld1q_f32(mapx + i + 4);

            auto y0 = vld1q_f32(mapy + i);
            auto y1 = vld1q_f32(mapy + i + 4);


            int64x2_t d = vreinterpretq_s64_s32(vld1q_s32((int32_t const*)(depth_image + i)));        //d7 d7 d6 d6 d5 d5 d4 d4 d3 d3 d2 d2 d1 d1 d0 d0

                                                     //split the depth pixel to 2 registers of 4 floats each
            int64x2_t d0 = shuffleMask(d, mask0);        // 00 00 d3 d3 00 00 d2 d2 00 00 d1 d1 00 00 d0 d0
            int64x2_t d1 = shuffleMask(d, mask1);        // 00 00 d7 d7 00 00 d6 d6 00 00 d5 d5 00 00 d4 d4

            float32x4_t depth0 = vcvtq_f32_s32(vreinterpretq_s32_s64(d0)); //convert int depth to float
            float32x4_t depth1 = vcvtq_f32_s32(vreinterpretq_s32_s64(d1)); //convert int depth to float

            depth0 = vmulq_f32(depth0, scale);
            depth1 = vmulq_f32(depth1, scale);

            auto p0x = vmulq_f32(depth0, x0);
            auto p0y = vmulq_f32(depth0, y0);

            auto p1x = vmulq_f32(depth1, x1);
            auto p1y = vmulq_f32(depth1, y1);

            //scattering of the x y z. If x_y0 etc are just setting up for xyz, why not just make it directly?
            //_mm_shuffle_ps(a, b, _MM_SHUFFLE(1, 0, 3, 2)) = b_1, b_0, a_3, a_2

            auto x_y0 = _mm_shuffle_ps(p0x, p0y, _MM_SHUFFLE(2, 0, 2, 0));      //x_y0 = p0y_2, p0y_0, p0x_2, p0x_0 
            auto z_x0 = _mm_shuffle_ps(depth0, p0x, _MM_SHUFFLE(3, 1, 2, 0));   //z_x0 = p0x_3, p0x_1, dep_2, dep_0
            auto y_z0 = _mm_shuffle_ps(p0y, depth0, _MM_SHUFFLE(3, 1, 3, 1));   //y_z0 = dep_3, dep_1, p0y_3, p0y_1

            auto xyz01 = _mm_shuffle_ps(x_y0, z_x0, _MM_SHUFFLE(2, 0, 2, 0));   //xyz1 = p0x_1, dep_0, p0y_0, p0x_0
            auto xyz02 = _mm_shuffle_ps(y_z0, x_y0, _MM_SHUFFLE(3, 1, 2, 0));   //xyz2 = p0y_2, p0x_2, dep_1, p0y_1
            auto xyz03 = _mm_shuffle_ps(z_x0, y_z0, _MM_SHUFFLE(3, 1, 3, 1));   //xyz3 = dep_3, p0y_3, p0x_3, dep_2

            //no idea which approach is faster, or if we should be doing set_lane instead to make the float32x4
            //auto xyz01 = float32x4_t{ vgetq_lane_f32(p0x, 1), vgetq_lane_f32(depth0, 0), vgetq_lane_f32(p0y, 0), vgetq_lane_f32(p0x, 0) };
            //auto xyz02 = float32x4_t{ vgetq_lane_f32(p0y, 2), vgetq_lane_f32(p0x, 2), vgetq_lane_f32(depth0, 1), vgetq_lane_f32(p0y, 1) };
            //auto xyz03 = float32x4_t{ vgetq_lane_f32(depth0, 3), vgetq_lane_f32(p0y, 3), vgetq_lane_f32(p0x, 3), vgetq_lane_f32(depth0, 2) };

            auto x_y1 = _mm_shuffle_ps(p1x, p1y, _MM_SHUFFLE(2, 0, 2, 0));
            auto z_x1 = _mm_shuffle_ps(depth1, p1x, _MM_SHUFFLE(3, 1, 2, 0));
            auto y_z1 = _mm_shuffle_ps(p1y, depth1, _MM_SHUFFLE(3, 1, 3, 1));

            auto xyz11 = _mm_shuffle_ps(x_y1, z_x1, _MM_SHUFFLE(2, 0, 2, 0));
            auto xyz12 = _mm_shuffle_ps(y_z1, x_y1, _MM_SHUFFLE(3, 1, 2, 0));
            auto xyz13 = _mm_shuffle_ps(z_x1, y_z1, _MM_SHUFFLE(3, 1, 3, 1));

            //auto xyz11 = float32x4_t{ vgetq_lane_f32(p1x, 1), vgetq_lane_f32(depth1, 0), vgetq_lane_f32(p1y, 0), vgetq_lane_f32(p1x, 0) };
            //auto xyz12 = float32x4_t{ vgetq_lane_f32(p1y, 2), vgetq_lane_f32(p1x, 2), vgetq_lane_f32(depth1, 1), vgetq_lane_f32(p1y, 1) };
            //auto xyz13 = float32x4_t{ vgetq_lane_f32(depth1, 3), vgetq_lane_f32(p1y, 3), vgetq_lane_f32(p1x, 3), vgetq_lane_f32(depth1, 2) };


            //store 8 points of x y z
#if __has_builtin(__builtin_nontemporal_store)
            __builtin_nontemporal_store(xyz01, &point[0]);
            __builtin_nontemporal_store(xyz02, &point[4]);
            __builtin_nontemporal_store(xyz03, &point[8]);
            __builtin_nontemporal_store(xyz11, &point[12]);
            __builtin_nontemporal_store(xyz12, &point[16]);
            __builtin_nontemporal_store(xyz13, &point[20]);
#else
            vst1q_f32(&point[0], xyz01);
            vst1q_f32(&point[4], xyz02);
            vst1q_f32(&point[8], xyz03);
            vst1q_f32(&point[12], xyz11);
            vst1q_f32(&point[16], xyz12);
            vst1q_f32(&point[20], xyz13);
#endif
            point += 24;
        }
#endif
        return (float3*)output.get_vertices();
    }

    void pointcloud_neon::get_texture_map(rs2::points output,
        const float3* points,
        const unsigned int width,
        const unsigned int height,
        const rs2_intrinsics& other_intrinsics,
        const rs2_extrinsics& extr,
        float2* pixels_ptr)
    {
        auto tex_ptr = (float2*)output.get_texture_coordinates();

#ifdef __ARM_NEON
        auto point = reinterpret_cast<const float*>(points);
        auto res = reinterpret_cast<float*>(tex_ptr);
        auto res1 = reinterpret_cast<float*>(pixels_ptr);

        float32x4_t r[9];
        float32x4_t t[3];
        float32x4_t c[5];

        for (int i = 0; i < 9; ++i)
        {
            r[i] = vdupq_n_f32(extr.rotation[i]);
        }
        for (int i = 0; i < 3; ++i)
        {
            t[i] = vdupq_n_f32(extr.translation[i]);
        }
        for (int i = 0; i < 5; ++i)
        {
            c[i] = vdupq_n_f32(other_intrinsics.coeffs[i]);
        }

        auto fx = vdupq_n_f32(other_intrinsics.fx);
        auto fy = vdupq_n_f32(other_intrinsics.fy);
        auto ppx = vdupq_n_f32(other_intrinsics.ppx);
        auto ppy = vdupq_n_f32(other_intrinsics.ppy);
        auto w = vdupq_n_f32(float(other_intrinsics.width));
        auto h = vdupq_n_f32(float(other_intrinsics.height));
        auto mask_inv_brown_conrady = vdupq_n_f32(RS2_DISTORTION_INVERSE_BROWN_CONRADY);
        auto zero = vdupq_n_f32(0);
        auto one = vdupq_n_f32(1);
        auto two = vdupq_n_f32(2);

        for (auto i = 0UL; i < height * width * 3; i += 12)
        {
            //load 4 points (x,y,z)
            auto xyz1 = vld1q_f32(point + i);
            auto xyz2 = vld1q_f32(point + i + 4);
            auto xyz3 = vld1q_f32(point + i + 8);

            //gather x,y,z
            auto yz = _mm_shuffle_ps(xyz1, xyz2, _MM_SHUFFLE(1, 0, 2, 1));
            auto xy = _mm_shuffle_ps(xyz2, xyz3, _MM_SHUFFLE(2, 1, 3, 2));

            auto x = _mm_shuffle_ps(xyz1, xy, _MM_SHUFFLE(2, 0, 3, 0));
            auto y = _mm_shuffle_ps(yz, xy, _MM_SHUFFLE(3, 1, 2, 0));
            auto z = _mm_shuffle_ps(yz, xyz3, _MM_SHUFFLE(3, 0, 3, 1));

            auto p_x = vaddq_f32(vmulq_f32(r[0], x), vaddq_f32(vmulq_f32(r[3], y), vaddq_f32(vmulq_f32(r[6], z), t[0])));
            auto p_y = vaddq_f32(vmulq_f32(r[1], x), vaddq_f32(vmulq_f32(r[4], y), vaddq_f32(vmulq_f32(r[7], z), t[1])));
            auto p_z = vaddq_f32(vmulq_f32(r[2], x), vaddq_f32(vmulq_f32(r[5], y), vaddq_f32(vmulq_f32(r[8], z), t[2])));

            p_x = divide(p_x, p_z);
            p_y = divide(p_y, p_z);

            // if(model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY)
            auto dist = vdupq_n_f32((float)other_intrinsics.model);

            auto r2 = vaddq_f32(vmulq_f32(p_x, p_x), vmulq_f32(p_y, p_y));
            auto r3 = vaddq_f32(vmulq_f32(c[1], vmulq_f32(r2, r2)), vmulq_f32(c[4], vmulq_f32(r2, vmulq_f32(r2, r2))));
            auto f = vaddq_f32(one, vaddq_f32(vmulq_f32(c[0], r2), r3));

            auto x_f = vmulq_f32(p_x, f);
            auto y_f = vmulq_f32(p_y, f);

            auto r4 = vmulq_f32(c[3], vaddq_f32(r2, vmulq_f32(two, vmulq_f32(x_f, x_f))));
            auto d_x = vaddq_f32(x_f, vaddq_f32(vmulq_f32(two, vmulq_f32(c[2], vmulq_f32(x_f, y_f))), r4));

            auto r5 = vmulq_f32(c[2], vaddq_f32(r2, vmulq_f32(two, vmulq_f32(y_f, y_f))));
            auto d_y = vaddq_f32(y_f, vaddq_f32(vmulq_f32(two, vmulq_f32(c[3], vmulq_f32(x_f, y_f))), r4));

            auto cmp = vreinterpretq_s32_u32(vceqq_f32(mask_inv_brown_conrady, dist));

            p_x = vreinterpretq_f32_s32(vorrq_s32(vandq_s32(cmp, vreinterpretq_s32_f32(d_x)), vbicq_s32(vreinterpretq_s32_f32(p_x), cmp))); //arg swaps on the bitwise clear
            p_y = vreinterpretq_f32_s32(vorrq_s32(vandq_s32(cmp, vreinterpretq_s32_f32(d_y)), vbicq_s32(vreinterpretq_s32_f32(p_y), cmp))); //arg swaps on the bitwise clear

            //TODO: add handle to RS2_DISTORTION_FTHETA

            //zero the x and y if z is zero
            cmp = vreinterpretq_s32_u32(vmvnq_u32(vceqq_f32(z, zero)));
            p_x = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(vaddq_f32(vmulq_f32(p_x, fx), ppx)), cmp)); //vreinterpretq_s32_f32 is cast, vcvtq_s32_f32 is convert w/ round
            p_y = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(vaddq_f32(vmulq_f32(p_y, fy), ppy)), cmp));

            //scattering of the x y before normalize and store in pixels_ptr
            auto xx_yy01 = _mm_shuffle_ps(p_x, p_y, _MM_SHUFFLE(2, 0, 2, 0));
            auto xx_yy23 = _mm_shuffle_ps(p_x, p_y, _MM_SHUFFLE(3, 1, 3, 1));

            auto xyxy1 = _mm_shuffle_ps(xx_yy01, xx_yy23, _MM_SHUFFLE(2, 0, 2, 0));
            auto xyxy2 = _mm_shuffle_ps(xx_yy01, xx_yy23, _MM_SHUFFLE(3, 1, 3, 1));

#if __has_builtin(__builtin_nontemporal_store)
            __builtin_nontemporal_store(xyxy1, res1);
            __builtin_nontemporal_store(xyxy2, res1+4);
#else
            vst1q_f32(res1, xyxy1);
            vst1q_f32(res1 + 4, xyxy2);
#endif
            res1 += 8;

            //normalize x and y
            p_x = divide(p_x, w);
            p_y = divide(p_y, h);

            //scattering of the x y after normalize and store in tex_ptr
            xx_yy01 = _mm_shuffle_ps(p_x, p_y, _MM_SHUFFLE(2, 0, 2, 0));
            xx_yy23 = _mm_shuffle_ps(p_x, p_y, _MM_SHUFFLE(3, 1, 3, 1));

            xyxy1 = _mm_shuffle_ps(xx_yy01, xx_yy23, _MM_SHUFFLE(2, 0, 2, 0));
            xyxy2 = _mm_shuffle_ps(xx_yy01, xx_yy23, _MM_SHUFFLE(3, 1, 3, 1));

#if __has_builtin(__builtin_nontemporal_store)
            __builtin_nontemporal_store(xyxy1, res);
            __builtin_nontemporal_store(xyxy2, res + 4);
#else
            vst1q_f32(res, xyxy1);
            vst1q_f32(res + 4, xyxy2);
#endif
            res += 8;
        }
#endif

    }
}
