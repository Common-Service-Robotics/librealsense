// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2019 Intel Corporation. All Rights Reserved.
//
// https://github.com/DLTcollab/sse2neon/blob/master/sse2neon.h
// https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics
//#ifdef __ARM_NEON

#include "neon-align.h"
#include "../include/librealsense2/hpp/rs_sensor.hpp"
#include "../include/librealsense2/hpp/rs_processing.hpp"
#include "../include/librealsense2/rsutil.h"

#include "core/video.h"
#include "proc/synthetic-stream.h"
#include "environment.h"
#include "stream.h"

/* Rounding mode macros. */
#define _ROUND_NEAREST 0x0000
#define _ROUND_DOWN 0x2000
#define _ROUND_UP 0x4000
#define _ROUND_TOWARD_ZERO 0x6000

#include <arm_neon.h>

/* Rounding functions require either Aarch64 instructions or libm failback */
#if !defined(__aarch64__)
#include <math.h>
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

using namespace librealsense;

template<int N> struct bytes { byte b[N]; };

bool is_special_resolution(const rs2_intrinsics& depth, const rs2_intrinsics& to)
{
    if ((depth.width == 640 && depth.height == 240 && to.width == 320 && to.height == 180) ||
        (depth.width == 640 && depth.height == 480 && to.width == 640 && to.height == 360))
        return true;
    return false;
}

template<rs2_distortion dist>
inline void distorte_x_y(const float32x4_t& x, const float32x4_t& y, float32x4_t* distorted_x, float32x4_t* distorted_y, const rs2_intrinsics& to)
{
    *distorted_x = x;
    *distorted_y = y;
}
template<>
inline void distorte_x_y<RS2_DISTORTION_MODIFIED_BROWN_CONRADY>(const float32x4_t& x, const float32x4_t& y, float32x4_t* distorted_x, float32x4_t* distorted_y, const rs2_intrinsics& to)
{
    float32x4_t c[5];
    auto one = vdupq_n_f32(1);
    auto two = vdupq_n_f32(2);

    for (int i = 0; i < 5; ++i)
    {
        c[i] = vdupq_n_f32(to.coeffs[i]);
    }
    auto r2_0 = vaddq_f32(vmulq_f32(x, x), vmulq_f32(y, y));
    auto r3_0 = vaddq_f32(vmulq_f32(c[1], vmulq_f32(r2_0, r2_0)), vmulq_f32(c[4], vmulq_f32(r2_0, vmulq_f32(r2_0, r2_0))));
    auto f_0 = vaddq_f32(one, vaddq_f32(vmulq_f32(c[0], r2_0), r3_0));

    auto x_f0 = vmulq_f32(x, f_0);
    auto y_f0 = vmulq_f32(y, f_0);

    auto r4_0 = vmulq_f32(c[3], vaddq_f32(r2_0, vmulq_f32(two, vmulq_f32(x_f0, x_f0))));
    auto d_x0 = vaddq_f32(x_f0, vaddq_f32(vmulq_f32(two, vmulq_f32(c[2], vmulq_f32(x_f0, y_f0))), r4_0));

    auto r5_0 = vmulq_f32(c[2], vaddq_f32(r2_0, vmulq_f32(two, vmulq_f32(y_f0, y_f0))));
    auto d_y0 = vaddq_f32(y_f0, vaddq_f32(vmulq_f32(two, vmulq_f32(c[3], vmulq_f32(x_f0, y_f0))), r4_0));

    *distorted_x = d_x0;
    *distorted_y = d_y0;
}

uint32_t GET_ROUNDING_MODE()
{
    //round nearest might be better
    return _ROUND_TOWARD_ZERO;
}

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

// Converts the four single-precision, floating-point values of a to signed
// 32-bit integer values.
//
//   r0 := (int) a0
//   r1 := (int) a1
//   r2 := (int) a2
//   r3 := (int) a3
//
// https://msdn.microsoft.com/en-us/library/vstudio/xdc42k5e(v=vs.100).aspx
// *NOTE*. The default rounding mode on SSE is 'round to even', which ARMv7-A
// does not support! It is supported on ARMv8-A however.
int32x4_t convertWithRound(float32x4_t a)
{
#if defined(__aarch64__)
    switch (GET_ROUNDING_MODE())
    {
    case _ROUND_NEAREST:
        return vcvtnq_s32_f32(a);
    case _ROUND_DOWN:
        return vcvtmq_s32_f32(a);
    case _ROUND_UP:
        return vcvtpq_s32_f32(a);
    default:  // _ROUND_TOWARD_ZERO
        return vcvtq_s32_f32(a);
    }
#else
    float* f = (float*)&a;
    switch (GET_ROUNDING_MODE())
    {
    case _ROUND_NEAREST: //for some reason rotates by 90 degrees?!?!
    {
        uint32x4_t signmask = vdupq_n_u32(0x80000000);
        float32x4_t half = vbslq_f32(signmask, a, vdupq_n_f32(0.5f)); /* +/- 0.5 */
        int32x4_t r_normal = vcvtq_s32_f32(vaddq_f32(a, half)); /* round to integer: [a + 0.5]*/
        int32x4_t r_trunc = vcvtq_s32_f32(a); /* truncate to integer: [a] */
        int32x4_t plusone = vreinterpretq_s32_u32(vshrq_n_u32(
            vreinterpretq_u32_s32(vnegq_s32(r_trunc)), 31)); /* 1 or 0 */
        int32x4_t r_even = vbicq_s32(vaddq_s32(r_trunc, plusone), vdupq_n_s32(1)); /* ([a] + {0,1}) & ~1 */
        float32x4_t delta = vsubq_f32(a, vcvtq_f32_s32(r_trunc)); /* compute delta: delta = (a - [a]) */
        uint32x4_t is_delta_half = vceqq_f32(delta, half); /* delta == +/- 0.5 */
        return vbslq_s32(is_delta_half, r_even, r_normal);
    }
    case _ROUND_DOWN:
        return vcvtq_s32_f32(float32x4_t{ floorf(f[3]), floorf(f[2]), floorf(f[1]), floorf(f[0]) });
    case _ROUND_UP:
        return vcvtq_s32_f32(float32x4_t{ ceilf(f[3]), ceilf(f[2]), ceilf(f[1]), ceilf(f[0]) });
    default:  // _ROUND_TOWARD_ZERO
        return int32x4_t{ (int32_t)f[3], (int32_t)f[2], (int32_t)f[1], (int32_t)f[0] };
    }
#endif
}


template<rs2_distortion dist>
inline void get_texture_map_neon(const uint16_t* depth,
    float depth_scale,
    const unsigned int size,
    const float* pre_compute_x, const float* pre_compute_y,
    byte* pixels_ptr_int,
    const rs2_intrinsics& to,
    const rs2_extrinsics& from_to_other)
{
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

    auto res = reinterpret_cast<int32x4_t*>(pixels_ptr_int);

    float32x4_t r[9];
    float32x4_t t[3];
    float32x4_t c[5];

    for (int i = 0; i < 9; ++i)
    {
        r[i] = vdupq_n_f32(from_to_other.rotation[i]);
    }
    for (int i = 0; i < 3; ++i)
    {
        t[i] = vdupq_n_f32(from_to_other.translation[i]);
    }
    for (int i = 0; i < 5; ++i)
    {
        c[i] = vdupq_n_f32(to.coeffs[i]);
    }
    auto zero = vdupq_n_f32(0);
    auto fx = vdupq_n_f32(to.fx);
    auto fy = vdupq_n_f32(to.fy);
    auto ppx = vdupq_n_f32(to.ppx);
    auto ppy = vdupq_n_f32(to.ppy);

    for (unsigned int i = 0; i < size; i += 8)
    {
        auto x0 = vld1q_f32(mapx + i);
        auto x1 = vld1q_f32(mapx + i + 4);

        auto y0 = vld1q_f32(mapy + i);
        auto y1 = vld1q_f32(mapy + i + 4);

        int64x2_t d = vreinterpretq_s64_s32(vld1q_s32((int32_t const*)(depth + i)));        //d7 d7 d6 d6 d5 d5 d4 d4 d3 d3 d2 d2 d1 d1 d0 d0

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

        auto p_x0 = vaddq_f32(vmulq_f32(r[0], p0x), vaddq_f32(vmulq_f32(r[3], p0y), vaddq_f32(vmulq_f32(r[6], depth0), t[0])));
        auto p_y0 = vaddq_f32(vmulq_f32(r[1], p0x), vaddq_f32(vmulq_f32(r[4], p0y), vaddq_f32(vmulq_f32(r[7], depth0), t[1])));
        auto p_z0 = vaddq_f32(vmulq_f32(r[2], p0x), vaddq_f32(vmulq_f32(r[5], p0y), vaddq_f32(vmulq_f32(r[8], depth0), t[2])));

        auto p_x1 = vaddq_f32(vmulq_f32(r[0], p1x), vaddq_f32(vmulq_f32(r[3], p1y), vaddq_f32(vmulq_f32(r[6], depth1), t[0])));
        auto p_y1 = vaddq_f32(vmulq_f32(r[1], p1x), vaddq_f32(vmulq_f32(r[4], p1y), vaddq_f32(vmulq_f32(r[7], depth1), t[1])));
        auto p_z1 = vaddq_f32(vmulq_f32(r[2], p1x), vaddq_f32(vmulq_f32(r[5], p1y), vaddq_f32(vmulq_f32(r[8], depth1), t[2])));

        p_x0 = divide(p_x0, p_z0);
        p_y0 = divide(p_y0, p_z0);

        p_x1 = divide(p_x1, p_z1);
        p_y1 = divide(p_y1, p_z1);

        distorte_x_y<dist>(p_x0, p_y0, &p_x0, &p_y0, to);
        distorte_x_y<dist>(p_x1, p_y1, &p_x1, &p_y1, to);

        //zero the x and y if z is zero
        auto cmp = vreinterpretq_s32_u32(vmvnq_u32(vceqq_f32(depth0, zero)));
        p_x0 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(vaddq_f32(vmulq_f32(p_x0, fx), ppx)), cmp)); //not sure if vreinterpretq_s32_f32 is better than vcvtq_f32_s32
        p_y0 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(vaddq_f32(vmulq_f32(p_y0, fy), ppy)), cmp));

        p_x1 = vaddq_f32(vmulq_f32(p_x1, fx), ppx);
        p_y1 = vaddq_f32(vmulq_f32(p_y1, fy), ppy);

        cmp = vreinterpretq_s32_u32(vmvnq_u32(vceqq_f32(depth0, zero)));
        auto half = vdupq_n_f32(0.5);
        auto u_round0 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(vaddq_f32(p_x0, half)), cmp));
        auto v_round0 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(vaddq_f32(p_y0, half)), cmp));

        auto uuvv1_0 = vcombine_f32(vget_low_f32(u_round0), vget_low_f32(v_round0)); //1, 0, 1, 0
        auto uuvv2_0 = vcombine_f32(vget_high_f32(u_round0), vget_high_f32(v_round0)); //3, 2, 3, 2

        //maybe this would be better as a table or some swaps
        auto res1_0 = float32x4_t{ vgetq_lane_f32(uuvv1_0, 3), vgetq_lane_f32(uuvv1_0, 1), vgetq_lane_f32(uuvv1_0, 2), vgetq_lane_f32(uuvv1_0, 0) }; //3, 1, 2, 0
        auto res2_0 = float32x4_t{ vgetq_lane_f32(uuvv2_0, 3), vgetq_lane_f32(uuvv2_0, 1), vgetq_lane_f32(uuvv2_0, 2), vgetq_lane_f32(uuvv2_0, 0) }; //3, 1, 2, 0

        auto res1_int0 = convertWithRound(res1_0);
        auto res2_int0 = convertWithRound(res2_0);


#if __has_builtin(__builtin_nontemporal_store)
        __builtin_nontemporal_store(res1_int0, &res[0]);
        __builtin_nontemporal_store(res2_int0, &res[1]);
#else
        vst1q_s32((int32_t*)&res[0], res1_int0);
        vst1q_s32((int32_t*)&res[1], res2_int0);
#endif
        res += 2;

        cmp = vreinterpretq_s32_u32(vmvnq_u32(vceqq_f32(depth1, zero)));
        auto u_round1 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(vaddq_f32(p_x1, half)), cmp));
        auto v_round1 = vreinterpretq_f32_s32(vandq_s32(vreinterpretq_s32_f32(vaddq_f32(p_y1, half)), cmp));

        auto uuvv1_1 = vcombine_f32(vget_low_f32(u_round1), vget_low_f32(v_round1)); //1, 0, 1, 0
        auto uuvv2_1 = vcombine_f32(vget_high_f32(u_round1), vget_high_f32(v_round1)); //3, 2, 3, 2

        auto res1 = float32x4_t{ vgetq_lane_f32(uuvv1_1, 3), vgetq_lane_f32(uuvv1_1, 1), vgetq_lane_f32(uuvv1_1, 2), vgetq_lane_f32(uuvv1_1, 0) }; //3, 1, 2, 0
        auto res2 = float32x4_t{ vgetq_lane_f32(uuvv2_1, 3), vgetq_lane_f32(uuvv2_1, 1), vgetq_lane_f32(uuvv2_1, 2), vgetq_lane_f32(uuvv2_1, 0) }; //3, 1, 2, 0

        auto res1_int1 = convertWithRound(res1);
        auto res2_int1 = convertWithRound(res2);

#if __has_builtin(__builtin_nontemporal_store)
        __builtin_nontemporal_store(res1_int1, &res[0]);
        __builtin_nontemporal_store(res2_int1, &res[1]);
#else
        vst1q_s32((int32_t*)&res[0], res1_int1);
        vst1q_s32((int32_t*)&res[1], res2_int1);
#endif
        res += 2;
    }
}

image_transform::image_transform(const rs2_intrinsics& from, float depth_scale)
    :_depth(from),
    _depth_scale(depth_scale),
    _pixel_top_left_int(from.width* from.height),
    _pixel_bottom_right_int(from.width* from.height)
{
}

void image_transform::pre_compute_x_y_map_corners()
{
    pre_compute_x_y_map(_pre_compute_map_x_top_left, _pre_compute_map_y_top_left, -0.5f);
    pre_compute_x_y_map(_pre_compute_map_x_bottom_right, _pre_compute_map_y_bottom_right, 0.5f);
}

void image_transform::pre_compute_x_y_map(std::vector<float>& pre_compute_map_x,
    std::vector<float>& pre_compute_map_y,
    float offset)
{
    pre_compute_map_x.resize(_depth.width * _depth.height);
    pre_compute_map_y.resize(_depth.width * _depth.height);

    for (int h = 0; h < _depth.height; ++h)
    {
        for (int w = 0; w < _depth.width; ++w)
        {
            const float pixel[] = { (float)w + offset, (float)h + offset };

            float x = (pixel[0] - _depth.ppx) / _depth.fx;
            float y = (pixel[1] - _depth.ppy) / _depth.fy;

            if (_depth.model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
            {
                float r2 = x * x + y * y;
                float f = 1 + _depth.coeffs[0] * r2 + _depth.coeffs[1] * r2 * r2 + _depth.coeffs[4] * r2 * r2 * r2;
                float ux = x * f + 2 * _depth.coeffs[2] * x * y + _depth.coeffs[3] * (r2 + 2 * x * x);
                float uy = y * f + 2 * _depth.coeffs[3] * x * y + _depth.coeffs[2] * (r2 + 2 * y * y);
                x = ux;
                y = uy;
            }

            pre_compute_map_x[h * _depth.width + w] = x;
            pre_compute_map_y[h * _depth.width + w] = y;
        }
    }
}

void image_transform::align_depth_to_other(const uint16_t* z_pixels, uint16_t* dest, int bpp, const rs2_intrinsics& depth, const rs2_intrinsics& to,
    const rs2_extrinsics& from_to_other)
{
    switch (to.model)
    {
    case RS2_DISTORTION_MODIFIED_BROWN_CONRADY:
        align_depth_to_other_neon<RS2_DISTORTION_MODIFIED_BROWN_CONRADY>(z_pixels, dest, depth, to, from_to_other);
        break;
    default:
        align_depth_to_other_neon(z_pixels, dest, depth, to, from_to_other);
        break;
    }
}

inline void image_transform::move_depth_to_other(const uint16_t* z_pixels, uint16_t* dest, const rs2_intrinsics& to,
    const std::vector<librealsense::int2>& pixel_top_left_int,
    const std::vector<librealsense::int2>& pixel_bottom_right_int)
{
    for (int y = 0; y < _depth.height; ++y)
    {
        for (int x = 0; x < _depth.width; ++x)
        {
            auto depth_pixel_index = y * _depth.width + x;
            // Skip over depth pixels with the value of zero, we have no depth data so we will not write anything into our aligned images
            if (z_pixels[depth_pixel_index])
            {
                for (int other_y = pixel_top_left_int[depth_pixel_index].y; other_y <= pixel_bottom_right_int[depth_pixel_index].y; ++other_y)
                {
                    for (int other_x = pixel_top_left_int[depth_pixel_index].x; other_x <= pixel_bottom_right_int[depth_pixel_index].x; ++other_x)
                    {
                        if (other_x < 0 || other_y < 0 || other_x >= to.width || other_y >= to.height)
                            continue;
                        auto other_ind = other_y * to.width + other_x;

                        dest[other_ind] = dest[other_ind] ? std::min(dest[other_ind], z_pixels[depth_pixel_index]) : z_pixels[depth_pixel_index];
                    }
                }
            }
        }
    }
}

void image_transform::align_other_to_depth(const uint16_t* z_pixels, const byte* source, byte* dest, int bpp, const rs2_intrinsics& to,
    const rs2_extrinsics& from_to_other)
{
    switch (to.model)
    {
    case RS2_DISTORTION_MODIFIED_BROWN_CONRADY:
    case RS2_DISTORTION_INVERSE_BROWN_CONRADY:
        align_other_to_depth_neon<RS2_DISTORTION_MODIFIED_BROWN_CONRADY>(z_pixels, source, dest, bpp, to, from_to_other);
        break;
    default:
        align_other_to_depth_neon(z_pixels, source, dest, bpp, to, from_to_other);
        break;
    }
}


template<rs2_distortion dist>
inline void image_transform::align_depth_to_other_neon(const uint16_t* z_pixels, uint16_t* dest, const rs2_intrinsics& depth, const rs2_intrinsics& to,
    const rs2_extrinsics& from_to_other)
{
    get_texture_map_neon<dist>(z_pixels, _depth_scale, _depth.height * _depth.width, _pre_compute_map_x_top_left.data(),
        _pre_compute_map_y_top_left.data(), (byte*)_pixel_top_left_int.data(), to, from_to_other);

    float fov[2];
    rs2_fov(&depth, fov);
    float2 pixels_per_angle_depth = { (float)depth.width / fov[0], (float)depth.height / fov[1] };

    rs2_fov(&to, fov);
    float2 pixels_per_angle_target = { (float)to.width / fov[0], (float)to.height / fov[1] };

    if (pixels_per_angle_depth.x < pixels_per_angle_target.x || pixels_per_angle_depth.y < pixels_per_angle_target.y || is_special_resolution(depth, to))
    {
        get_texture_map_neon<dist>(z_pixels, _depth_scale, _depth.height * _depth.width, _pre_compute_map_x_bottom_right.data(),
            _pre_compute_map_y_bottom_right.data(), (byte*)_pixel_bottom_right_int.data(), to, from_to_other);

        move_depth_to_other(z_pixels, dest, to, _pixel_top_left_int, _pixel_bottom_right_int);
    }
    else
    {
        move_depth_to_other(z_pixels, dest, to, _pixel_top_left_int, _pixel_top_left_int);
    }

}

template<rs2_distortion dist>
inline void image_transform::align_other_to_depth_neon(const uint16_t* z_pixels, const byte* source, byte* dest, int bpp, const rs2_intrinsics& to,
    const rs2_extrinsics& from_to_other)
{
    get_texture_map_neon<dist>(z_pixels, _depth_scale, _depth.height * _depth.width, _pre_compute_map_x_top_left.data(),
        _pre_compute_map_y_top_left.data(), (byte*)_pixel_top_left_int.data(), to, from_to_other);

    std::vector<int2>& bottom_right = _pixel_top_left_int;
    if (to.height < _depth.height && to.width < _depth.width)
    {
        get_texture_map_neon<dist>(z_pixels, _depth_scale, _depth.height * _depth.width, _pre_compute_map_x_bottom_right.data(),
            _pre_compute_map_y_bottom_right.data(), (byte*)_pixel_bottom_right_int.data(), to, from_to_other);

        bottom_right = _pixel_bottom_right_int;
    }

    switch (bpp)
    {
    case 1:
        move_other_to_depth(z_pixels, reinterpret_cast<const bytes<1>*>(source), reinterpret_cast<bytes<1>*>(dest), to,
            _pixel_top_left_int, bottom_right);
        break;
    case 2:
        move_other_to_depth(z_pixels, reinterpret_cast<const bytes<2>*>(source), reinterpret_cast<bytes<2>*>(dest), to,
            _pixel_top_left_int, bottom_right);
        break;
    case 3:
        move_other_to_depth(z_pixels, reinterpret_cast<const bytes<3>*>(source), reinterpret_cast<bytes<3>*>(dest), to,
            _pixel_top_left_int, bottom_right);
        break;
    case 4:
        move_other_to_depth(z_pixels, reinterpret_cast<const bytes<4>*>(source), reinterpret_cast<bytes<4>*>(dest), to,
            _pixel_top_left_int, bottom_right);
        break;
    default:
        break;
    }
}

template<class T >
void image_transform::move_other_to_depth(const uint16_t* z_pixels,
    const T* source,
    T* dest, const rs2_intrinsics& to,
    const std::vector<librealsense::int2>& pixel_top_left_int,
    const std::vector<librealsense::int2>& pixel_bottom_right_int)
{
    // Iterate over the pixels of the depth image
    for (int y = 0; y < _depth.height; ++y)
    {
        for (int x = 0; x < _depth.width; ++x)
        {
            auto depth_pixel_index = y * _depth.width + x;
            // Skip over depth pixels with the value of zero, we have no depth data so we will not write anything into our aligned images
            if (z_pixels[depth_pixel_index])
            {
                for (int other_y = pixel_top_left_int[depth_pixel_index].y; other_y <= pixel_bottom_right_int[depth_pixel_index].y; ++other_y)
                {
                    for (int other_x = pixel_top_left_int[depth_pixel_index].x; other_x <= pixel_bottom_right_int[depth_pixel_index].x; ++other_x)
                    {
                        if (other_x < 0 || other_y < 0 || other_x >= to.width || other_y >= to.height)
                            continue;
                        auto other_ind = other_y * to.width + other_x;

                        dest[depth_pixel_index] = source[other_ind];
                    }
                }
            }
        }
    }
}

void align_neon::reset_cache(rs2_stream from, rs2_stream to)
{
    _stream_transform = nullptr;
}

void align_neon::align_z_to_other(rs2::video_frame& aligned, const rs2::video_frame& depth, const rs2::video_stream_profile& other_profile, float z_scale)
{
    byte* aligned_data = reinterpret_cast<byte*>(const_cast<void*>(aligned.get_data()));
    auto aligned_profile = aligned.get_profile().as<rs2::video_stream_profile>();
    memset(aligned_data, 0, aligned_profile.height() * aligned_profile.width() * aligned.get_bytes_per_pixel());

    auto depth_profile = depth.get_profile().as<rs2::video_stream_profile>();

    auto z_intrin = depth_profile.get_intrinsics();
    auto other_intrin = other_profile.get_intrinsics();
    auto z_to_other = depth_profile.get_extrinsics_to(other_profile);

    auto z_pixels = reinterpret_cast<const uint16_t*>(depth.get_data());

    if (_stream_transform == nullptr)
    {
        _stream_transform = std::make_shared<image_transform>(z_intrin, z_scale);
        _stream_transform->pre_compute_x_y_map_corners();
    }
    _stream_transform->align_depth_to_other(z_pixels, reinterpret_cast<uint16_t*>(aligned_data), 2, z_intrin, other_intrin, z_to_other);
}

void align_neon::align_other_to_z(rs2::video_frame& aligned, const rs2::video_frame& depth, const rs2::video_frame& other, float z_scale)
{
    byte* aligned_data = reinterpret_cast<byte*>(const_cast<void*>(aligned.get_data()));
    auto aligned_profile = aligned.get_profile().as<rs2::video_stream_profile>();
    memset(aligned_data, 0, aligned_profile.height() * aligned_profile.width() * aligned.get_bytes_per_pixel());

    auto depth_profile = depth.get_profile().as<rs2::video_stream_profile>();
    auto other_profile = other.get_profile().as<rs2::video_stream_profile>();

    auto z_intrin = depth_profile.get_intrinsics();
    auto other_intrin = other_profile.get_intrinsics();
    auto z_to_other = depth_profile.get_extrinsics_to(other_profile);

    auto z_pixels = reinterpret_cast<const uint16_t*>(depth.get_data());
    auto other_pixels = reinterpret_cast<const byte*>(other.get_data());

    if (_stream_transform == nullptr)
    {
        _stream_transform = std::make_shared<image_transform>(z_intrin, z_scale);
        _stream_transform->pre_compute_x_y_map_corners();
    }

    _stream_transform->align_other_to_depth(z_pixels, other_pixels, aligned_data, other.get_bytes_per_pixel(), other_intrin, z_to_other);
}
#endif
