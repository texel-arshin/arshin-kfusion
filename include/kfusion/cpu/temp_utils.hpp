#pragma once
#include "cuda_runtime.h"
#include <cmath>

namespace cpu_warp {

    float3 operator+(const float3 &v1, const float3 &v2) {
        return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
    }

    float3 operator-(const float3 &v1, const float3 &v2) {
        return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
    }

    float4 &operator+=(float4 &v1, const float4 &v2) {
        v1.x += v2.x;
        v1.y += v2.y;
        v1.z += v2.z;
        v1.w += v2.w;
        return v1;
    }

    float3 &operator+=(float3 &v1, const float3 &v2) {
        v1.x += v2.x;
        v1.y += v2.y;
        v1.z += v2.z;
        return v1;
    }

    float dot(const float3 &v1, const float3 &v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    float dot(const float4 &v1, const float4 &v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
    }

    float rsqrtf(const float x) {
        return 1. / std::sqrt(x);
    }

    int __float2int_rd(float x) {
        return (int) x;
    }
}