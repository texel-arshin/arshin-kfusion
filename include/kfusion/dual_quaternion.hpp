#pragma once
#include <kfusion/cuda/device_array.hpp>
#include <memory>
#include "kfusion/types.hpp"
#include "vector_functions.h"
struct DualQuaternion {
    union {
        struct {
            float4 rot, tr;
        };
        float data[8];
    };

    static const int PARAMS_LENGTH = 8;
    static const int ROTATION_PARAMS_LENGTH = 4;
    __host__ __device__ void zero_val() {
        rot = make_float4(0.,0.,0.,0.);
        tr = make_float4(0.,0.,0.,0.);
    }
//    //__kf_device__ DualQuaternion& operator*(float w);
//    __kf_device__ DualQuaternion& operator+=(const DualQuaternion &a);
};