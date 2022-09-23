#pragma once

#include <kfusion/types.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/cuda/imgproc.hpp>
#include "internal.hpp"
#include <iostream>
#include "vector_functions.h"

namespace kfusion
{
    template<typename D, typename S>
    inline D device_cast(const S& source)
    {
//        typename S::Scalar *ptr;
//        Eigen::Map<S>(ptr, source.rows(), source.cols()) = source;
        auto ptr = source.data();
        return *reinterpret_cast<const D*>(ptr);

    }

    template<>
    inline device::Aff3f device_cast<device::Aff3f, Affine3f>(const Affine3f& source)
    {
        device::Aff3f aff;
        Mat3f R = source.rotation();
        Vec3f t = source.translation();
        aff.R = device_cast<device::Mat3f>(R);
        aff.t = device_cast<device::Vec3f>(t);
        return aff;
    }
}
