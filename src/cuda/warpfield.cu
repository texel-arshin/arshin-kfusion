#include <kfusion/cuda/device_array.hpp>
#include "safe_call.hpp"
#include "kfusion/cuda/knn.hpp"
#include "kfusion/cuda/warpfield.hpp"
#include "device.hpp"
#include <stdio.h>

using namespace kfusion::device;
__kf_device__ DualQuaternion & operator+=(DualQuaternion &a, const DualQuaternion &b) {
    a.rot += b.rot;
    a.tr += b.tr;
    return a;
}

namespace warp
{


    constexpr float MINIMUM_SQ_NORM = 1e-10;

__kf_device__ void pstatus(DualQuaternion *params)
{
    for (int i=0; i<7; i++) {
        printf("[%d] Curr param: %f %f %f %f %f %f %f %f\n", i, params[i].rot.x, params[i].rot.y, params[i].rot.z, params[i].rot.w,
               params[i].tr.x, params[i].tr.y, params[i].tr.z, params[i].tr.w);
        printf("DATA [%d] Curr param: ", i);
        for (int j=0; j<8; j++)
        {
            printf("%f ", params[i].data[j]);
        }
        printf("\n");
    }
}

__kf_device__ DualQuaternion operator*(const DualQuaternion x, const float w)
{
    DualQuaternion a;
    a.rot = make_float4(x.rot.x*w,x.rot.y*w,x.rot.z*w,x.rot.w*w);
    a.tr = make_float4(x.tr.x*w,x.tr.y*w,x.tr.z*w,x.tr.w*w);
    return a;
}

__kf_device__
float compute_omega_w(float3 vertex, float3 node, float weigth)
{
    float3 vec = vertex - node;
    float sqdist = dot(vec,vec);
    float sqweight = weigth*weigth;
    if (sqdist>9*sqweight)
    {
        return 0;
    }
    return expf(-sqdist/(2*sqweight));
}

__kf_device__
float3 apply_dq(const DualQuaternion c, const float3 vertex)
{
    float3 tvec, warped_vertex;
    tvec.x = 2*(-c.tr.w*c.rot.x + c.tr.x*c.rot.w - c.tr.y*c.rot.z + c.tr.z*c.rot.y);
    tvec.y = 2*(-c.tr.w*c.rot.y + c.tr.x*c.rot.z + c.tr.y*c.rot.w - c.tr.z*c.rot.x);
    tvec.z = 2*(-c.tr.w*c.rot.z - c.tr.x*c.rot.y + c.tr.y*c.rot.x + c.tr.z*c.rot.w);

    warped_vertex.x =
    (1-2*c.rot.y*c.rot.y-2*c.rot.z*c.rot.z)*vertex.x+
    2*(c.rot.x*c.rot.y-c.rot.w*c.rot.z)*vertex.y+
    2*(c.rot.x*c.rot.z+c.rot.w*c.rot.y)*vertex.z;
    warped_vertex.y =
    2*(c.rot.x*c.rot.y+c.rot.w*c.rot.z)*vertex.x+
    (1-2*c.rot.x*c.rot.x-2*c.rot.z*c.rot.z)*vertex.y+
    2*(c.rot.y*c.rot.z-c.rot.w*c.rot.x)*vertex.z;
    warped_vertex.z =
    2*(c.rot.x*c.rot.z-c.rot.w*c.rot.y)*vertex.x+
    2*(c.rot.y*c.rot.z+c.rot.w*c.rot.x)*vertex.y+
    (1-2*c.rot.x*c.rot.x-2*c.rot.y*c.rot.y)*vertex.z;
    warped_vertex += tvec;
    return warped_vertex;
}

__kf_device__
float3 pretransform_vertex(const DualQuaternion c, const float3 vertex)
{
    float3 tvec, warped_vertex;
    tvec.x = 2*(-c.tr.w*c.rot.x + c.tr.x*c.rot.w - c.tr.y*c.rot.z + c.tr.z*c.rot.y);
    tvec.y = 2*(-c.tr.w*c.rot.y + c.tr.x*c.rot.z + c.tr.y*c.rot.w - c.tr.z*c.rot.x);
    tvec.z = 2*(-c.tr.w*c.rot.z - c.tr.x*c.rot.y + c.tr.y*c.rot.x + c.tr.z*c.rot.w);

    warped_vertex.x =
            (0-2*c.rot.y*c.rot.y-2*c.rot.z*c.rot.z)*vertex.x+
            2*(c.rot.x*c.rot.y-c.rot.w*c.rot.z)*vertex.y+
            2*(c.rot.x*c.rot.z+c.rot.w*c.rot.y)*vertex.z;
    warped_vertex.y =
            2*(c.rot.x*c.rot.y+c.rot.w*c.rot.z)*vertex.x+
            (0-2*c.rot.x*c.rot.x-2*c.rot.z*c.rot.z)*vertex.y+
            2*(c.rot.y*c.rot.z-c.rot.w*c.rot.x)*vertex.z;
    warped_vertex.z =
            2*(c.rot.x*c.rot.z-c.rot.w*c.rot.y)*vertex.x+
            2*(c.rot.y*c.rot.z+c.rot.w*c.rot.x)*vertex.y+
            (0-2*c.rot.x*c.rot.x-2*c.rot.y*c.rot.y)*vertex.z;
    warped_vertex += tvec;
    return warped_vertex;
}

__kf_device__
        float3 pretransform_normal(const DualQuaternion c, const float3 normal)
{
    float3 warped_normal;
    warped_normal.x =
            (0-2*c.rot.y*c.rot.y-2*c.rot.z*c.rot.z)*normal.x+
            2*(c.rot.x*c.rot.y-c.rot.w*c.rot.z)*normal.y+
            2*(c.rot.x*c.rot.z+c.rot.w*c.rot.y)*normal.z;
    warped_normal.y =
            2*(c.rot.x*c.rot.y+c.rot.w*c.rot.z)*normal.x+
            (0-2*c.rot.x*c.rot.x-2*c.rot.z*c.rot.z)*normal.y+
            2*(c.rot.y*c.rot.z-c.rot.w*c.rot.x)*normal.z;
    warped_normal.z =
            2*(c.rot.x*c.rot.z-c.rot.w*c.rot.y)*normal.x+
            2*(c.rot.y*c.rot.z+c.rot.w*c.rot.x)*normal.y+
            (0-2*c.rot.x*c.rot.x-2*c.rot.y*c.rot.y)*normal.z;
    return warped_normal;
}

__kf_device__
float3 apply_dq_norm(const DualQuaternion c, const float3 normal)
{
    float3 warped_normal;
    warped_normal.x =
    (1-2*c.rot.y*c.rot.y-2*c.rot.z*c.rot.z)*normal.x+
    2*(c.rot.x*c.rot.y-c.rot.w*c.rot.z)*normal.y+
    2*(c.rot.x*c.rot.z+c.rot.w*c.rot.y)*normal.z;
    warped_normal.y =
    2*(c.rot.x*c.rot.y+c.rot.w*c.rot.z)*normal.x+
    (1-2*c.rot.x*c.rot.x-2*c.rot.z*c.rot.z)*normal.y+
    2*(c.rot.y*c.rot.z-c.rot.w*c.rot.x)*normal.z;
    warped_normal.z =
    2*(c.rot.x*c.rot.z-c.rot.w*c.rot.y)*normal.x+
    2*(c.rot.y*c.rot.z+c.rot.w*c.rot.x)*normal.y+
    (1-2*c.rot.x*c.rot.x-2*c.rot.y*c.rot.y)*normal.z;
    return warped_normal;
}

__kf_device__ DualQuaternion compute_dqb(const WarpField wf, const float3 vertex)
{
    int3 knn_field_coords = make_int3(
            __float2int_rd(vertex.x/wf.knn_field_gpu.field_cell_size),
            __float2int_rd(vertex.y/wf.knn_field_gpu.field_cell_size),
            __float2int_rd(vertex.z/wf.knn_field_gpu.field_cell_size));
    //searching NN
    knn::DiscreteBruteFieldKNN::elem_type *knn_ptr = knn::get_knn_field_coords(wf.knn_field_gpu, knn_field_coords);
    //weighting & summing
    DualQuaternion bt, ct;
    bt.zero_val();
    for (int i = 0; (i < knn::DiscreteBruteFieldKNN::k) && (knn_ptr[i]!=knn::DiscreteBruteFieldKNN::UNDEFINED_OFFSET); i++) {
        int curr_offset = (int)knn_ptr[i];
        float3 curr_node_coords = make_float3(wf.warp_coords[3*curr_offset], wf.warp_coords[3*curr_offset + 1], wf.warp_coords[3*curr_offset + 2]);
        float w = compute_omega_w(vertex, curr_node_coords, wf.warp_weights[curr_offset]);
        bt += (wf.warp_params[curr_offset] * w);
    }
    //normalize
    float bt_norm = dot(bt.rot, bt.rot);
    if (bt_norm<MINIMUM_SQ_NORM)
    {
        bt.zero_val();
        bt.rot.w = 1;
        return bt;
    }
    ct = bt*rsqrtf(bt_norm);
    return ct;
}

__kf_device__ bool is_unsupported(const WarpField wf, const float3 vertex)
{
    int3 knn_field_coords = make_int3(
            __float2int_rd(vertex.x/wf.knn_field_gpu.field_cell_size),
            __float2int_rd(vertex.y/wf.knn_field_gpu.field_cell_size),
            __float2int_rd(vertex.z/wf.knn_field_gpu.field_cell_size));
    //searching NN
    knn::DiscreteBruteFieldKNN::elem_type *knn_ptr = knn::get_knn_field_coords(wf.knn_field_gpu, knn_field_coords);
    for (int i = 0; (i < knn::DiscreteBruteFieldKNN::k) && (knn_ptr[i]!=knn::DiscreteBruteFieldKNN::UNDEFINED_OFFSET); i++) {
        int curr_offset = (int)knn_ptr[i];
        float3 curr_node_coords = make_float3(wf.warp_coords[3*curr_offset], wf.warp_coords[3*curr_offset + 1], wf.warp_coords[3*curr_offset + 2]);
        float3 vec = vertex - curr_node_coords;
        float sqdist = dot(vec,vec);
        float w = wf.warp_weights[curr_offset];
        if (sqdist/(w*w)<1)
        {
            return false;
        }
    }
    return true;
}

__device__ float4 warp_vertex_nograd(const WarpField &wf, const float3 vertex) {
    int3 knn_field_coords = make_int3(
            __float2int_rd(vertex.x/wf.knn_field_gpu.field_cell_size),
            __float2int_rd(vertex.y/wf.knn_field_gpu.field_cell_size),
            __float2int_rd(vertex.z/wf.knn_field_gpu.field_cell_size));
    //searching NN
    knn::DiscreteBruteFieldKNN::elem_type *knn_ptr = knn::get_knn_field_coords(wf.knn_field_gpu, knn_field_coords);
    //weighting & summing
    DualQuaternion bt, ct;
    float wsum = 0;
    bt.zero_val();
    for (int i = 0; i < knn::DiscreteBruteFieldKNN::k; i++) {
        int curr_offset = (int)knn_ptr[i];
        float3 curr_node_coords = make_float3(wf.warp_coords[3*curr_offset], wf.warp_coords[3*curr_offset + 1], wf.warp_coords[3*curr_offset + 2]);
        float w = compute_omega_w(vertex, curr_node_coords, wf.warp_weights[curr_offset]);
        bt += (wf.warp_params[curr_offset] * w);
        wsum += w;
    }
    //pstatus(warp_params);
    //normalize
    float bt_norm = dot(bt.rot, bt.rot);
    if (bt_norm<MINIMUM_SQ_NORM)
    {
        return make_float4(vertex.x, vertex.y, vertex.z, 0.f);
    }
    wsum/=knn::DiscreteBruteFieldKNN::k;
    ct = bt*rsqrtf(bt_norm);

    //warp vertex
    float3 resvertex = apply_dq(ct, vertex);
    return make_float4(resvertex.x, resvertex.y, resvertex.z, wsum);
}

__kf_device__ void warp_nograd_nowsum(const WarpField &wf, const float3 vertex, const float3 normal, float3 &warped_vertex, float3 &warped_normal) {
    int3 knn_field_coords = make_int3(
            __float2int_rd(vertex.x/wf.knn_field_gpu.field_cell_size),
            __float2int_rd(vertex.y/wf.knn_field_gpu.field_cell_size),
            __float2int_rd(vertex.z/wf.knn_field_gpu.field_cell_size));
    //searching NN
    knn::DiscreteBruteFieldKNN::elem_type *knn_ptr = knn::get_knn_field_coords(wf.knn_field_gpu, knn_field_coords);
    //weighting & summing
    DualQuaternion bt, ct;
    bt.zero_val();
    for (int i = 0; i < knn::DiscreteBruteFieldKNN::k; i++) {
        int curr_offset = (int)knn_ptr[i];
        float3 curr_node_coords = make_float3(wf.warp_coords[3*curr_offset], wf.warp_coords[3*curr_offset + 1], wf.warp_coords[3*curr_offset + 2]);
        float w = compute_omega_w(vertex, curr_node_coords, wf.warp_weights[curr_offset]);
        bt += (wf.warp_params[curr_offset] * w);
    }
    float bt_norm = dot(bt.rot, bt.rot);
    if (bt_norm<MINIMUM_SQ_NORM)
    {
        warped_vertex = vertex;
        warped_normal = normal;
    }
    else {
        ct = bt * rsqrtf(bt_norm);

        //warp vertex and normal
        warped_vertex = apply_dq(ct, vertex);
        warped_normal = apply_dq_norm(ct, normal);
    }
}

__kf_device__
float3 WarpField::warp_vertex(WarpFieldCache &wfc, float3 vertex) const {
    wfc.last_V = vertex;
    int3 knn_field_coords = make_int3(
            __float2int_rd(vertex.x/knn_field_gpu.field_cell_size),
            __float2int_rd(vertex.y/knn_field_gpu.field_cell_size),
            __float2int_rd(vertex.z/knn_field_gpu.field_cell_size));
    //searching NN
    wfc.V_knn = knn::get_knn_field_coords(knn_field_gpu, knn_field_coords);
    //weighting & summing
    DualQuaternion b;
    b.zero_val();
    for (int i = 0; i < knn::DiscreteBruteFieldKNN::k; i++) {
        int curr_offset = (int)wfc.V_knn[i];
        float3 curr_node_coords = make_float3(warp_coords[3*curr_offset], warp_coords[3*curr_offset + 1], warp_coords[3*curr_offset + 2]);
        float w = compute_omega_w(vertex, curr_node_coords, warp_weights[curr_offset]);
        wfc.warp_omega_weights[i] = w;
        b += (warp_params[curr_offset] * w);
    }
    float b_norm_sq = dot(b.rot, b.rot);
    if (b_norm_sq<MINIMUM_SQ_NORM)
    {
        wfc.warp_applied = false;
        wfc.b_inv_norm=0;
        wfc.b_inv_norm_sq=0;
        return vertex;
    }
    wfc.b_inv_norm = rsqrtf(b_norm_sq);
    wfc.b_inv_norm_sq = wfc.b_inv_norm * wfc.b_inv_norm;

//    wfc.c = b * wfc.b_inv_norm;
    wfc.warp_applied = true;
    //warp vertex
    wfc.b = b;
    float3 pV = pretransform_vertex(wfc.b, vertex);
    wfc.pretransformed_V[0] = pV.x;
    wfc.pretransformed_V[1] = pV.y;
    wfc.pretransformed_V[2] = pV.z;
    return make_float3(wfc.b_inv_norm_sq*pV.x, wfc.b_inv_norm_sq*pV.y, wfc.b_inv_norm_sq*pV.z)+vertex;


}

__kf_device__
float3 WarpField::warp_normal(WarpFieldCache &wfc, float3 normal) const { //call ONLY after warp_vertex
    wfc.last_N = normal;
    float3 pN = pretransform_normal(wfc.b, normal);
    wfc.pretransformed_N[0] = pN.x;
    wfc.pretransformed_N[1] = pN.y;
    wfc.pretransformed_N[2] = pN.z;
    return make_float3(wfc.b_inv_norm_sq*pN.x,wfc.b_inv_norm_sq*pN.y,wfc.b_inv_norm_sq*pN.z)+normal;
}

__kf_device__
void WarpField::compute_dpN_wrt_db(WarpFieldCache &wfc) const
{
    float3 &n = wfc.last_N;
    DualQuaternion &c = wfc.b;
    wfc.dN_wrt_db[0].rot.x = 2*(c.rot.y*n.y+c.rot.z*n.z);
    wfc.dN_wrt_db[0].rot.y = 2*(-2*c.rot.y*n.x+c.rot.x*n.y+c.rot.w*n.z);
    wfc.dN_wrt_db[0].rot.z = 2*(-2*c.rot.z*n.x-c.rot.w*n.y+c.rot.x*n.z);
    wfc.dN_wrt_db[0].rot.w = 2*(-c.rot.z*n.y+c.rot.y*n.z);

    wfc.dN_wrt_db[1].rot.x = 2*(c.rot.y*n.x-2*c.rot.x*n.y-c.rot.w*n.z);
    wfc.dN_wrt_db[1].rot.y = 2*(c.rot.x*n.x+c.rot.z*n.z);
    wfc.dN_wrt_db[1].rot.z = 2*(c.rot.w*n.x-2*c.rot.z*n.y+c.rot.y*n.z);
    wfc.dN_wrt_db[1].rot.w = 2*(c.rot.z*n.x-c.rot.x*n.z);

    wfc.dN_wrt_db[2].rot.x = 2*(c.rot.z*n.x+c.rot.w*n.y-2*c.rot.x*n.z);
    wfc.dN_wrt_db[2].rot.y = 2*(-c.rot.w*n.x+c.rot.z*n.y-2*c.rot.y*n.z);
    wfc.dN_wrt_db[2].rot.z = 2*(c.rot.x*n.x + c.rot.y*n.y);
    wfc.dN_wrt_db[2].rot.w = 2*(-c.rot.y*n.x+c.rot.x*n.y);


    for (int i=0; i<3; i++) {
        wfc.dN_wrt_db[i].tr = make_float4(0.,0.,0.,0.);
    }
}

__kf_device__
void WarpField::compute_dpV_wrt_db(WarpFieldCache &wfc) const
{
    float3 &v = wfc.last_V;
    DualQuaternion &c = wfc.b;

    wfc.dV_wrt_db[0].rot.x = 2*(c.rot.y*v.y+c.rot.z*v.z)-2*c.tr.w;
    wfc.dV_wrt_db[0].rot.y = 2*(-2*c.rot.y*v.x+c.rot.x*v.y+c.rot.w*v.z)+2*c.tr.z;
    wfc.dV_wrt_db[0].rot.z = 2*(-2*c.rot.z*v.x-c.rot.w*v.y+c.rot.x*v.z)-2*c.tr.y;
    wfc.dV_wrt_db[0].rot.w = 2*(-c.rot.z*v.y+c.rot.y*v.z)+2*c.tr.x;

    wfc.dV_wrt_db[1].rot.x = 2*(c.rot.y*v.x-2*c.rot.x*v.y-c.rot.w*v.z)-2*c.tr.z;
    wfc.dV_wrt_db[1].rot.y = 2*(c.rot.x*v.x+c.rot.z*v.z)-2*c.tr.w;
    wfc.dV_wrt_db[1].rot.z = 2*(c.rot.w*v.x-2*c.rot.z*v.y+c.rot.y*v.z)+2*c.tr.x;
    wfc.dV_wrt_db[1].rot.w = 2*(c.rot.z*v.x-c.rot.x*v.z)+2*c.tr.y;

    wfc.dV_wrt_db[2].rot.x = 2*(c.rot.z*v.x+c.rot.w*v.y-2*c.rot.x*v.z)+2*c.tr.y;
    wfc.dV_wrt_db[2].rot.y = 2*(-c.rot.w*v.x+c.rot.z*v.y-2*c.rot.y*v.z)-2*c.tr.x;
    wfc.dV_wrt_db[2].rot.z = 2*(c.rot.x*v.x + c.rot.y*v.y)-2*c.tr.w;
    wfc.dV_wrt_db[2].rot.w = 2*(-c.rot.y*v.x+c.rot.x*v.y)+2*c.tr.z;

    wfc.dV_wrt_db[0].tr.x = 2*c.rot.w;
    wfc.dV_wrt_db[0].tr.y = -2*c.rot.z;
    wfc.dV_wrt_db[0].tr.z = 2*c.rot.y;
    wfc.dV_wrt_db[0].tr.w = -2*c.rot.x;

    wfc.dV_wrt_db[1].tr.x = 2*c.rot.z;
    wfc.dV_wrt_db[1].tr.y = 2*c.rot.w;
    wfc.dV_wrt_db[1].tr.z = -2*c.rot.x;
    wfc.dV_wrt_db[1].tr.w = -2*c.rot.y;

    wfc.dV_wrt_db[2].tr.x = -2*c.rot.y;
    wfc.dV_wrt_db[2].tr.y = 2*c.rot.x;
    wfc.dV_wrt_db[2].tr.z = 2*c.rot.w;
    wfc.dV_wrt_db[2].tr.w = -2*c.rot.z;

}

__kf_device__
void WarpField::compute_dN_wrt_db(WarpFieldCache &wfc) const {
    float b_inv_norm_quad = wfc.b_inv_norm_sq*wfc.b_inv_norm_sq;
    for (int i=0; i<3; i++) {
        for (int j = 0; j < DualQuaternion::ROTATION_PARAMS_LENGTH; j++) {
            wfc.dN_wrt_db[i].data[j] = -2*b_inv_norm_quad*wfc.b.data[j]*wfc.pretransformed_N[i]+wfc.b_inv_norm_sq*wfc.dN_wrt_db[i].data[j];
        }
    }

    for (int i=0; i<3; i++) {
        wfc.dN_wrt_db[i].tr = make_float4(0.,0.,0.,0.);
    }
}

__kf_device__
void WarpField::compute_dV_wrt_db(WarpFieldCache &wfc) const
{
    float b_inv_norm_quad = wfc.b_inv_norm_sq*wfc.b_inv_norm_sq;
    for (int i=0; i<3; i++) {
        for (int j = 0; j < DualQuaternion::ROTATION_PARAMS_LENGTH; j++) {
            wfc.dV_wrt_db[i].data[j] = -2*b_inv_norm_quad*wfc.b.data[j]*wfc.pretransformed_V[i]+wfc.b_inv_norm_sq*wfc.dV_wrt_db[i].data[j];
        }
        for (int j=DualQuaternion::ROTATION_PARAMS_LENGTH; j< DualQuaternion::PARAMS_LENGTH; j++)
        {
            wfc.dV_wrt_db[i].data[j] = wfc.b_inv_norm_sq*wfc.dV_wrt_db[i].data[j];
        }
    }
}

__kf_device__
void WarpField::compute_dV_wrt_dq(WarpFieldCache &wfc) const
{
    for (int nn=0; nn<knn::DiscreteBruteFieldKNN::k; nn++) {
        for (int vdim=0; vdim<3; vdim++)
        {
            for (int pnum=0; pnum<DualQuaternion::PARAMS_LENGTH; pnum++) {
                wfc.dV_wrt_params[(nn*3+vdim)*DualQuaternion::PARAMS_LENGTH+pnum] =
                wfc.dV_wrt_db[vdim].data[pnum]*wfc.warp_omega_weights[nn];
            }
        }
    }
}

__kf_device__
void WarpField::compute_dN_wrt_dq(WarpFieldCache &wfc) const
{
    for (int nn=0; nn<knn::DiscreteBruteFieldKNN::k; nn++) {
        for (int ndim=0; ndim<3; ndim++)
        {
            for (int pnum=0; pnum<DualQuaternion::ROTATION_PARAMS_LENGTH; pnum++) {
                wfc.dN_wrt_params[(nn*3+ndim)*DualQuaternion::ROTATION_PARAMS_LENGTH+pnum] =
                    wfc.dN_wrt_db[ndim].data[pnum]*wfc.warp_omega_weights[nn];
            }
        }
    }
}

__kf_device__
void WarpField::warp_forward(WarpFieldCache &wfc, float3 *vertex_ptr_in, float3 *vertex_ptr_out, float3 *normal_ptr_in, float3* normal_ptr_out) const
{
    *vertex_ptr_out = warp_vertex(wfc, *vertex_ptr_in);
    if (wfc.warp_applied) {
        *normal_ptr_out = warp_normal(wfc, *normal_ptr_in);
    }
    else
    {
        *normal_ptr_out = *normal_ptr_in;
    }
}

//Compute gradients V_warped_wrt_params_of_all_knn_nodes
__kf_device__
void WarpField::warp_backward(WarpFieldCache &wfc, float *gradV_memory, float *gradN_memory, int64_t *knn_memory) const { //call ONLY after forward pass
    if (wfc.warp_applied) {
        compute_dpN_wrt_db(wfc);
        compute_dpV_wrt_db(wfc);
        compute_dN_wrt_db(wfc);
        compute_dV_wrt_db(wfc);
        compute_dN_wrt_dq(wfc);
        compute_dV_wrt_dq(wfc);
        for (int i = 0; i < 3 * knn::DiscreteBruteFieldKNN::k * DualQuaternion::PARAMS_LENGTH; i++) {
            gradV_memory[i] = wfc.dV_wrt_params[i];
        }
        for (int i = 0; i < 3 * knn::DiscreteBruteFieldKNN::k * DualQuaternion::ROTATION_PARAMS_LENGTH; i++) {
            gradN_memory[i] = wfc.dN_wrt_params[i];
        }
    }
    else
    {
        for (int i = 0; i < 3 * knn::DiscreteBruteFieldKNN::k * DualQuaternion::PARAMS_LENGTH; i++) {
            gradV_memory[i] = 0;
        }
        for (int i = 0; i < 3 * knn::DiscreteBruteFieldKNN::k * DualQuaternion::ROTATION_PARAMS_LENGTH; i++) {
            gradN_memory[i] = 0;
        }
    }
    for (int i=0; i<knn::DiscreteBruteFieldKNN::k; i++) {

        knn_memory[i] = ((int64_t)wfc.V_knn[i]);//*knn::DiscreteBruteFieldKNN::k+i;
    }
}

__global__
void warp_with_grad_kernel(const WarpField wf, int verts_count, const float *verts_in, float *verts_out, const float *norms_in, float *norms_out,
                           float *gradV_memory, float *gradN_memory, int64_t *knn_memory)
{
    int off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off >= verts_count) {
        return;
    }
    int f3off = 3*off;
    WarpFieldCache wfc;
    wf.warp_forward(wfc, (float3*)(verts_in+f3off), (float3*)(verts_out+f3off), (float3*)(norms_in+f3off), (float3*)(norms_out+f3off));
    //pstatus(wf.warp_params);
    wf.warp_backward(wfc, gradV_memory+off*3*knn::DiscreteBruteFieldKNN::k*DualQuaternion::PARAMS_LENGTH,
            gradN_memory+off*3*knn::DiscreteBruteFieldKNN::k*DualQuaternion::ROTATION_PARAMS_LENGTH,
            knn_memory+off*knn::DiscreteBruteFieldKNN::k);
}

__global__
void dual_quaternion_blending_kernel(const WarpField wf, int verts_count, const float *verts_in, DualQuaternion *params_out)
{
    int off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off >= verts_count) {
        return;
    }
    params_out[off] = compute_dqb(wf, *((float3*)(verts_in+3*off)));
}

__global__
void get_unsupported_mask_kernel(const WarpField wf, int verts_count, const float *verts_in, unsigned char *mask_out)
{
    int off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off >= verts_count) {
        return;
    }
    mask_out[off] = is_unsupported(wf, *((float3*)(verts_in+3*off)))?1:0;
}

__global__
void warp_nograd_kernel(const WarpField wf, int verts_count, const float *verts_in, float *verts_out, const float *norms_in, float *norms_out)
{
    int off = threadIdx.x + blockIdx.x * blockDim.x;
    if (off >= verts_count) {
        return;
    }
    int f3off = 3*off;
    warp_nograd_nowsum(wf, *((float3*)(verts_in+f3off)), *((float3*)(norms_in+f3off)), *((float3*)(verts_out+f3off)), *((float3*)(norms_out+f3off)));
}

void WarpField_Host::warp_with_grad(int verts_count, const float *verts_in, float *verts_out, const float *norms_in, float *norms_out,
                    float *gradV_memory, float *gradN_memory, int64_t *knn_memory)
{
    dim3 block(256,1,1);
    dim3 grid (divUp (verts_count, block.x),1,1);
    check_warpfield();
    warp_with_grad_kernel<<<grid, block>>>(*warpfield_ptr, verts_count, verts_in, verts_out, norms_in, norms_out, gradV_memory, gradN_memory, knn_memory);
    cudaSafeCall (cudaGetLastError ());
    //cudaSafeCall ( cudaDeviceSynchronize() );
}

void WarpField_Host::dual_quaternion_blending(int verts_count, const float *verts_in, float *params_out)
{
    dim3 block(256,1,1);
    dim3 grid (divUp (verts_count, block.x),1,1);
    check_warpfield();
    dual_quaternion_blending_kernel<<<grid, block>>>(*warpfield_ptr, verts_count, verts_in, reinterpret_cast<DualQuaternion *>(params_out));
    cudaSafeCall (cudaGetLastError ());
    cudaSafeCall ( cudaDeviceSynchronize() );
}

void WarpField_Host::warp_nograd(int verts_count, const float *verts_in, float *verts_out, const float *norms_in, float *norms_out)
{
    dim3 block(256,1,1);
    dim3 grid (divUp (verts_count, block.x),1,1);
    check_warpfield();
    warp_nograd_kernel<<<grid, block>>>(*warpfield_ptr, verts_count, verts_in, verts_out, norms_in, norms_out);
    cudaSafeCall (cudaGetLastError ());
    //cudaSafeCall ( cudaDeviceSynchronize() );
}

void WarpField_Host::get_unsupported_mask(int verts_count, const float *verts_in, unsigned char *mask_out)
{
    dim3 block(256,1,1);
    dim3 grid (divUp (verts_count, block.x),1,1);
    check_warpfield();
    get_unsupported_mask_kernel<<<grid, block>>>(*warpfield_ptr, verts_count, verts_in, mask_out);
    cudaSafeCall (cudaGetLastError ());
    cudaSafeCall ( cudaDeviceSynchronize() );
}

void WarpField_Host::reset_nodes(float *warp_coords, float *warp_weights, int node_count) {
    node_coords_data.create(node_count*sizeof(float)*3);
    node_weights_data.create(node_count*sizeof(float));
    node_params_data.create(node_count*sizeof(DualQuaternion));
    warpfield_ptr->warp_coords = node_coords_data.ptr<float>();
    warpfield_ptr->warp_weights = node_weights_data.ptr<float>();
    warpfield_ptr->warp_params = node_params_data.ptr<DualQuaternion>();
    warpfield_ptr->node_count = node_count;
    cudaMemcpy(warpfield_ptr->warp_coords, warp_coords, node_count*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    cudaMemcpy(warpfield_ptr->warp_weights, warp_weights, node_count*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaSafeCall (cudaGetLastError ());
    cudaSafeCall ( cudaDeviceSynchronize() );
    knn_field.clear();
    knn_field.recompute_field(warpfield_ptr->warp_coords, 0, node_count);
}

void WarpField_Host::reset_node_params(DualQuaternion *warp_params)
{
    cudaMemcpy(warpfield_ptr->warp_params, warp_params, warpfield_ptr->node_count*sizeof(DualQuaternion), cudaMemcpyDeviceToDevice);
    cudaSafeCall (cudaGetLastError ());
    //cudaSafeCall ( cudaDeviceSynchronize() );
}

}