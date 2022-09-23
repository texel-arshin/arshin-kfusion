#include "kfusion/cpu/cpu_knn.hpp"
#include "kfusion/kdtree.hpp"

#include <stdio.h>
#include <vector>


namespace cpu_knn {


    typedef DiscreteBruteFieldKNN::elem_type elem_type;

    elem_type *get_voxel(DiscreteBruteFieldKNN df, int3 coords) {
        int lin_coords = (coords.x * df.field_dims.y * df.field_dims.z + coords.y * df.field_dims.z + coords.z) * df.k;
        return df.field_ptr + lin_coords;
    }

    float3 get_node(float *ptr, elem_type offset) {
        ptr += offset * 3;
        return make_float3(ptr[0], ptr[1], ptr[2]);
    }


    elem_type *get_knn_tsdf_coords(DiscreteBruteFieldKNN df, int3 coords) {
        int3 voxel_coords = make_int3(coords.x / df.field_divider, coords.y / df.field_divider, coords.z / df.field_divider);
        return get_voxel(df, voxel_coords);
    }

    elem_type *get_knn_field_coords(DiscreteBruteFieldKNN df, int3 coords) {
        return get_voxel(df, coords);
    }

    void query_field_kernel(DiscreteBruteFieldKNN df, int3 coords, elem_type *answers) {
        elem_type *voxel = get_knn_tsdf_coords(df, coords);
        for (int i = 0; i < df.k; i++) {
            answers[i] = voxel[i];
        }
    }


//void DiscreteBruteFieldKNN_Host::recompute_field(float *all_nodes_ptr, elem_type new_nodes_start_off, int new_nodes_count, bool use_kdtree)
//{
//    dim3 block(32, 8);
//    dim3 grid (divUp (discrete_field->field_dims.x, block.x), divUp (discrete_field->field_dims.y, block.y));
//    if (!use_kdtree) {
//        recompute_field_kernel(*discrete_field, all_nodes_ptr, new_nodes_start_off, new_nodes_count);
//    }
//    else
//    {
//        KDtree tree;
//        CUDA_KDTree gpu_tree;
//        Point *gpu_data = reinterpret_cast<Point *>(all_nodes_ptr+3*((int)new_nodes_start_off));
//        std::vector<Point> data(new_nodes_count);
//        cudaMemcpy(&data[0], gpu_data, sizeof(Point)*new_nodes_count, cudaMemcpyDeviceToHost);
//        tree.Create(data, 5);
//        gpu_tree.CreateKDTree(tree.GetRoot(), tree.GetNumNodes(), data);
//        recompute_field_with_kd_kernel(*discrete_field, gpu_tree.m_gpu_nodes, gpu_tree.m_gpu_indexes, gpu_tree.m_gpu_points, new_nodes_start_off, all_nodes_ptr);
//    }
//
//}

//void DiscreteBruteFieldKNN_Host::clear()
//{
//    dim3 block(32, 8);
//    dim3 grid (divUp (discrete_field->field_dims.x, block.x), divUp (discrete_field->field_dims.y, block.y));
//
//    clear_field_kernel(*discrete_field);
//}
//
    void DiscreteBruteFieldKNN_Host::query_field(int3 coords, elem_type *answers) {
//        dim3 block(32, 8);
//        dim3 grid (divUp (discrete_field->field_dims.x, block.x), divUp (discrete_field->field_dims.y, block.y));
        //printf("QUERY START\n");
        query_field_kernel(*discrete_field, coords, answers);
    }

}