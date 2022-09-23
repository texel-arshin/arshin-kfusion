#include "device.hpp"
#include "kfusion/cuda/knn.hpp"
#include "kfusion/kdtree.hpp"
#include "kfusion/cuda/cuda_kdtree.hpp"
#include <iostream>

#include <stdio.h>
#include <float.h>
#include <vector>

namespace knn {

    using namespace kfusion::device;

    static const int NODE_IDX_STACK_SIZE = 100; // fixed size stack elements for each thread, increase as required. Used in SearchAtNodeRange.

    typedef DiscreteBruteFieldKNN::elem_type elem_type;

    __device__ __forceinline__
    elem_type* get_voxel(const DiscreteBruteFieldKNN &df, int3 coords) {
        int lin_coords = (coords.z*df.field_dims.y*df.field_dims.x + coords.y*df.field_dims.x + coords.x) * df.k;
        return df.field_ptr+lin_coords;
    }

    __device__ __forceinline__
    float3 get_node(float* __restrict__ ptr,
                    elem_type offset)
    {
        ptr+=offset*3;
        return make_float3(ptr[0], ptr[1], ptr[2]);
    }

    __kf_device__
    float dist(float3 a, float3 b)
    {
//        int3 d = make_int3(a.x-b.x,a.y-b.y,a.z-b.z);
//        return d.x*d.x+d.y*d.y+d.z*d.z;
        float3 d = a-b;
        return dot(d,d);
    }

    __kf_device__
    void insert_node(DiscreteBruteFieldKNN df, elem_type* curr_voxel, float3 voxel_coords, float3 node_coords, elem_type node_off, float *all_nodes_ptr) {
//        int x = threadIdx.x + blockIdx.x * blockDim.x;
//        int y = threadIdx.y + blockIdx.y * blockDim.y;
//        if (x==0 && y ==0)
//            printf("Voxel coords: (%f, %f, %f)", voxel_coords.x, voxel_coords.y, voxel_coords.z);
        float node_dist = dist(node_coords, voxel_coords);
        for (int i=0; i<df.k; i++)
        {
            bool place = false;

            elem_type v_off = curr_voxel[i];
            if (v_off!=DiscreteBruteFieldKNN::UNDEFINED_OFFSET) {

                float3 off_node_coords = get_node(all_nodes_ptr,v_off);



                place = (dist(off_node_coords, voxel_coords) > node_dist);
            }
            else
            {
                place = true;
            }
            if (place)
            {
                elem_type old_val, curr_val = node_off;
                for (int j=i; j<df.k; j++)
                {
                    old_val = curr_voxel[j];
                    curr_voxel[j] = curr_val;
                    curr_val = old_val;
                }
                return;
            }
        }
    }

    __global__
    void clear_field_kernel(DiscreteBruteFieldKNN df)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x >= df.field_dims.x || y >= df.field_dims.y)
            return;

        elem_type *curr_ptr = df.field_ptr+(x*df.field_dims.y*df.field_dims.z+y*df.field_dims.z)*df.k;

        for (int z = 0; z<df.field_dims.z; z++)
        {
            for (int k=0; k<df.k; k++)
            {
                *curr_ptr = DiscreteBruteFieldKNN::UNDEFINED_OFFSET;
                curr_ptr++;
            }
        }
    }

    __global__
    void recompute_field_kernel(DiscreteBruteFieldKNN df, float *all_nodes_ptr,
            elem_type new_nodes_start_off, int new_nodes_count) {
      int id = threadIdx.x + blockDim.x * blockIdx.x;
      if (id < df.field_dims.x * df.field_dims.y * df.field_dims.z) {
        int z =  id / (df.field_dims.y * df.field_dims.x);
        int y = (id - z * (df.field_dims.y * df.field_dims.x)) / df.field_dims.x;
        int x = id - (z * df.field_dims.y + y) * df.field_dims.x;

        elem_type curr_node_off = new_nodes_start_off;
        for (int i = 0; i < new_nodes_count; i++, curr_node_off++)
        {
            {
                elem_type *curr_voxel = get_voxel(df, make_int3(x, y, z));
                insert_node(df, curr_voxel, make_float3((x+0.5) * df.field_cell_size, (y+0.5) * df.field_cell_size, (z+0.5) * df.field_cell_size),
                            get_node(all_nodes_ptr, curr_node_off), curr_node_off, all_nodes_ptr);
            }
        }

      }
    }



    __device__ __forceinline__
    int insert_kd_node(const DiscreteBruteFieldKNN &df,
                       elem_type * __restrict__ curr_voxel,
                       float3 &voxel_coords,
                       float node_dist,
                       elem_type node_off,
                       float * __restrict__ all_nodes_ptr,
                       int start_i) {
        for (int i=start_i; i<df.k; i++)
        {
            bool place = false;

            elem_type v_off = curr_voxel[i];
            if (v_off!=DiscreteBruteFieldKNN::UNDEFINED_OFFSET) {

                float3 off_node_coords = get_node(all_nodes_ptr,v_off);



                place = (dist(off_node_coords, voxel_coords) > node_dist);
            }
            else
            {
                place = true;
            }
            if (place)
            {
                elem_type old_val, curr_val = node_off;
                for (int j=i; j<df.k; j++)
                {
                    old_val = curr_voxel[j];
                    curr_voxel[j] = curr_val;
                    curr_val = old_val;
                }
                return i+1;
            }
        }
        return df.k;
    }

    namespace {

    __device__ __forceinline__
    float Distance(const Point &a, const Point &b) {
        float dist = 0;

        for (int i = 0; i < KDTREE_DIM; i++) {
            float d = a.coords[i] - b.coords[i];
            dist += d * d;
        }

        return dist;
    }

    __device__ __forceinline__
    bool CheckIgnored(int idx,
                      int * __restrict__ ignored_indices,
                      int ignored_count)
    {
        for (int i=0; i<ignored_count; i++)
        {
            if (idx == ignored_indices[i])
            {
                return false;
            }
        }
        return true;
    }

    __device__ __forceinline__
    void SearchAtNode(const CUDA_KDNode * __restrict__ nodes,
                      const int * __restrict__ indexes,
                      const Point * __restrict__ pts,
                      int cur,
                      const Point &query,
                      int * __restrict__ ret_index,
                      float * __restrict__ ret_dist,
                      int * __restrict__ ret_node,
                      int * __restrict__ ignored_indices,
                      int ignored_count) {
        // Finds the first potential candidate

        int best_idx = -1;
        float best_dist = FLT_MAX;

        while (true) {
            int split_axis = nodes[cur].level % KDTREE_DIM;

            if (nodes[cur].left == -1) {
                *ret_node = cur;

                for (int i = 0; i < nodes[cur].num_indexes; i++) {
                    int idx = indexes[nodes[cur].indexes + i];
                    float dist = Distance(query, pts[idx]);
                    if ((dist < best_dist) && CheckIgnored(idx, ignored_indices, ignored_count)) {
                        best_dist = dist;
                        best_idx = idx;
                    }
                }

                break;
            } else if (query.coords[split_axis] < nodes[cur].split_value) {
                cur = nodes[cur].left;
            } else {
                cur = nodes[cur].right;
            }
        }

        *ret_index = best_idx;
        *ret_dist = best_dist;
    }




    __device__ __forceinline__
    void SearchAtNodeRange(const CUDA_KDNode * __restrict__ nodes,
                           const int * __restrict__ indexes,
                           const Point * __restrict__ pts,
                           const Point &query,
                           int cur,
                           float range,
                           int * __restrict__ ret_index,
                           float * __restrict__ ret_dist,
                           int * __restrict__ ignored_indices,
                           int ignored_count) {
        // Goes through all the nodes that are within "range"

        int best_idx = -1;
        float best_dist = FLT_MAX;

        // Ok, we don't have nice STL vectors to use, and we can't dynamically allocate memory with CUDA??
        // We'll use a fixed length stack, increase this as required
        int to_visit[NODE_IDX_STACK_SIZE];
        int to_visit_pos = 0;

        to_visit[to_visit_pos++] = cur;

        while (to_visit_pos) {
            int next_search[NODE_IDX_STACK_SIZE];
            int next_search_pos = 0;

            while (to_visit_pos) {
                cur = to_visit[to_visit_pos - 1];
                to_visit_pos--;

                int split_axis = nodes[cur].level % KDTREE_DIM;

                if (nodes[cur].left == -1) {
                    for (int i = 0; i < nodes[cur].num_indexes; i++) {
                        int idx = indexes[nodes[cur].indexes + i];
                        float d = Distance(query, pts[idx]);

                        if ((d < best_dist) && CheckIgnored(idx, ignored_indices, ignored_count)) {
                            best_dist = d;
                            best_idx = idx;
                        }
                    }
                } else {
                    float d = query.coords[split_axis] - nodes[cur].split_value;

                    // There are 3 possible scenarios
                    // The hypercircle only intersects the left region
                    // The hypercircle only intersects the right region
                    // The hypercricle intersects both

                    if (fabs(d) > range) {
                        if (d < 0)
                            next_search[next_search_pos++] = nodes[cur].left;
                        else
                            next_search[next_search_pos++] = nodes[cur].right;
                    } else {
                        next_search[next_search_pos++] = nodes[cur].left;
                        next_search[next_search_pos++] = nodes[cur].right;
                    }
                }
            }

            // No memcpy available??
            for (int i = 0; i < next_search_pos; i++)
                to_visit[i] = next_search[i];

            to_visit_pos = next_search_pos;
        }

        *ret_index = best_idx;
        *ret_dist = best_dist;
    }


    __device__ __forceinline__
    void SearchOne(const CUDA_KDNode * __restrict__ nodes,
                   const int * __restrict__ indexes,
                   const Point * __restrict__ pts,
                   const Point &query,
                   int * __restrict__ ret_index,
                   float * __restrict__ ret_dist,
                   int * __restrict__ ignored_indices,
                   int ignored_count) {
        // Find the first closest node, this will be the upper bound for the next searches
        int best_node = 0;
        int best_idx = -1;
        float best_dist = FLT_MAX;
        float radius = 0;

        SearchAtNode(nodes, indexes, pts, 0 /* root */, query, &best_idx, &best_dist, &best_node, ignored_indices, ignored_count);

        radius = sqrt(best_dist);

        // Now find other possible candidates
        int cur = best_node;

        while (nodes[cur].parent != -1) {
            // Go up
            int parent = nodes[cur].parent;
            int split_axis = nodes[parent].level % KDTREE_DIM;

            // Search the other node
            float tmp_dist = FLT_MAX;
            int tmp_idx;

            if (fabs(nodes[parent].split_value - query.coords[split_axis]) <= radius) {
                // Search opposite node
                if (nodes[parent].left != cur)
                    SearchAtNodeRange(nodes, indexes, pts, query, nodes[parent].left, radius, &tmp_idx, &tmp_dist, ignored_indices, ignored_count);
                else
                    SearchAtNodeRange(nodes, indexes, pts, query, nodes[parent].right, radius, &tmp_idx, &tmp_dist, ignored_indices, ignored_count);
            }

            if (tmp_dist < best_dist) {
                best_dist = tmp_dist;
                best_idx = tmp_idx;
            }

            cur = parent;
        }

        *ret_index = best_idx;
        *ret_dist = best_dist;
    }

    __device__ __forceinline__
    void CudaKDSearchKnn(const CUDA_KDNode * __restrict__ nodes,
                         const int * __restrict__ indexes,
                         const Point * __restrict__ pts,
                         const Point &query,
                         int * __restrict__ ret_index,
                         float * __restrict__ ret_dist,
                         int k_neighbours)
    {
        for (int nn_count = 0; nn_count<k_neighbours; nn_count++)
        {
            SearchOne(nodes, indexes, pts, query, ret_index+nn_count, ret_dist+nn_count, ret_index, nn_count);
        }
    }

  }
    __global__
    void recompute_field_with_kd_kernel(const DiscreteBruteFieldKNN &df,
                                        const CUDA_KDNode * __restrict__ m_gpu_nodes,
                                        const int * __restrict__ m_gpu_indexes,
                                        const Point * __restrict__ m_gpu_points,
                                        elem_type new_nodes_start_off,
                                        float * __restrict__ all_nodes_ptr) {
      int id = threadIdx.x + blockDim.x * blockIdx.x;
      if (id < df.field_dims.x * df.field_dims.y * df.field_dims.z) {
        int z =  id / (df.field_dims.y * df.field_dims.x);
        int y = (id - z * (df.field_dims.y * df.field_dims.x)) / df.field_dims.x;
        int x = id - (z * df.field_dims.y + y) * df.field_dims.x;
        int found_inds[df.k];
        float found_dists[df.k];

        {
            elem_type *curr_voxel = get_voxel(df, make_int3(x, y, z));
            Point query;
            query.coords[0] = (x+0.5) * df.field_cell_size;
            query.coords[1] = (y+0.5) * df.field_cell_size;
            query.coords[2] = (z+0.5) * df.field_cell_size;
            float3 voxel_coords = make_float3(query.coords[0],query.coords[1],query.coords[2]);
            CudaKDSearchKnn(m_gpu_nodes, m_gpu_indexes, m_gpu_points, query, found_inds, found_dists, df.k);
            int last_i=0;
            for (int found_i=0; found_i<df.k; found_i++)
            {
                if (last_i<df.k && found_inds[found_i]!=-1) {
                    last_i = insert_kd_node(df, curr_voxel, voxel_coords, found_dists[found_i], (elem_type) found_inds[found_i] + new_nodes_start_off,
                                   all_nodes_ptr, last_i);
                }
            }
        }

      }
    }

    __kf_device__
    void change_coords(const DiscreteBruteFieldKNN &df, int3 &c)
    {
        if (c.x < 0) c.x = 0;
        else if (c.x >= df.field_dims.x) c.x = df.field_dims.x;
        if (c.y < 0) c.y = 0;
        else if (c.y >= df.field_dims.y) c.y = df.field_dims.y;
        if (c.z < 0) c.z = 0;
        else if (c.z >= df.field_dims.z) c.z = df.field_dims.z;
    }

    __device__
    elem_type* get_knn_tsdf_coords(const DiscreteBruteFieldKNN &df, int3 coords)
    {
        int3 voxel_coords = make_int3(coords.x/df.field_divider, coords.y/df.field_divider, coords.z/df.field_divider);
        change_coords(df, voxel_coords);
        return get_voxel(df, voxel_coords);
    }

    __device__
    elem_type* get_knn_field_coords(const DiscreteBruteFieldKNN &df, int3 coords)
    {
        change_coords(df, coords);
        return get_voxel(df, coords);
    }

    __global__
    void query_field_kernel(DiscreteBruteFieldKNN df, int3 coords, elem_type* answers)
    {
        elem_type* voxel = get_knn_tsdf_coords(df, coords);
        for (int i = 0; i<df.k; i++)
        {
            answers[i] = voxel[i];
        }
    }


    void DiscreteBruteFieldKNN_Host::recompute_field(float *all_nodes_ptr,
                                                     elem_type new_nodes_start_off,
                                                     int new_nodes_count, bool use_kdtree)
    {
        dim3 block(1024, 1, 1); // Optimum for RTX2070
        dim3 grid (divUp (discrete_field->field_dims.x *
                          discrete_field->field_dims.y *
                          discrete_field->field_dims.z, block.x), 1, 1);
        if (!use_kdtree) {
            recompute_field_kernel <<< grid, block >>> (*discrete_field, all_nodes_ptr, new_nodes_start_off, new_nodes_count);
        }
        else
        {
            KDtree tree;
            CUDA_KDTree gpu_tree;
            Point *gpu_data = reinterpret_cast<Point *>(all_nodes_ptr+3*((int)new_nodes_start_off));
            std::vector<Point> data(new_nodes_count);
            cudaMemcpy(&data[0], gpu_data, sizeof(Point)*new_nodes_count, cudaMemcpyDeviceToHost);
            tree.Create(data, 5);
            gpu_tree.CreateKDTree(tree.GetRoot(), tree.GetNumNodes(), data);
            DiscreteBruteFieldKNN* cuda_df;
            cudaSafeCall ( cudaMalloc(&cuda_df, sizeof(DiscreteBruteFieldKNN)) );
            cudaSafeCall ( cudaMemcpy(cuda_df, discrete_field.get(),
                                      sizeof(DiscreteBruteFieldKNN),
                                      cudaMemcpyHostToDevice) );

            recompute_field_with_kd_kernel<<< grid, block >>>(*cuda_df,
                                                              gpu_tree.m_gpu_nodes,
                                                              gpu_tree.m_gpu_indexes,
                                                              gpu_tree.m_gpu_points,
                                                              new_nodes_start_off,
                                                              all_nodes_ptr);
        }
        cudaSafeCall (cudaGetLastError ());
        cudaSafeCall ( cudaDeviceSynchronize() );

    }

    void DiscreteBruteFieldKNN_Host::clear()
    {
        size_t field_size = discrete_field->field_dims.x *
            discrete_field->field_dims.y *
            discrete_field->field_dims.z *
            discrete_field->k;
        cudaSafeCall(cudaMemset(discrete_field->field_ptr, -1, field_size * sizeof(elem_type)));
        cudaSafeCall (cudaGetLastError ());
        cudaSafeCall ( cudaDeviceSynchronize() );
    }

    void DiscreteBruteFieldKNN_Host::query_field(int3 coords, elem_type* answers)
    {
//        dim3 block(32, 8);
//        dim3 grid (divUp (discrete_field->field_dims.x, block.x), divUp (discrete_field->field_dims.y, block.y));
        //printf("QUERY START\n");
        query_field_kernel<<<1, 1>>>(*discrete_field, coords, answers);
        cudaSafeCall (cudaGetLastError ());
        cudaSafeCall ( cudaDeviceSynchronize() );
    }

}
