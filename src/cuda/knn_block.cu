//#include "device.hpp"
//#include "knn.hpp"
//
//using namespace kfusion::cuda;
//
//__kf_device__
//int read_node(DiscreteFieldKNN df, elem_type local_off, int* blocks)
//{
//
//}
//
//__kf_device__
//void insert_node(DiscreteFieldKNN df, int3 coords, elem_type* field_ptr, int* blocks)
//{
//
//}
//
//__kf_device__
//int linearize_block_coords(DiscreteFieldKNN df, int3 coords) {
//    return (coords.x * df.block_dims.y * df.block_dims.z + coords.y * df.block_dims.z + coords.z) * df.block_size;
//}
//
//__kf_device__
//void get_blocks(DiscreteFieldKNN df, int3 coords, int* blocks)
//{
//    int3 block_coords = (coords+df.all_divider-1)/df.all_divider;
//    int block_counter = 0;
//    for (int i=-1; i<1; i++)
//    {
//        for (int j=-1; j<1; j++)
//        {
//            for (int k=-1; k<1; k++)
//            {
//                if (((block_coords.x+i) < df.block_dims.x) && ((block_coords.x+i) >= 0) &&
//                        ((block_coords.y+j) < df.block_dims.y) && ((block_coords.y+j) >= 0) &&
//                        ((block_coords.z+k) < df.block_dims.z) && ((block_coords.z+k) >= 0))
//                {
//                    blocks[block_counter++] = linearize_block_coords(df, block_coords);
//                }
//            }
//        }
//    }
//    for (int i=block_counter; i<8; i++)
//    {
//        block_counter[i] = -1;
//    }
//}
//
//__kf_device__
//void recompute_field_kernel(DiscreteFieldKNN df, int3 *coords, int nodes_count)
//{
//    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;
//
//    if (x >= df.field_dims.x || y >= df.field_dims.y)
//        return;
//
//    int blocks[8];
//
//    for (int i=0; i<nodes_count; i++; coords++)
//    {
//
//    }
//
//}
//
//DiscreteFieldKNN_Host::recompute_field(int3 *new_coords, int new_count, int3 *all_coords, int all_count)
//{
//
//}
