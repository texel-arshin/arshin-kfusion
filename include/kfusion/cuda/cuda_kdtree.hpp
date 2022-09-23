#ifndef __CUDA_KDTREE_HPP__
#define __CUDA_KDTREE_HPP__

#include "kfusion/kdtree.hpp"
#include <vector>

namespace knn {

    struct CUDA_KDNode {
        int level;
        int parent, left, right;
        float split_value;
        int num_indexes;
        int indexes;
    };


    struct CUDA_KDTree {
        ~CUDA_KDTree();

        void CreateKDTree(KDNode *root, int num_nodes, const std::vector <Point> &data);

        void Search(const std::vector <Point> &queries, std::vector<int> &indexes, std::vector<float> &dists);

        CUDA_KDNode *m_gpu_nodes;
        int *m_gpu_indexes;
        Point *m_gpu_points;

        int m_num_points;
    };

    __device__ void CudaKDSearch(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, const Point &query, int *ret_index, float *ret_dist, int k_neighbours = 1);

    void CheckCUDAError(const char *msg);
}

#endif
