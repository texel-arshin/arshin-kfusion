#include "precomp.hpp"

using namespace kfusion;
using namespace kfusion::cuda;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume::Entry

//float kfusion::cuda::TsdfVolume::Entry::half2float(half)
//{ throw "Not implemented"; }
//
//kfusion::cuda::TsdfVolume::Entry::half kfusion::cuda::TsdfVolume::Entry::float2half(float value)
//{ throw "Not implemented"; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume

kfusion::cuda::TsdfVolume::TsdfVolume(const Vec3i& dims) : data_(), trunc_dist_(0.03f), max_weight_(128), dims_(dims),
    size_(Vec3f::Constant(3.f)), pose_(Affine3f::Identity()), gradient_delta_factor_(0.75f), raycast_step_factor_(0.75f)
{ create(dims_); }

kfusion::cuda::TsdfVolume::~TsdfVolume() {}

void kfusion::cuda::TsdfVolume::create(const Vec3i& dims)
{
    //std::cout<<"Create TSDF"<<std::endl;
    dims_ = dims;
    int voxels_number = dims_[0] * dims_[1] * dims_[2];
    //std::cout<<"Voxels "<<voxels_number<<std::endl;
    data_.create(voxels_number * sizeof(int));
    //std::cout<<"Allocated "<<voxels_number * sizeof(int)<<std::endl;
    setTruncDist(trunc_dist_);
    //std::cout<<"Ready to clear"<<std::endl;
    clear();
    //std::cout<<"Creation done"<<std::endl;
}

Vec3i kfusion::cuda::TsdfVolume::getDims() const
{ return dims_; }

Vec3f kfusion::cuda::TsdfVolume::getVoxelSize() const
{
    return Vec3f(size_[0]/dims_[0], size_[1]/dims_[1], size_[2]/dims_[2]);
}

const CudaData kfusion::cuda::TsdfVolume::data() const { return data_; }
CudaData kfusion::cuda::TsdfVolume::data() {  return data_; }
Vec3f kfusion::cuda::TsdfVolume::getSize() const { return size_; }

void kfusion::cuda::TsdfVolume::setSize(const Vec3f& size)
{ size_ = size; setTruncDist(trunc_dist_); }

float kfusion::cuda::TsdfVolume::getTruncDist() const { return trunc_dist_; }

void kfusion::cuda::TsdfVolume::setTruncDist(float distance)
{
    Vec3f vsz = getVoxelSize();
    float max_coeff = std::max<float>(std::max<float>(vsz[0], vsz[1]), vsz[2]);
    trunc_dist_ = std::max (distance, 2.1f * max_coeff);
}

float kfusion::cuda::TsdfVolume::getMaxWeight() const { return max_weight_; }
void kfusion::cuda::TsdfVolume::setMaxWeight(float weight) { max_weight_ = weight; }
Affine3f kfusion::cuda::TsdfVolume::getPose() const  { return pose_; }
void kfusion::cuda::TsdfVolume::setPose(const Affine3f& pose) { pose_ = pose; }
float kfusion::cuda::TsdfVolume::getRaycastStepFactor() const { return raycast_step_factor_; }
void kfusion::cuda::TsdfVolume::setRaycastStepFactor(float factor) { raycast_step_factor_ = factor; }
float kfusion::cuda::TsdfVolume::getGradientDeltaFactor() const { return gradient_delta_factor_; }
void kfusion::cuda::TsdfVolume::setGradientDeltaFactor(float factor) { gradient_delta_factor_ = factor; }
void kfusion::cuda::TsdfVolume::swap(CudaData& data) { data_.swap(data); }
void kfusion::cuda::TsdfVolume::applyAffine(const Affine3f& affine) { pose_ = affine * pose_; }

void kfusion::cuda::TsdfVolume::clear()
{
    //std::cout<<"Prepairing to cast"<<std::endl;
    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    //std::cout<<"Casted, allocating"<<std::endl;
    device::TsdfVolume volume(data_.ptr<half2>(), dims, vsz, trunc_dist_, max_weight_);
    //std::cout<<"Clearing"<<std::endl;
    device::clear_volume(volume);
    //std::cout<<"Cleared"<<std::endl;
}

void kfusion::cuda::TsdfVolume::integrate(const Dists& dists, const Affine3f& camera_vol2cam, const Intr& intr,
                                          const std::shared_ptr<warp::WarpField_Host> &wf_host_ptr)
{
    Affine3f vol2cam = camera_vol2cam * pose_;

    //std::cout<<"VOL2CAM\n"<<vol2cam.matrix()<<std::endl;
    //std::cout<<"CAMERA_POSE\n"<<camera_pose.matrix()<<std::endl;
    //std::cout<<"POSE_\n"<<pose_.matrix()<<std::endl;

    device::Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    //std::cout<<"DIMS: "<<dims.x << " "<<dims.y<<" "<< dims.z<<std::endl;
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff = device_cast<device::Aff3f>(vol2cam);
    //std::cout<<"AFF: "<<aff.R.data[0].x << " " << aff.t.x<<aff.t.y<<aff.t.z <<std::endl;

    device::TsdfVolume volume(data_.ptr<half2>(), dims, vsz, trunc_dist_, max_weight_);
    device::integrate(dists, volume, aff, proj, wf_host_ptr->get_warpfield());
}

void kfusion::cuda::TsdfVolume::integrate_nowarp(const Dists& dists, const Affine3f& camera_vol2cam, const Intr& intr)
{
    Affine3f vol2cam = camera_vol2cam * pose_;

    //std::cout<<"VOL2CAM\n"<<vol2cam.matrix()<<std::endl;
    //std::cout<<"CAMERA_POSE\n"<<camera_pose.matrix()<<std::endl;
    //std::cout<<"POSE_\n"<<pose_.matrix()<<std::endl;

    device::Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    //std::cout<<"DIMS: "<<dims.x << " "<<dims.y<<" "<< dims.z<<std::endl;
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff = device_cast<device::Aff3f>(vol2cam);
    //std::cout<<"AFF: "<<aff.R.data[0].x << " " << aff.t.x<<aff.t.y<<aff.t.z <<std::endl;

    device::TsdfVolume volume(data_.ptr<half2>(), dims, vsz, trunc_dist_, max_weight_);
    device::integrate_nowarp(dists, volume, aff, proj);
}

void kfusion::cuda::TsdfVolume::integrate_cpu(const Dists& dists, const Affine3f& camera_vol2cam, const Intr& intr, cpu_warp::WarpField_Host *wf_host_ptr)
{
    Affine3f vol2cam = camera_vol2cam * pose_;
    //std::cout<<"CPU2GPU INTEGRATION START\n"<<std::endl;
    //std::cout<<"VOL2CAM\n"<<vol2cam.matrix()<<std::endl;
    //std::cout<<"CAMERA_POSE\n"<<camera_pose.matrix()<<std::endl;
    //std::cout<<"POSE_\n"<<pose_.matrix()<<std::endl;

    device::Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    //std::cout<<"DIMS: "<<dims.x << " "<<dims.y<<" "<< dims.z<<std::endl;
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff = device_cast<device::Aff3f>(vol2cam);
    //std::cout<<"AFF: "<<aff.R.data[0].x << " " << aff.t.x<<aff.t.y<<aff.t.z <<std::endl;

    device::TsdfVolume volume(data_.ptr<half2>(), dims, vsz, trunc_dist_, max_weight_);
    device::integrate(dists, volume, aff, proj, wf_host_ptr->get_gpu_warpfield());
}

void kfusion::cuda::TsdfVolume::query_volume_with_grad(const DeviceArray<float> &verts, DeviceArray<float> &tsdfs, DeviceArray<float> &tsdfs_grad)
{
    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::TsdfVolume volume(data_.ptr<half2>(), dims, vsz, trunc_dist_, max_weight_);
    device::query_volume_with_grad(volume, verts, tsdfs, tsdfs_grad);
}

void kfusion::cuda::TsdfVolume::raycast(const Affine3f& vol2cam, const Intr& intr, Depth& depth, Normals& normals)
{
    DeviceArray2D<device::Normal>& n = (DeviceArray2D<device::Normal>&)normals;

    Affine3f cam2vol = pose_.inverse() * vol2cam.inverse();

    device::Aff3f aff = device_cast<device::Aff3f>(cam2vol);
    device::Mat3f Rinv = device_cast<device::Mat3f, Mat3f>(cam2vol.rotation().inverse());

    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::TsdfVolume volume(data_.ptr<half2>(), dims, vsz, trunc_dist_, max_weight_);
    device::raycast(volume, aff, Rinv, reproj, depth, n, raycast_step_factor_, gradient_delta_factor_);

}

void kfusion::cuda::TsdfVolume::raycast(const Affine3f& vol2cam, const Intr& intr, Cloud& points, Normals& normals)
{
    device::Normals& n = (device::Normals&)normals;
    device::Points& p = (device::Points&)points;

    Affine3f cam2vol = pose_.inverse() * vol2cam.inverse();

    device::Aff3f aff = device_cast<device::Aff3f>(cam2vol);
    device::Mat3f Rinv = device_cast<device::Mat3f, Mat3f>(cam2vol.rotation().inverse());

    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::TsdfVolume volume(data_.ptr<half2>(), dims, vsz, trunc_dist_, max_weight_);
    device::raycast(volume, aff, Rinv, reproj, p, n, raycast_step_factor_, gradient_delta_factor_);
}

DeviceArray<Point> kfusion::cuda::TsdfVolume::fetchCloud(DeviceArray<Point>& cloud_buffer) const
{
    enum { DEFAULT_CLOUD_BUFFER_SIZE = 10 * 1000 * 1000 };

    if (cloud_buffer.empty ())
        cloud_buffer.create (DEFAULT_CLOUD_BUFFER_SIZE);

    DeviceArray<device::Point>& b = (DeviceArray<device::Point>&)cloud_buffer;

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff  = device_cast<device::Aff3f>(pose_);

    device::TsdfVolume volume((half2*)data_.ptr<half2>(), dims, vsz, trunc_dist_, max_weight_);
    size_t size = extractCloud(volume, aff, b);

    return DeviceArray<Point>((Point*)cloud_buffer.ptr(), size);
}

void kfusion::cuda::TsdfVolume::fetchNormals(const DeviceArray<Point>& cloud, DeviceArray<Normal>& normals) const
{
    normals.create(cloud.size());
    DeviceArray<device::Point>& c = (DeviceArray<device::Point>&)cloud;

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff  = device_cast<device::Aff3f>(pose_);
    device::Mat3f Rinv = device_cast<device::Mat3f, Mat3f>(pose_.rotation().inverse());

    device::TsdfVolume volume((half2*)data_.ptr<half2>(), dims, vsz, trunc_dist_, max_weight_);
    device::extractNormals(volume, c, aff, Rinv, gradient_delta_factor_, (float4*)normals.ptr());
}
