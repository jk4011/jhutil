import open3d as o3d
import numpy as np
import copy
import torch


def preprocess_pcd_open3d(pcd_raw, normal=None, voxel_size=0.01):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_raw)

    radius_normal = voxel_size * 2

    if normal is None:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    else:
        pcd.normals = o3d.utility.Vector3dVector(normal)

    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,  # pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_down, pcd_fpfh


def ransac_open3d(source_down, target_down, source_fpfh, target_fpfh, voxel_size, normal_angle=0.05):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)

    # target_down.normals

    target_down.normals = o3d.utility.Vector3dVector(-torch.Tensor(target_down.normals))
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            # o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
            #     0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(normal_angle),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold * 0.1),
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.999))

    target_down.normals = o3d.utility.Vector3dVector(-torch.Tensor(target_down.normals))

    return result.transformation


def icp_open3d(src_pcd, ref_pcd, trans_init=np.identity(4), src_normal=None, ref_normal=None, voxel_size=0.01, part_assembly=True):
    if src_pcd.dim() != 2 or ref_pcd.dim() != 2:
        raise ValueError("src_pcd and ref_pcd must be 2D tensor")
    if src_pcd.shape[1] != 3 or ref_pcd.shape[1] != 3:
        raise ValueError("src_pcd and ref_pcd must have 3 columns")
    if len(src_pcd) == 0 or len(ref_pcd) == 0:
        return trans_init
        # raise ValueError("src_pcd and ref_pcd must have at least one point")
        
    distance_threshold = voxel_size * 0.4

    src, src_down, source_fpfh = preprocess_pcd_open3d(src_pcd, src_normal)
    ref, dst_down, target_fpfh = preprocess_pcd_open3d(ref_pcd, ref_normal)

    if part_assembly:
        normal = -np.array(ref.normals)
        ref.normals = o3d.utility.Vector3dVector(normal)

    result = o3d.pipelines.registration.registration_icp(
        src, ref, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    if part_assembly:
        normal = -np.array(ref.normals)
        ref.normals = o3d.utility.Vector3dVector(normal)
    return torch.Tensor(result.transformation.copy())


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, T):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result



def fast_global_registration_open3d(source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f"
          % distance_threshold)
    target_down.normals = o3d.utility.Vector3dVector(-torch.Tensor(target_down.normals))
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    target_down.normals = o3d.utility.Vector3dVector(-torch.Tensor(target_down.normals))
    return result.transformation
