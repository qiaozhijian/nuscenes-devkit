import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from pyquaternion import Quaternion
from PIL import Image
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

def map_pointcloud_to_image(nusc,
                            pointsensor_token: str,
                            camera_token: str,
                            min_dist: float = 1.0):
    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
        pc = LidarPointCloud.from_file(pcl_path)
    else:
        pc = RadarPointCloud.from_file(pcl_path)
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    raw_lidar_data = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    pixels = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove pixels that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure pixels are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, pixels[0, :] > 1)
    mask = np.logical_and(mask, pixels[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, pixels[1, :] > 1)
    mask = np.logical_and(mask, pixels[1, :] < im.size[1] - 1)
    pixels = pixels[:, mask]
    depths = depths[mask]
    points3d = raw_lidar_data[mask, :3].T

    points_colored = np.zeros((6, points3d.shape[1]))
    rgb = np.array(im)
    for i in range(points_colored.shape[1]):
        points_colored[:3, i] = points3d[:, i]
        points_colored[3:, i] = rgb[int(pixels[1, i]), int(pixels[0, i]), :]
    points_colored = points_colored.T
    return pixels, depths, points_colored