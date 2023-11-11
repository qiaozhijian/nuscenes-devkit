import os
import open3d as o3d
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from utils import map_pointcloud_to_image

class NuScenesMapLiDARAlign:
    def __init__(self, data_root_dir, version, scene_id):
        self.nusc = NuScenes(version=version, dataroot=data_root_dir, verbose=False)
        self.scene = self.nusc.scene[scene_id]
        self.scene_log = self.nusc.get('log', self.scene['log_token'])
        self.nusc_map = NuScenesMap(dataroot=data_root_dir, map_name=self.scene_log['location'])
        self.layer_names = ['road_divider', 'lane_divider', 'ped_crossing']
    
    def get_lidar_tokens(self, scene):
        sample_token = scene['first_sample_token']
        lidar_sample_tokens = []
        lidar_sample_token = self.nusc.get('sample', sample_token)['data']['LIDAR_TOP']
        while lidar_sample_token != '':
            sample_data = self.nusc.get('sample_data', lidar_sample_token)
            if sample_data['is_key_frame']:
                lidar_sample_tokens.append(lidar_sample_token)
            lidar_sample_token = sample_data['next']

        return lidar_sample_tokens
    
    def get_map_patch(self, patch):
        records = self.nusc_map.get_records_in_patch(patch, self.layer_names)
        gts = []
        for layer in self.layer_names:
            for token in records[layer]:
                record = self.nusc_map.explorer.map_api.get(layer, token)
                if layer == 'ped_crossing':
                    polygon = self.nusc_map.extract_polygon(record['polygon_token'])
                    polygon = np.array(polygon.exterior.xy).T
                    polygon = np.vstack([polygon, polygon[0]])
                    gts.append(polygon)
                else:
                    line = self.nusc_map.extract_line(record['line_token'])
                    gts.append(np.array(line.xy).T)
        return gts

    def reserve_ground(self, points):
        height_over_ground = 1
        in_range_flags = ((points[:, 2] >= -height_over_ground) & (points[:, 2] <= height_over_ground))
        points = points[in_range_flags]
        return points

    def get_lidar2global(self, lidar_token):
        lidar_data = self.nusc.get('sample_data', lidar_token)
        calibrated_sensor_record = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        ego_pose_record = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        l2e_t = np.array(calibrated_sensor_record["translation"])
        l2e_r = Quaternion(calibrated_sensor_record["rotation"]).rotation_matrix
        e2g_t = np.array(ego_pose_record["translation"])
        e2g_r = Quaternion(ego_pose_record["rotation"]).rotation_matrix
        l2g_r = e2g_r @ l2e_r
        l2g_t = l2e_t @ e2g_r.T + e2g_t
        return l2g_r, l2g_t

    def get_lidar_points(self, lidar_token):
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        return points

    def paint_points(self, lidar_token):
        sample_token = self.nusc.get('sample_data', lidar_token)['sample_token']
        sample = self.nusc.get('sample', sample_token)
        cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        points_colored_lst = []
        for camera in cameras:
            cam_token = sample['data'][camera]
            points, depth, points_colored = map_pointcloud_to_image(self.nusc, lidar_token, cam_token, 1)
            points_colored_lst.append(points_colored)
        points_colored = np.concatenate(points_colored_lst)
        return points_colored

    def get_lidar_map(self):
        lidar_tokens = self.get_lidar_tokens(self.scene)
        lidar_map = []
        for lidar_token in lidar_tokens:
            # lidar_points = self.get_lidar_points(lidar_token)
            lidar_points = self.paint_points(lidar_token)
            l2g_r, l2g_t = self.get_lidar2global(lidar_token)
            lidar_points[:, :3] = lidar_points[:, :3] @ l2g_r.T + l2g_t
            lidar_map.append(lidar_points)
        lidar_map = np.concatenate(lidar_map)
        lidar_map = self.reserve_ground(lidar_map)
        return lidar_map

    def vis3d(self, points, gts):
        lines_pcd_lst = []
        for gt in gts:
            n_gt = len(gt)
            lines_pcd = o3d.geometry.LineSet()
            GT = np.hstack([gt, np.ones((n_gt, 1)) * 0.5])
            lines_pcd.points = o3d.utility.Vector3dVector(GT)
            lines = np.stack([np.arange(n_gt - 1), np.arange(1, n_gt)]).T
            lines_pcd.lines = o3d.utility.Vector2iVector(lines)
            color = [[0, 255, 0]] * len(lines)
            lines_pcd.colors = o3d.utility.Vector3dVector(color)
            lines_pcd_lst.append(lines_pcd)

        pcd = o3d.open3d.geometry.PointCloud()
        pcd.points = o3d.open3d.utility.Vector3dVector(points[:, :3])
        color = points[:, 3:] / 255
        pcd.colors = o3d.open3d.utility.Vector3dVector(color)
        pcd = pcd.voxel_down_sample(voxel_size=0.1)

        v = o3d.visualization.Visualizer()
        v.create_window()
        render_option = v.get_render_option()
        render_option.background_color = np.array([0, 0, 0])
        v.add_geometry(pcd)
        for lines_pcd in lines_pcd_lst:
            v.add_geometry(lines_pcd)

        v.run()
        v.destroy_window()


if __name__ == '__main__':

    data_root_dir = '/media/qzj/Document/datasets/nuscenes'
    version = 'v1.0-mini'

    map_lidar_align = NuScenesMapLiDARAlign(data_root_dir, version, 1)
    points = map_lidar_align.get_lidar_map()
    patch = (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max())

    # Get gts within patch.
    gts = map_lidar_align.get_map_patch(patch)

    print('Show the map and lidar points in global coordinate system.')
    map_lidar_align.vis3d(points, gts)