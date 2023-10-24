import cv2
import os
import numpy as np

from project_lidar_to_camera import *
import open3d as o3d
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

# import cProfile
# import pstats
# with cProfile.Profile() as pr:

# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats()

if __name__ == "__main__":
    image_00_files = load_files("data/kitti_sequence05/image_00/data/*")
    image_01_files = load_files("data/kitti_sequence05/image_01/data/*")
    image_02_files = load_files("data/kitti_sequence05/image_02/data/*")
    image_03_files = load_files("data/kitti_sequence05/image_03/data/*")
    lidar_files = load_files("data/kitti_sequence05/velodyne_points/data/*")
    parameters = o3d.io.read_pinhole_camera_parameters("./parameters.json")
    Tr = load_lidar_to_refcam_calib("data/kitti_sequence05/calib/calib_velo_to_cam.txt")
    Tr_inv = inverse_rigid_trans(Tr)
    P, R = load_refcam_to_cam_calib("data/kitti_sequence05/calib/calib_cam_to_cam.txt", "02")
    print(P)
    print(R)

    # open3d
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=3000, height=1500, visible=True)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 3
    ctr = vis.get_view_control()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(load_lidar_file(lidar_files[0])[:, :3])
    vis.add_geometry(pcd)

    for ind in range(len(image_00_files)):
        lidar_file = load_lidar_file(lidar_files[ind])
        img = load_img(image_02_files[ind])
        h, w ,c = img.shape

        print(P.shape, R.shape, Tr.shape)
        pts_3d_hom = cart2hom(lidar_file[:, :3]) # lidar 3d point to homogeneous point with xyz1
        pts_2d_hom = project_lidar_to_image(P, R, Tr, pts_3d_hom.T)
        print(pts_2d_hom.shape)

        pts_2d_hom = pts_2d_hom[pts_2d_hom[:, 2] > 0]
        pts_2d_hom = np.hstack((pts_2d_hom[..., [0, 1]] / pts_2d_hom[..., [2]], 
                                pts_2d_hom[..., [2]])) # homogeneous to image coordinate
        
        pts_2d_hom = pts_2d_hom[pts_2d_hom[..., 0] < w]
        pts_2d_hom = pts_2d_hom[pts_2d_hom[..., 1] < h]

        min = pts_2d_hom[..., 2].min()
        max = pts_2d_hom[..., 2].max()

        points_arr = np.zeros_like(img)
        for cx, cy, z in pts_2d_hom:
            cx, cy = map(int, (cx, cy))
            z = (z - min) / max * 255
            z = 255-z
            cv2.circle(points_arr, center=(cx, cy), radius=1,
                        color=(z,z,z), thickness=-1, lineType=cv2.LINE_AA)
            
        img = cv2.addWeighted(img, .5, points_arr, 1, 1)

        # pts_3d_uv = project_image_to_rect(pts_2d_hom, P)
        # proj_to_lidar = (Tr_inv @ np.linalg.inv(R) @ pts_3d_uv.T).T

        pcd.points = o3d.utility.Vector3dVector(lidar_file[:, :3])
        vis.update_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(parameters)

        keep_running = vis.poll_events()
        vis.update_renderer()
        # vis.run()

        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break
        

    cv2.destroyAllWindows()
    o3d.destroy_window()