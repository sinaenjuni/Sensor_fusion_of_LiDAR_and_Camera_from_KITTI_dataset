import cv2
import os
import numpy as np

from project_lidar_to_camera import *
from o3d import O3D

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import time
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
    # parameters = O3D.load_calib("./parameters.json")
    # parameters = o3d.io.read_pinhole_camera_parameters("./parameters.json")
    Tr = load_lidar_to_refcam_calib("data/kitti_sequence05/calib/calib_velo_to_cam.txt")
    Tr_inv = inverse_rigid_trans(Tr)
    P, R = load_refcam_to_cam_calib("data/kitti_sequence05/calib/calib_cam_to_cam.txt", "02")
    print(P)
    print(R)

    # open3d
    vis_3d_truth = O3D(width=2048, height=1024, 
                init_points=load_lidar_file(lidar_files[0])[:, :3], 
                calib="./o3d_calibration.json"
                )
    vis_3d_reprojection = O3D(width=2048, height=1024, 
                init_points=load_lidar_file(lidar_files[0])[:, :3], 
                calib="./o3d_calibration.json"
                )
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

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
            z = ((z - min) / max) * 255
            z = 255-z
            cv2.circle(points_arr, center=(cx, cy), radius=1,
                        color=(z,z,z), thickness=-1, lineType=cv2.LINE_4)
            
        img = cv2.addWeighted(img, 1, points_arr, 1, 1)

        pts_3d_uv = cart2hom(project_image_to_rect(pts_2d_hom, P))
        proj_to_lidar = (Tr_inv @ np.linalg.inv(R) @ pts_3d_uv.T).T


        # blocking
        # vis.run()

        cv2.imshow("img", img)
        key = cv2.waitKey(1)
        if key == 32:
            while(cv2.waitKey(5) != 32):
                key = cv2.waitKey(5)
                if key == 27: break
        if key == 27: break

        vis_3d_truth.updata_pcd(lidar_file[:, :3])
        vis_3d_reprojection.updata_pcd(proj_to_lidar[:, :3])


    del(vis_3d_truth)
    del(vis_3d_reprojection)
    cv2.destroyAllWindows()