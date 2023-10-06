import cv2
import os
import numpy as np
from glob import glob


def load_files(dir_path):
    ''' 
    Input: image directory path
    Oupput: list of image files
    '''
    return sorted(glob(dir_path))

def load_lidar_file(file_path):
    ''' 
    Input: *.bin file path
    Oupput: numpy array shaped (n, 4)
    '''
    return np.fromfile(file_path, dtype=np.float32).reshape((-1, 4)) # xyzr


def calib_file(calib_path):
    calib = {}
    with open(calib_path) as f:
        for line in f.readlines():
            key, val = line.replace("\n", "").split(": ")
            calib[key] = val
    T = np.eye(4)
    T[:3, 3] = np.array(calib["T"].split()).astype(np.float32)
    R = np.eye(4)
    R[:3, :3] = np.array(calib["R"].split()).astype(np.float32).reshape(3, 3)

    TR = T @ R
    return TR

if __name__ == "__main__":
    image_00_files = load_files("data/kitti_sequence05/image_00/data/*")
    image_01_files = load_files("data/kitti_sequence05/image_01/data/*")
    image_02_files = load_files("data/kitti_sequence05/image_02/data/*")
    image_03_files = load_files("data/kitti_sequence05/image_03/data/*")

    lidar_files = load_files("data/kitti_sequence05/velodyne_points/data/*")
    print(load_lidar_file(lidar_files[0]))
    
    Tr = calib_file("data/kitti_sequence05/calib/calib_velo_to_cam.txt")


    img_file = cv2.imread(image_02_files[0])
    # img_file = cv2.cvtColor(image_02, cv2.COLOR_BGR2RGB)
    cv2.imshow("img", img_file)
    cv2.waitKey()
    
    num_images = len(image_00_files)

    # for ind in range(num_images):
    #     image_00 = cv2.imread(image_00_files[ind])
    #     image_01 = cv2.imread(image_01_files[ind])
    #     image_02 = cv2.imread(image_02_files[ind])
    #     image_03 = cv2.imread(image_03_files[ind])
    #     merged_up = cv2.hconcat([image_00, image_01])
    #     merged_dw = cv2.hconcat([image_02, image_03])
    #     merged = cv2.vconcat([merged_up, merged_dw])
        
    #     cv2.imshow("img", merged)
    #     key = cv2.waitKey()
    #     if key == 'a':
    #         break
        

        # print(file)
    # print("hello")