from glob import glob
import numpy as np
import cv2


def load_files(dir_path:str) -> list:
    ''' 
    Input: directory path of files
    Oupput: sorted list of data files
    '''
    return sorted(glob(dir_path))

def load_lidar_file(file_path):
    ''' 
    Input: file_path is path of a *.bin file
    Oupput: numpy array with shape (n, 4)
    '''
    return np.fromfile(file_path, dtype=np.float32).reshape((-1, 4)) # xyzr

def load_lidar_to_refcam_calib(calib_path):
    '''
    Input: calibration information *.txt file path
    Output: Tr matrix with shape (4, 4)
    '''
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

def load_refcam_to_cam_calib(calib_path, cam_number):
    ''' 
    Input: 
        calib_path is *.txt file path of cam to cam calibration
    Oupput: 
        P is projection matrix with shape (4, 4)
        R is rotation matrix with shape (3, 4)
    '''
    calib = {}
    with open(calib_path) as f:
        for line in f.readlines():
            key, val = line.replace("\n", "").split(": ")
            calib[key] = val
            
    R = np.eye(4)
    R[:3, :3] = np.array(calib["R_rect_"+cam_number].split()).astype(np.float32).reshape(3,3)
    P = np.array(calib["P_rect_"+cam_number].split()).astype(np.float32).reshape(3,4)
    return P, R


def cart2hom(pts_3d):
    ''' 
    Input: points with shape (n, 3) in Cartesian 
    Oupput: points with shape (n, 4) in Homogeneous by pending 1 
    '''
    return np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))


def project_lidar_to_image(P, R, Tr, homo_3d_points):
    '''
    Input: 
        P is projection matrix to project rectification frame from reference frame with shape (4, 4)
        R is rotation matrix to project rectification frame from reference frame with shape (3, 4)
        Tr is result of dot product of rotation matrix and translation matrix with shape (4,4)
        homo_3d_points is 3d world points at homogeneous coordinates with shape (n, 4), xyz1
    Output: 
        2d image points at homogeneous coordinates with shape (n, 3), uv1
    '''
    return (P @ R @ Tr @ homo_3d_points).T




def load_img(file_path):
    return cv2.imread(file_path)


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr



def project_image_to_rect(uv_depth, P):
    ''' Input: nx3 first two channels are uv, 3rd channel
                is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    c_u = P[0, 2]
    c_v = P[1, 2]
    f_u = P[0, 0]
    f_v = P[1, 1]
    b_x = P[0, 3] / (-f_u)  # relative
    b_y = P[1, 3] / (-f_v)
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
    pts_3d_rect = np.zeros((n, 3))
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    return pts_3d_rect