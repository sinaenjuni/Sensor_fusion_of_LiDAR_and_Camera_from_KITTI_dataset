import cv2
import numpy as np
from multiprocessing import shared_memory, Process, Event
import open3d as o3d
import time
from project_lidar_to_camera import *


WIDTH    = 1242
HEIGHT   = 375
CHANNELS = 3

sample_img_array = np.zeros((HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)
sample_lidar_array = np.zeros((200000, 4), dtype=np.float32)

def show_img(shm_name, win_name):
    while True:
        try:
            stream_shm = shared_memory.SharedMemory(name=shm_name, create=False, size=sample_img_array.nbytes)
            # stream_shm = shared_memory.SharedMemory(name=shm_name, create=False)
            break
        except FileNotFoundError:
            continue

    img = np.ndarray(sample_img_array.shape, dtype=sample_img_array.dtype, buffer=stream_shm.buf)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WIDTH, HEIGHT)

    while True:
        cv2.imshow(win_name, img)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyWindow(win_name)

    stream_shm.unlink()
    stream_shm.close()

def show_3d_points(shm_name, win_name):
    while True:
        try:
            stream_shm = shared_memory.SharedMemory(name=shm_name, create=False)
            break
        except FileNotFoundError:
            print("not found")
            continue
    shared_array = np.frombuffer(stream_shm.buf, dtype=np.float32).reshape(-1, 4)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=2000, height=1000, visible=True)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 3

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(shared_array[:, :3])
    # pcd.colors = o3d.utility.Vector3dVector(np.array([[255,0,0]]*len(shared_array)))
    vis.add_geometry(pcd)
    # vis.run()

    keep_running = True
    while keep_running:
        print(keep_running)
        shared_array = np.frombuffer(stream_shm.buf, dtype=np.float32).reshape(-1, 4)
        print("receive", shared_array.shape)
        print(len(shared_array))
        print("receive", shared_array) 

        # vis.clear_geometries()
        # pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(shared_array[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(np.array([[255,0,0]]*len(shared_array)))
        vis.update_geometry(pcd)    
        # vis.add_geometry(pcd, False)
    

        keep_running = vis.poll_events()
        vis.update_renderer()
        
    vis.destroy_window()
    stream_shm.unlink()
    stream_shm.close()
    # raw_data = stream_shm.buf[:]
    # data_array = np.frombuffer(stream_shm.buf, dtype=np.float32)
    # print(data_array)
    # lidar_file = np.ndarray(dtype=np.float32, buffer=stream_shm.buf)
    # print(lidar_file)
    # # open3d
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=3000, height=1500, visible=True)

    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])
    # opt.point_size = 3

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(lidar_file[:, :3])
    # vis.add_geometry(pcd, True)
    
    # vis.run()



def main(): 
    image_00_files = load_files("data/kitti_sequence05/image_00/data/*")
    image_01_files = load_files("data/kitti_sequence05/image_01/data/*")
    image_02_files = load_files("data/kitti_sequence05/image_02/data/*")
    image_03_files = load_files("data/kitti_sequence05/image_03/data/*")
    lidar_files = load_files("data/kitti_sequence05/velodyne_points/data/*")

    Tr = load_lidar_to_refcam_calib("data/kitti_sequence05/calib/calib_velo_to_cam.txt")
    Tr_inv = inverse_rigid_trans(Tr)
    P, R = load_refcam_to_cam_calib("data/kitti_sequence05/calib/calib_cam_to_cam.txt", "02")



    lidar_shm = shared_memory.SharedMemory(name ='lidar_shm', create=True, size=sample_lidar_array.nbytes)
    img_shm = shared_memory.SharedMemory(name ='img_shm', create=True, size=sample_img_array.nbytes)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])
    # opt.point_size = 3

    # pcd.points = o3d.utility.Vector3dVector(load_lidar_file(lidar_files[0])[:, :3])
    # vis.add_geometry(pcd)

    # for i in range(icp_iteration):
    #     # do ICP single iteration
    #     # transform geometry using ICP
    #     vis.update_geometry(geometry)
    #     vis.poll_events()
    #     vis.update_renderer()

    for ind in range(100):
        lidar_file = load_lidar_file(lidar_files[ind])
        pts_3d_hom = cart2hom(lidar_file[:, :3]) # lidar 3d point to homogeneous point with xyz1
        pts_2d_hom = project_lidar_to_image(P, R, Tr, pts_3d_hom.T)
        pts_2d_hom = pts_2d_hom[pts_2d_hom[:, 2] > 0]
        pts_2d_hom = np.concatenate(
             (pts_2d_hom[..., [0, 1]] / pts_2d_hom[..., [2]], pts_2d_hom[..., [2]]), 
             axis=1
             ) # homogeneous to image coordinate
        pts_2d_hom = pts_2d_hom[pts_2d_hom[..., 0] < WIDTH]
        pts_2d_hom = pts_2d_hom[pts_2d_hom[..., 1] < HEIGHT]
        min = pts_2d_hom[..., 2].min()
        max = pts_2d_hom[..., 2].max()

        img = load_img(image_02_files[ind])
        img = cv2.resize(img, (WIDTH, HEIGHT))
        points_arr = np.zeros_like(img)
        for cx, cy, z in pts_2d_hom:
            cx, cy = map(int, (cx, cy))
            z = (z - min) / max * 255
            cv2.circle(
                 points_arr, center=(cx, cy), radius=2, 
                 color=(z,z,z), thickness=-1, lineType=cv2.LINE_AA
                 )

        img = cv2.addWeighted(img, 1, points_arr, 1, 1)

        # print(lidar_file.dtype, lidar_file.shape)
        # vis.clear_geometries()
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(lidar_file[:, :3])
        # vis.add_geometry(pcd)
        # vis.poll_events()
        # vis.update_renderer()

        lidar_shared = np.ndarray(sample_lidar_array.shape, dtype=sample_lidar_array.dtype, buffer=lidar_shm.buf)
        lidar_shared[:len(lidar_file)] = lidar_file

        # img_shared = np.ndarray(sample_img_array.shape, dtype=sample_img_array.dtype, buffer=img_shm.buf)
        # img_shared[:] = img

        # print(lidar_shared[0])
        # time.sleep(0.1)

    lidar_shm.unlink()
    lidar_shm.close()
    img_shm.unlink()
    img_shm.close()
    exit()
    stream_shm = shared_memory.SharedMemory(name ='img_shm', create=True, size=sample_img_array.nbytes)
    shared_a = np.ndarray(sample_img_array.shape, dtype=sample_img_array.dtype, buffer=stream_shm.buf)

    for i in range(10):
        shared_a[:] = np.zeros((500, 1000, 3))
        time.sleep(1)

    stream_shm.unlink()
    stream_shm.close()
    exit()


    for ind in range(len(image_00_files)):
        img = cv2.imread(image_00_files[ind])
        img = cv2.resize(img, (WIDTH, HEIGHT))
        
        shared_a = np.ndarray(sample_img_array.shape, dtype=sample_img_array.dtype, buffer=stream_shm.buf)
        shared_a[:] = img

        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break

    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("영상 파일을 열 수 없습니다.")
    #     exit()

    # stream_shm = shared_memory.SharedMemory(name ='img_shm', create=True, size=sample_img_array.nbytes)
    # while 1:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     frame = cv2.resize(frame, (WIDTH, HEIGHT))

    #     shared_a = np.ndarray(frame.shape, dtype=frame.dtype, buffer=stream_shm.buf)
    #     shared_a[:] = frame

    #     cv2.imshow("frame", frame)
    #     if cv2.waitKey(1) == ord("q"):
    #         break

    
    stream_shm.unlink()
    stream_shm.close()
    # cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    import time
    # p = Process(target=show_img, args=("img_shm", "img1"))
    # p.start()
    p = Process(target=show_3d_points, args=("lidar_shm", "lidar"))
    p.start()

    main_process = Process(target=main)
    main_process.start()

    