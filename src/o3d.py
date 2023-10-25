import open3d as o3d
import numpy as np
import os

class O3D:
    def __init__(self, width, height, init_points, calib=None) -> None:
        if calib is not None:
            self.calib = self.load_calib(calib)
        else:
            self.calib = calib
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=width, height=height, visible=True)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 3
        self.ctr = self.vis.get_view_control()
        
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(init_points)
        # pcd.points = o3d.utility.Vector3dVector(load_lidar_file(lidar_files[0])[:, :3])
        self.vis.add_geometry(self.pcd)


    def updata_pcd(self, pcd):
        self.pcd.points = o3d.utility.Vector3dVector(pcd)
        self.vis.update_geometry(self.pcd)
        if self.calib is not None:
            self.ctr.convert_from_pinhole_camera_parameters(self.calib)
        keep_leaning = self.vis.poll_events()
        self.vis.update_renderer()
        
        # blocking
        # self.vis.run()

    def save_file(self, ind):
        # save files
        o3d_save_path = os.path.join("./outputs/o3d", f"{ind}.png")
        if not os.path.exists(os.path.dirname(o3d_save_path)):
            os.makedirs(os.path.dirname(o3d_save_path))
        self.vis.capture_screen_image(o3d_save_path)

    
    @staticmethod
    def load_calib(path):
        return o3d.io.read_pinhole_camera_parameters(path)

    def __del__(self):
        o3d.destroy_window()
