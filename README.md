# lidar to camera calibration with KITTI dataset


# Dataset

1. Visit [KITTI dataset website](https://www.cvlibs.net/datasets/kitti/raw_data.php) and login
2. Download object tracking or any sequence dataset (Recommend Raw data tab)

### Dataset tree
```text
kitti_lidar_to_camera_calibration
├─ data
│  └─ kitti_sequence05
│     ├─ image_00
│     │  ├─ data
│     │  │  ├─ 0000000000.png
│     │  │  └─ ...
│     │  └─ timestamps.txt
│     ├─ image_01
│     │  ├─ data
│     │  │  ├─ 0000000000.png
│     │  │  └─ ...
│     │  └─ timestamps.txt
│     ├─ image_02
│     │  ├─ data
│     │  │  ├─ 0000000000.png
│     │  │  └─ ...
│     │  └─ timestamps.txt
│     ├─ image_03
│     │  ├─ data
│     │  │  ├─ 0000000000.png
│     │  │  └─ ...
│     │  └─ timestamps.txt
│     ├─ label_02
│     │  ├─ 0000.txt
│     │  └─ ...
│     ├─ oxts
│     │  ├─ data
│     │  │  ├─ 0000000000.txt
│     │  │  └─ ...
│     │  ├─ dataformat.txt
│     │  └─ timestamps.txt
│     └─ velodyne_points
│        ├─ data
│        │  ├─ 0000000000.bin
│        │  └─ ...
│        ├─ timestamps.txt
│        ├─ timestamps_end.txt
│        └─ timestamps_start.txt
└─ ...
```

