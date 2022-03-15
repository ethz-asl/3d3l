# R2D2 network for LiDAR scan data
Final version of adapted R2D2 network for LiDAR scans. Includes Training pair generation, keypoint extraction, matching, registration and pose estimation algrithms.

### Organize LiDAR data
 The LiDAR scan images need to be in .npy file format in a folder with the corresponding timestamp as name of the folder. The folder must include a range image (heightxwidthx1) named range.npy, a intensity image (heightxwidthx1) named intensity.npy, the 3D coordinates (heightxwidthx3) named xyz.npy and the mask of valid points (heightxwidthx1) named valid_mask.npy. The range and intensity images are both normalized with the datasets mean and variance. All the timestamp folders are in a folder named data. Otherwise, the lidar_dataset (in datasets/lidar_dataset.py) class will not work. Moreover, a file with ground truth poses needs to be located outside the data folder. 
 ```
    dataset
    ├── data
    │   ├── 1308069532730
    │   │   ├── intensity.npy
    │   │   ├── range.npy
    │   │   ├── valid_mask.npy
    │   │   └── xyz.npy
    │   ├── ...
    │   └── 1311168731710
    │       ├── intensity.npy
    │       ├── range.npy
    │       ├── valid_mask.npy
    │       └── xyz.npy
    └── vertices.csv
```

With the file tools/manage_trajectory_files.py the ground truth poses of the dataset is adapted to the format used by the code and may need to be adapted to the specific format of the used ground truth poses file.
The file tools/lidar_pairs_from_pose.py generated a file with timestamps and transformations between those two poses for training the network with true scan pairs.
  
### Train network
The network can be trained with the train.py file. It takes multiple arguments, most importantly the weight save path wit --save-path "path_to_model_weights". Other arguments can be found in the code. The standard arguments are as described in the thesis. The path to the datasets has to be adapted before training.
  
### Extract key points and descriptors
The file extract.py offers a method called extract keypoints, which returns the pixel locations of the 2D scan image of the extracted keypoints, the corresponding scores and descriptors. The method takes the 2-channel scan image and a list of arguments as input. Running the extract.py file standalone has standard parameters for those arguments and uses the lidar_dataset class to load a scan image.
 
### Matching
The matching.py file offers a visualization of two matched point clouds using the extract method and the lidar_dataset class. It uses predefined arguments which can be changed in the code, mostly the root of the lidar dataset folder.
 
### Global registration
The global registration algorithm is run by the global_registraion.py file. All the parameters need to be adapted in the code, i.e the path to the dataset.
