import numpy as np
import pandas as pd
import random
import os
from PIL import Image, ImageDraw
import psutil
import time
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from scipy.spatial import KDTree


def get_homog_matrix(pose):
    """
    Transforms a pose to a homogeneous matrix
    :param pose: [x, y, z, qx, qy, qz, qw]
    :return: 4x4 homogeneous matrix
    """
    m = np.eye(4)
    rot = R.from_quat(pose[3:])
    for i in range(3):
        m[i, 3] = pose[i]
    m[0:3, 0:3] = rot.as_matrix()
    return m

# get pose x, y, z, tx, ty, tz, tw from matrix m
def get_pose_from_homog(m):
    pose = np.zeros((7, ))
    pose[0:3] = m[0:3, 3].squeeze()
    pose[3:] = R.from_matrix(m[:3, :3]).as_quat()
    return pose

# load scan image from lidar data
def get_image(root, ts):
        folder_path = os.path.join(root, ts)
        range = np.load(folder_path + '/range.npy')
        reflectivity = np.load(folder_path + '/reflectivity.npy')
        intensity = np.load(folder_path + '/intensity.npy')
        # intensity = intensity.point(lambda i: i * 20)
        # img = Image.merge('RGB', (range, reflectivity, intensity))
        img = Image.fromarray(((intensity / 8. + 1.) / 2. * 255).astype(np.uint8)).convert('RGB')
        w, h = img.size
        return img

# draw two unaligne point clouds after aligning them with transformation
def draw_registration_result(source, target, transformation):
    import copy
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

# get transformation matrix from pose (x, y, z, tx, ty, tz, tw)
# get_homog_matrix(pose2) = tf * get_homog_matrix(pose1)
def get_tf(pose1, pose2, tf_base_os):
    m1 = get_homog_matrix(pose1)
    m2 = get_homog_matrix(pose2)
    m1 = np.dot(m1, tf_base_os)
    m2 = np.dot(m2, tf_base_os)
    tf = np.dot(np.linalg.inv(m2), m1)
    return tf

# load point cloud
def get_xyz(path):
    xyz = np.load(path + '/xyz.npy')
    return xyz

# load mask of valid lidar points
def get_valid_range_mask(path):
    mask = np.load(path + '/valid_mask.npy')
    return mask

# visualization helper: get line set from poses (trajectory)
def get_line_set_from_poses(poses, color):
    lines = np.arange(poses.shape[0] - 1)[..., np.newaxis]
    lines = np.concatenate((lines, lines + 1), axis=1)
    lines.astype(int)
    colors = [color for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(poses)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


# check if two point clouds (same frame) overlap by at least overlap_thresh
def check_overlap(pcA, pcB, corr_dist=0.4, overlap_thresh=0.2):
    tree = KDTree(pcA)
    neighbour_dist, neighbour_idx = KDTree.query(tree, pcB, 1, distance_upper_bound=corr_dist)

    inlier_B_mask = neighbour_dist < corr_dist
    inlier_A_mask = np.zeros(pcA.shape[0], dtype=bool)
    inlier_A_mask[neighbour_idx[inlier_B_mask]] = True
    return inlier_A_mask.sum()/pcA.shape[0] >= overlap_thresh

# define lidar camer extrinsics
# newer college
tf_base_baseMount = np.hstack(([-0.08425, -0.025, -0.06325], R.from_euler('xyz', [0, -0, 0]).as_quat()))
tf_baseMount_os1Sensor = np.hstack(([0, 0, 0.077258], R.from_euler('xyz', [0, 0, -0.785398]).as_quat()))
tf_os1Sensor_os1Lidar = np.hstack(([0, 0, 0.03618], R.from_euler('xyz', [0, -0, -3.14159]).as_quat()))
tf_base_os1Lidar = np.dot(np.dot(get_homog_matrix(tf_base_baseMount), get_homog_matrix(tf_baseMount_os1Sensor)),
                          get_homog_matrix(tf_os1Sensor_os1Lidar))

# lidarstick 150 os1
# tf_base_os1Lidar = np.array([[-0.6978447576305695, 0.7161518144580716, 0.01180139381358678, -0.1021019447103219],
#                              [0.7160088243900917, 0.697942707663664, -0.01439931299167545, -0.08936602296234536],
#                              [-0.01854879087839803, -0.001598582974091693, -0.999826678424528, -0.07460082421278309],
#                              [0, 0, 0, 1]])
# tf_base_os1Lidar = np.dot(tf_base_os1Lidar, np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

# lidarmace
# tf_base_os1Lidar = np.array([[0.9997770641962381,  0.016299234847643,  -0.01344410000000, 0.0331989],
#                              [-0.016286738363819,   0.999867935740224,  0.001094290000000, -0.121313],
#                              [0.01346018117580,   -0.00087509685356,   0.999909000000000, 0.15737],
#                              [0, 0, 0, 1]])

# lidar data path
root = '/media/dominic/Extreme SSD/datasets/new_college/long_experiment/'
# ground truth pose file path
path = 'registered_poses_xyz_smoothed.csv'
# root = '/media/dominic/Extreme SSD/datasets/asl_koze/data_both_side/'
# path = 'stamped_groundtruth_all.txt'
# root = '/media/dominic/Extreme SSD/datasets/lidarmace_data/ethz_outside/'
# path = 'stamped_groundtruth.txt'

poses = pd.read_csv(root + path, header=0, sep=",")
icp = True
overlap = True
xyz = poses[['x', 'y', 'z']].to_numpy()[:]
quat = poses[['qx', 'qy', 'qz', 'qw']].to_numpy()[:]
t = poses[['sec', 'nsec']].to_numpy()[:]
ts = t[:, 0] * 1e9 + t[:, 1]
# ts = poses[['#ts']].to_numpy()[:].squeeze()[:]
line_set1 = get_line_set_from_poses(xyz, [0, 1, 0])
# get ts and transformation of anchor-sample pairs
anchor = 0
pairs = {'ts1': [], 'ts2': [], 'x': [], 'y': [], 'z': [], 'qx': [], 'qy': [], 'qz': [], 'qw': []}
while True:
    print(len(pairs['x']))
    anchor_coords = xyz[anchor, :]
    # distances to current anchor
    d = np.linalg.norm(xyz - anchor_coords, axis=1)
    # list of possible neighbours
    min_r = 3
    max_r = 9
    pn = np.nonzero((min_r < d) * (d < max_r))[0]
    # select random sample of neighbours
    i = random.randint(0, pn.shape[0]-1)
    sample = pn[i]
    # get corresponding timestamps
    ts1 = format(ts[anchor], '.0f')
    ts2 = format(ts[sample], '.0f')
    # get corresponding transformation between poses
    pose1 = np.hstack((xyz[anchor, :], quat[anchor, :]))
    pose2 = np.hstack((xyz[sample, :], quat[sample, :]))
    m = get_tf(pose1, pose2, tf_base_os1Lidar)

    # run icp for better relative transformation
    if icp:
        # run icp
        maskA = get_valid_range_mask(root + 'data/' + ts1)
        maskB = get_valid_range_mask(root + 'data/' + ts2)
        xyzA = get_xyz(root + '/data/' + ts1)[maskA].reshape(-1, 3)
        xyzB = get_xyz(root + '/data/' + ts2)[maskB].reshape(-1, 3)
        pcdA = o3d.geometry.PointCloud()
        pcdA.points = o3d.utility.Vector3dVector(xyzA)
        pcdB = o3d.geometry.PointCloud()
        pcdB.points = o3d.utility.Vector3dVector(xyzB)
        reg_p2p = o3d.registration.registration_icp(
                    pcdA, pcdB, 0.25, m,
                    o3d.registration.TransformationEstimationPointToPoint(),
                    o3d.registration.ICPConvergenceCriteria(max_iteration=100))
        m = reg_p2p.transformation

        if overlap:
            pcdA = pcdA.transform(m)
            if not check_overlap(np.asarray(pcdA.points), np.asarray(pcdB.points)):
                anchor += 1
                if anchor >= ts.shape[0]:
                    break
                continue

    # add pose to dataframe
    pairs['ts1'].append(ts1)
    pairs['ts2'].append(ts2)
    pose = get_pose_from_homog(m)
    pairs['x'].append(pose[0])
    pairs['y'].append(pose[1])
    pairs['z'].append(pose[2])
    pairs['qx'].append(pose[3])
    pairs['qy'].append(pose[4])
    pairs['qz'].append(pose[5])
    pairs['qw'].append(pose[6])

    # add next anchor point according to number of samples
    break
    anchor += 2
    if anchor >= ts.shape[0]:
        break
pairs_df = pd.DataFrame.from_dict(pairs)
pairs_df.to_csv(root + "/tf.csv", index_label="id")