import pandas as pd
import numpy as np
import os
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


# interpolate pose at missing timestamp from two neighbouring poses
def interpolate_tf(ts0, ts1, pose0, pose1, ts):
    def lerp(v0, v1, t):
        return (1 - t) * v0 + t * v1

    # normalize timestamp s.t. ts is in [0, 1]
    t = (ts - ts0) / (ts1 - ts0)

    # split poses in translation and rotation part
    # Quaternion uses (w, x, y, z)
    # Rot uses (x, y, z, w)
    v0 = pose0[:3]
    q0 = Quaternion(matrix=R.from_quat(pose0[3:]).as_matrix())
    v1 = pose1[:3]
    q1 = Quaternion(matrix=R.from_quat(pose1[3:]).as_matrix())

    q = Quaternion.slerp(q0, q1, t).elements
    v = lerp(v0, v1, t)
    return np.append(np.append(v, q[1:]), q[0])


newer_college = False
maplab = True
# root = '/media/dominic/Extreme SSD/datasets/new_college/long_experiment/'
# root = '/media/dominic/Extreme SSD/datasets/asl_koze/data_both_side/'
root = '/media/dominic/Extreme SSD/datasets/lidarmace_data/ethz_outside/'
root = '/media/dominic/Extreme SSD/datasets/lidarmace_data/lee_terrace/'
if newer_college:
    path = root + 'registered_poses_xyz_smoothed.csv'
    poses = pd.read_csv(path, header=0)

    xyz = poses[['x', 'y', 'z']].to_numpy()[:]
    quat = poses[['qx', 'qy', 'qz', 'qw']].to_numpy()[:]
    t = poses[['sec', 'nsec']].to_numpy()[:]
    ts = (t[:, 0] * 1e9 + t[:, 1])[..., np.newaxis]

if maplab:
    path = root + 'vertices.csv'
    traje_info = pd.read_csv(path, header=0)
    ts_gt = traje_info[[' timestamp [ns]']].to_numpy()[:]

    # interpolate timestamps to fit timestamps where lidar data was recorded)
    ts_est = np.array(os.listdir(root+"/data")[3:-5:1]).astype(int)
    idx = []
    for ts in ts_est:
        i = np.argmin(np.abs(ts_gt - ts))
        idx.append(i)
    idx = np.array(idx)
    ts = ts_est[..., np.newaxis]
    xyz = traje_info[[' position x [m]', ' position y [m]', ' position z [m]']].to_numpy()
    quat = traje_info[[' quaternion x', ' quaternion y', ' quaternion z', ' quaternion w']].to_numpy()

    xyz_t = np.zeros((ts.shape[0], 3))
    quat_t = np.zeros((ts.shape[0], 4))
    for k in range(idx.shape[0]):
        if ts[k] < ts_gt[idx[k]]:
            pose = interpolate_tf(ts_gt[idx[k]-1], ts_gt[idx[k]],
                                  np.append(xyz[idx[k]-1], quat[idx[k]-1]), np.append(xyz[idx[k]], quat[idx[k]]),
                                  ts[k])
        else:
            pose = interpolate_tf(ts_gt[idx[k]], ts_gt[idx[k] + 1],
                                  np.append(xyz[idx[k]], quat[idx[k]]), np.append(xyz[idx[k] + 1], quat[idx[k] + 1]),
                                  ts[k])
        xyz_t[k, :] = pose[:3]
        quat_t[k, :] = pose[3:]
    xyz = xyz_t
    quat = quat_t

# save formatted poses
formatted_poses = pd.DataFrame(data=np.concatenate((ts, xyz, quat), axis=1), columns=['ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
formatted_poses['ts'] = formatted_poses['ts'].apply(lambda x: '%.0f' % x).values.tolist()
formatted_poses.to_csv(root + "stamped_groundtruth.txt", index=False, sep=" ", header=['#ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
print(formatted_poses)