from PIL import Image
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def project_point_cloud(points, height=64., width=1024., lidar_type="OS1"):
    """ Projects 3D points from a 360° horizontal scan to a 2D image plane.
    Args:
        points: (np array)
            The numpy array containing the point cloud. .
            The shape should be at least Nx3 (allowing for more columns)
            - Where N is the number of points, and
            - each point is specified by at least 3 values (x, y, z)
        height: (int)
            resulting 2D scan image height
        width: (int)
            resulting 2D scan image width
	lidar_type: (str)
	    Type of used LiDAR
    Returns:
        idx:
            2D image indices of the projected 3D points
    """
    # Set lidar intrinsics for chosen type:
    if lidar_type == "OS1":
        fov_up = 16.611 * pi / 180
        fov_down = 16.611 * pi / 180.
    elif lidar_type == "OS0":
        fov_up = 45.0 * pi / 180
        fov_down = 45.0 * pi / 180.
        points[:, 0] = -points[:, 0]
        points[:, 1] = -points[:, 1]
    else:
        raise ValueError('Invalid LiDAR type: "%s"' % lidar_type)
    # Projecting to 2D
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    r = np.sqrt(x_points ** 2 + y_points ** 2, z_points ** 2) + 1e-6 # distance to origin

    fov_up = abs(fov_up)
    fov_down = abs(fov_down)
    fov = fov_up + fov_down

    pitch = np.arcsin(np.clip(z_points/r, -1, 1))
    yaw = np.arctan2(y_points, -x_points)
    u = 1.0 - (pitch + fov_down)/fov
    v = 0.5 * (yaw/pi + 1.0)

    x_img = (v * width).squeeze()
    x_img[x_img < 0] = x_img[x_img < 0] + width
    x_img[x_img >= width] = x_img[x_img >= width] - width
    y_img = (u * height).squeeze()

    idx = (x_img, y_img)
    return idx


def get_homog_matrix(*arg):
    """ get homogeneous matrix from rotational matrix and translation vector.
    Args:
        2 args:
            R: rotation matrix as 3x3 numpy array
            t: translation vector as 3x1 numpy array
        1 arg:
            pose: [x, y, z, qx, qy, qz, qw]
    Returns:
        m: homogeneous matrix as 4x4 numpy array
    """
    if len(arg) == 2:
        r = arg[0]
        t = arg[1]
        m = np.eye(4)
        m[0:3, 3:] = t
        m[0:3, 0:3] = r

    if len(arg) == 1:
        pose = arg[0]
        m = np.eye(4)
        rot = R.from_quat(pose[3:])
        for i in range(3):
            m[i, 3] = pose[i]
        m[0:3, 0:3] = rot.as_matrix()

    return m


def transform_point_cloud(xyz, tf):
    """ Transform xyz from frame1 to frame2.
        Args:
            xyz:  Nx3 numpy array of (x, y, z)-coordinates in frame1
            tf: homogeneous transformation matrix from frame1 to frame2: xyz_t = tf * xyz
        Returns:
            xyz_t: transformed (x, y, z)-coordinates to frame2 as Nx3 numpy array
        """
    xyz = np.append(xyz, np.ones_like(xyz[:, 0:1]), axis=1)
    xyz_t = np.dot(tf, xyz.T).T
    return xyz_t[:, :3]


def viz_point_cloud(xyz, intensity=None, fig=None):
    """ Visualize point cloud.
    Args:
        xyz: Nx3 numpy array of (x, y, z)-coordinates to be displayed
        intensity: list of N intensities corresponding to 3d points in range [0,1]
        fig: If given, point cloud is drawn in the first subplot of this figure
    """
    if intensity is None:
        intensity = np.ones((xyz.shape[0], 3)) * 0.5
    if isinstance(intensity, (int, int, int)):
        intensity = np.ones((xyz.shape[0], 3)) * np.array(intensity)
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.get_axes()[0]

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=intensity, s=0.1)
    ax.view_init(30, 180)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.autoscale()
