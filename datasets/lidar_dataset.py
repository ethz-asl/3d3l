import numpy as np
import os
from PIL import Image
import random
import pandas as pd
from scipy.spatial.transform import Rotation as R
import cv2
import math
from scipy.signal import convolve2d
from scipy.spatial import KDTree

from datasets.dataset import Dataset
from tools import point_cloud_utils as pcu

import matplotlib.pyplot as plt


class LidarBase (Dataset):
    def __init__(self, crop=True, type='OS1'):
        Dataset.__init__(self)
        self.crop = crop
        self.type = type
        if self.type == 'OS1':
            self.crop_size = 256
            self.H = 64
        if self.type == 'OS0':
            self.crop_size = 250
            self.H = 128

    def open_lidar_image(self, folder_path):
        # convert to image
        def img_conv(m):
            im = ((m / 8. + 1.) / 2. * 255).astype(np.uint8)
            return im
        rang = img_conv(self.open_img(folder_path + '/range.npy'))
        refl = img_conv(self.open_img(folder_path + '/reflectivity.npy'))
        inte = img_conv(self.open_img(folder_path + '/intensity.npy'))

        img = np.stack((rang, inte, refl), axis=2)
        img = Image.fromarray(img, mode='RGB')

        return img

    @staticmethod
    def open_img(path):
        return np.load(path)


class LidarSynthetic(LidarBase):
    """ load all images from lidar db without transformation. Used for synthetic pairs
    """
    def __init__(self, root, skip=(0, -1, 1), crop=True, type='OS1'):
        LidarBase.__init__(self, crop, type)
        self.img_folders = os.listdir(root)[skip[0]: skip[1]: skip[2]]
        self.img_folders.sort()
        self.nimg = len(self.img_folders)
        self.root = root

    def get_image(self, img_idx):
        folder_path = os.path.join(self.root, self.img_folders[img_idx])
        img = self.open_lidar_image(folder_path)
        # img = np.array(img)
        # img[~self.get_valid_range_mask(img_idx), :] = 255
        # img = Image.fromarray(img)
        # crop image to avoid statically undefined areas
        if self.type == 'OS0' and self.crop:
            regions = np.array([[91, 351], [604, 864]])
            region = int(random.random() < 0.5)
            w, h = img.size
            shape = (regions[region, 0], 0, regions[region, 1], h)
            img = img.crop(shape)
        w, h = img.size
        if self.crop:
            x = random.randint(0, w - self.crop_size)
            img = img.crop((x, 0, x + self.crop_size, h))
        return img

    def get_xyz(self, idx):
        folder_path = os.path.join(self.root, self.img_folders[idx])
        xyz = np.load(folder_path + '/xyz.npy')
        return xyz

    def get_valid_range_mask(self, idx):
        folder_path = os.path.join(self.root, self.img_folders[idx])
        mask = np.load(folder_path + '/valid_mask.npy')
        return mask

    def get_gt_pose(self, idx):
        ts = self.img_folders[idx]
        ts = int(ts)
        sec = math.floor(ts * 1e-9)
        nsec = int(ts - sec * 1e9)
        ts_string = str(sec) + ',' + str(nsec)
        path = os.path.join(self.root[:-10]+'registered_poses_xyz_smoothed.csv')
        with open(path, 'r') as f:
            for line in f:
                # For each line, check if line contains the string
                if ts_string in line:
                    string = line
        pose = string.split(',')[3:]
        pose[-1] = pose[-1][:-1]
        pose = np.array(pose).astype(float)
        return pose


class LidarPairDataset(LidarBase):
    def __init__(self, root, skip=(0, -1, 1), crop=False, reproject=True, type="OS1"):
        LidarBase.__init__(self, crop, type=type)
        self.root = root
        self.pair_dict = pd.read_csv(os.path.join(root, 'tf.csv'), header=0)[skip[0]:skip[1]:skip[2]].reset_index(drop=True)
        self.npairs = self.pair_dict.shape[0]
        self.crop_size = 150
        self.reproject = reproject

    def __len__(self):
        self.npairs = self.pair_dict.shape[0]
        return self.npairs

    def open_lidar_image_projected(self, folder_path):
        img = self.open_img(folder_path + '/scan_combined_projected.npy')
        mask = np.load(folder_path + '/valid_mask_proj.npy')
        return img, mask

    def get_pair(self, idx, output=()):
        """ returns (img1, img2, `metadata`)
        """
        img1 = np.array(self.open_lidar_image(os.path.join(
            self.root, 'data', format(self.pair_dict['ts1'][idx], '.0f'))))
        mask1 = np.load(os.path.join(
            self.root, 'data', format(self.pair_dict['ts1'][idx], '.0f'), 'valid_mask.npy')).reshape(-1)

        if self.reproject:
            try:
                img2, mask2 = self.open_lidar_image_projected(
                    os.path.join(self.root, 'data', format(self.pair_dict['ts2'][idx], '.0f')))
                project_image = False
                mask2 = mask2.reshape(-1)
            except:
                project_image = True
                img2 = np.array(self.open_lidar_image(
                    os.path.join(self.root, 'data', format(self.pair_dict['ts2'][idx], '.0f'))))
                mask2 = np.load(os.path.join(
                    self.root, 'data', format(self.pair_dict['ts2'][idx], '.0f'), 'valid_mask.npy')).reshape(-1)
        else:
            project_image = False
            img2 = np.array(self.open_lidar_image(
                os.path.join(self.root, 'data', format(self.pair_dict['ts2'][idx], '.0f'))))
            mask2 = np.load(os.path.join(
                self.root, 'data', format(self.pair_dict['ts2'][idx], '.0f'), 'valid_mask.npy')).reshape(-1)

        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        img1 = img1.reshape(-1, 3)
        img2 = img2.reshape(-1, 3)

        img1[np.invert(mask1), :] = 0
        img2[np.invert(mask2), :] = 0

        if project_image:
            # project image 2 in frame 2
            xyz2 = np.load(
                os.path.join(self.root, 'data', format(self.pair_dict['ts2'][idx], '.0f'), 'xyz.npy')).reshape(-1, 3)
            h2 = int(h2 * 1.0)
            w2 = int(w2 * 1.0)
            proj_idx = pcu.project_point_cloud(xyz2, height=h2, width=w2, lidar_type=self.type)
            x_img = proj_idx[0].astype(int)
            y_img = proj_idx[1].astype(int)

            # use projection indices to project image2 to frame2
            in_bound_mask = np.logical_and(y_img >= 0, y_img < h2)
            img2_proj = np.zeros((h2, w2, 3), dtype=np.uint8)

            img2_proj[y_img[in_bound_mask], x_img[in_bound_mask], :] = img2[in_bound_mask, :]

            mask2_proj = np.zeros((h2, w2), dtype=bool)
            mask2_proj[y_img[in_bound_mask], x_img[in_bound_mask]] = mask2[in_bound_mask]
            mask2_proj = mask2_proj.reshape((h2, w2))

            img2_proj, mask2_proj = self.correct_missing_pixels_in_projected_scan(img2_proj, mask2_proj)

            # save projected image
            np.save(os.path.join(self.root, 'data',
                                 format(self.pair_dict['ts2'][idx], '.0f'), "scan_combined_projected.npy"), img2_proj)
            np.save(os.path.join(self.root, 'data',
                                 format(self.pair_dict['ts2'][idx], '.0f'), "valid_mask_proj.npy"), mask2_proj)
        else:
            # print("use saved image")
            img2_proj = img2
            mask2_proj = mask2.reshape((h2, w2))

        # reshape masks of valid pixels to image sizes
        mask1 = mask1.reshape(h1, w1)

        # get flow according to pair index and reprojected mask2
        flow, mask_valid_in_2 = self.get_pixel_match(idx, mask2_proj)

        # redefine flow mask with invalid pixels in image1 and maks generated by finding flow
        mask = (mask1 * mask_valid_in_2).astype(bool)

        # set flow for invalid pixels to nan, which is ignored during training
        flow[~mask, :] = np.nan

        # crop image
        img2 = Image.fromarray(img2_proj.reshape(h1, w1, 3))
        img1 = Image.fromarray(img1.reshape(h1, w1, 3))

        if self.crop:
            # find crop window which has enough valid points based on flow mask
            x_crop = self.find_best_window_offset(mask, thr=0.4, win_size=(self.H, self.crop_size))
            # set flow outside crop region out of bound
            flow_crop = flow
            flow_crop[:, 0:x_crop, :] = -1
            flow_crop[:, x_crop + self.crop_size:, :] = -1
            x_img = flow_crop[:, :, 0].reshape(-1).astype(int)
            y_img = flow_crop[:, :, 1].reshape(-1).astype(int)

            # find pixels which are out of image2 bounds
            mask_x = (x_img > 0) * (x_img < w2)
            mask_y = (y_img > 0) * (y_img < h2)
            mask_oob = mask_x * mask_y

            # find crop region 2 where enough of the flow is in image2
            mask_y = np.zeros((h2, w2), dtype=np.uint8)
            mask_y[y_img[mask_oob], x_img[mask_oob]] = 1
            # Image.fromarray(mask_y.astype(np.uint8)*255).show()
            x_crop2 = self.find_best_window_offset(mask_y, win_size=(h2, self.crop_size), thr=0.2)

            # crop flow according to crop region
            flow = flow_crop[:, x_crop:x_crop + self.crop_size, :]
            flow[:, :, 0] = flow[:, :, 0] - x_crop2
            # crop flow mask according to crop region
            mask = mask[:, x_crop:x_crop + self.crop_size]
            # crop images with found regions
            img1 = img1.crop((x_crop, 0, x_crop + self.crop_size, h1))
            img2 = img2.crop((x_crop2, 0, x_crop2 + self.crop_size, h2))

        meta = {'aflow': flow, 'mask': mask}
        return img1, img2, meta

    def get_pixel_match(self, idx, mask2):
        # get position vectors for first lidar scan
        xyz1 = np.load(os.path.join(self.root, 'data', format(self.pair_dict['ts1'][idx], '.0f'), 'xyz.npy'))
        # get position vectors for second lidar scan
        xyz2 = np.load(os.path.join(self.root, 'data', format(self.pair_dict['ts2'][idx], '.0f'), 'xyz.npy'))
        # get relative transformation between img1 and img2
        pose = self.pair_dict.loc[idx, ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].to_numpy()
        tf = self.get_homog_matrix(pose)
        # Make position vectors homogeneous
        pos1 = np.append(xyz1, np.ones_like(xyz1[:, :, 0:1]), axis=2)
        h1, w1, _ = pos1.shape
        h2, w2 = mask2.shape
        # transform all position vectors from img1 to frame2
        pos1 = pos1.reshape((h1*w1, 4)).transpose((1, 0))
        xyz_t = np.dot(tf, pos1).transpose((1, 0))[:, :3]

        # get indices of pixel correspondences
        proj_idx_1 = pcu.project_point_cloud(xyz_t, height=h2, width=w2, lidar_type=self.type)
        proj_idx_1 = np.array(proj_idx_1)
        # create flow matrix needed for training from pixel correspondences
        flow = np.zeros((h1, w1, 2))
        x_img = proj_idx_1[0, :].reshape(h1, w1)
        y_img = proj_idx_1[1, :].reshape(h1, w1)
        flow[:, :, 0] = x_img
        flow[:, :, 1] = y_img

        # get mask of which pixels in image1 are valid in image2 (out of bounds and invalid defined by mask2)
        x_img = flow[:, :, 0].reshape(-1).astype(int)
        y_img = flow[:, :, 1].reshape(-1).astype(int)
        mask_valid_in_2 = np.zeros(h1*w1, dtype=bool)
        mask_in_bound = (y_img >= 0) * (y_img < h2) * (x_img >= 0) * (x_img < w2)
        mask_valid_in_2[mask_in_bound] = mask2[y_img[mask_in_bound], x_img[mask_in_bound]]

        # get ranges of valid image2 points
        r2 = np.linalg.norm(xyz2[y_img[mask_valid_in_2], x_img[mask_valid_in_2]], axis=1)
        # get corresponding transformed ranges of image1
        r1_t = np.linalg.norm(xyz_t.reshape(-1, 3)[mask_valid_in_2], axis=1)

        # check if points in image1 are occluded in 2
        d = (r2 - r1_t)/r1_t
        # 20% threshold when close to each other
        not_occluded = d > -0.2
        # define occlusion mask
        occlusion_mask = np.ones((h1 * w1), dtype=bool)
        occlusion_mask[mask_valid_in_2] = not_occluded

        # set mask_valid_in_2 false when occluded
        mask_valid_in_2 = np.logical_and(mask_valid_in_2, occlusion_mask)
        mask_valid_in_2 = mask_valid_in_2.reshape(h1, w1)

        # mask flow according to occluded points
        flow[:, :, 0][np.invert(mask_valid_in_2)] = np.nan
        flow[:, :, 1][np.invert(mask_valid_in_2)] = np.nan

        return flow, mask_valid_in_2

    @staticmethod
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

    @staticmethod
    def find_best_window_offset(mask, win_size, thr=0.4):
        score = convolve2d(mask, np.ones(win_size), mode='valid')
        score = score.squeeze()/(win_size[0]*win_size[1])
        xs = np.arange(score.shape[0])
        thr_mask = score > thr
        if thr_mask.sum() == 0:
            x = np.argmax(score)
        else:
            x = np.random.choice(xs[thr_mask])
        return x

    @staticmethod
    def correct_missing_pixels_in_projected_scan(scan, mask1):
        mask = mask1.copy()
        mask1 = mask1.reshape(-1)
        height, width, _ = scan.shape

        # generate grid for frame 2
        y, x = np.mgrid[0:height, 0:width]
        x = x.reshape(-1)
        y = y.reshape(-1)

        # refine mask to ignore large undefined regions
        k = 2
        filter = np.ones((k, k))
        mask = (convolve2d((mask), filter, mode='same') == 0).astype(np.uint8)
        mask = 1 - (convolve2d((mask), filter, mode='same') == 0).astype(np.uint8)
        mask = 1 - (1 - mask1.reshape(height, width)) * (1 - mask)
        mask = mask.reshape(-1).astype(bool)

        # define missing pixels after re-projection
        mesh_query = np.array([x[~mask], y[~mask]]).T
        mesh_data = np.array([x[mask], y[mask]]).T

        # find missing indices due to imprecision
        tree = KDTree(mesh_data)
        dist, mesh_idx = KDTree.query(tree, mesh_query, 1, distance_upper_bound=1.1)
        dist_mask = dist <= 1.1
        missing_idx = tree.data[mesh_idx[dist_mask], :]
        scan[mesh_query[dist_mask, 1], mesh_query[dist_mask, 0], :] = scan[missing_idx[:, 1], missing_idx[:, 0], :]

        mask1 = mask1.reshape(height, width)
        mask1[mesh_query[dist_mask, 1], mesh_query[dist_mask, 0]] = True

        return scan, mask1
