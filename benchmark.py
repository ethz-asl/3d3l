import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import tools.point_cloud_utils as pcu
from datasets.lidar_dataset import LidarPairDataset
import matching
import extract
import open3d as o3d
import pandas as pd


def transform_points(pts1, pts2, mask1, mask2, tf):
    """
    Transformes pts1 in frame2 and pts2 in frame, while masking points visible in both scans
    :param pts1: pts1 in frame1
    :param pts2: pts2 in frame2
    :param mask1: mask of valid points in scan1
    :param mask2: mask of valid points in scan2
    :param tf: transformation matrix between frame1 and frame2
    :return: transformed 3D points in both frames, masked by visibility in both scans
    """
    pts1_t = pcu.transform_point_cloud(pts1, tf)
    pts2_t = pcu.transform_point_cloud(pts2, np.linalg.inv(tf))
    img_coords1_t = pcu.project_point_cloud(pts1_t)
    img_coords2_t = pcu.project_point_cloud(pts2_t)

    visible_mask1 = np.logical_and((img_coords1_t[1] >= 0), (img_coords1_t[1] < 64))
    visible_mask2 = np.logical_and((img_coords2_t[1] >= 0), (img_coords2_t[1] < 64))

    temp_mask = np.zeros_like(img_coords1_t[1]).astype(bool)
    temp_mask[visible_mask1] = mask2[img_coords1_t[1][visible_mask1].astype(int), img_coords1_t[0][visible_mask1].astype(int)]
    visible_mask1 = visible_mask1 * temp_mask

    temp_mask = np.zeros_like(img_coords2_t[1]).astype(bool)
    temp_mask[visible_mask2] = mask1[img_coords2_t[1][visible_mask2].astype(int), img_coords2_t[0][visible_mask2].astype(int)]
    visible_mask2 = visible_mask2 * temp_mask

    return pts1_t, visible_mask1, pts2_t, visible_mask2


def corr(x1, x2, beta):
    """
    Calculate number of correct points x2 compared to the points x1 (already transformed to image2) and their
    average distance. A point x1[i] is correct if the distance to the closest point x2[j] is at most beta pixels.
    :param x1: points1
    :param x2: points2 transformed to image1 (pseudo ground truth)
    :return: number of correct points1 and their average distance to the transformed points2
    """
    # define mask which points do correspond correctly
    corr_mask = np.zeros(x1.shape[0], dtype=bool)
    # calculate correspondences
    corr = 0
    dist = 0
    for i in range(0, x1.shape[0]):
        d = np.min(np.linalg.norm(x2 - x1[i], axis=1))
        if d < beta:
            corr += 1
            dist += d
            corr_mask[i] = True
    if corr != 0:
        avg_corr_dist = dist/corr
    else:
        avg_corr_dist = beta
    return corr, avg_corr_dist, corr_mask


def interest_point_score(pts1, pts2, mask1, mask2, tf_true, beta):
    """
    Calculates the repeatability score and localization error of two sets of points.
    :param pts1: interest points in image1 (N x 3)
    :param pts2: interest points in image2 (M x 3)
    :param mask1: mask of valid points in scan1
    :param mask2: mask of valid points in scan2
    :param tf_true: ground truth transformation between two scans
    :param beta: pixel error which is considered true positive
    :return: repeatability score, localization error
    """
    # transform points in other image
    pts1_t, mask_shared1, pts2_t, mask_shared2 = transform_points(pts1, pts2, mask1, mask2, tf_true)
    pts1 = pts1[mask_shared1, :]
    pts2 = pts2[mask_shared2, :]
    pts1_t = pts1_t[mask_shared1, :]
    pts2_t = pts2_t[mask_shared2, :]
    # calculate how many points are inside view of other image
    N1 = pts1.shape[0]
    N2 = pts2.shape[0]
    if N1 == 0 or N2 == 0:
        return 0, beta
    # calculate correct points
    corr1, LE1, _ = corr(pts1, pts2_t, beta)
    corr2, LE2, _ = corr(pts2, pts1_t, beta)
    return (corr1 + corr2) / (N1 + N2), (LE1 + LE2) / 2


def descriptor_score(pts1, pts2, desc1, desc2, xyz1, xyz2, tf_true, beta, icp, icp_dist, show):
    """
    Calculates the matching score and the transformation accuracy
    :param pts1: pts1 in image1 (N x 3)
    :param pts2: pts2 in image2 (N x 3)
    :param h_true: true homography between images 1 and 2
    :param h_pred: predicted homography between images 1 and 2
    :return: matching score, transfromation error, matches, correct matches mask, estimated transformation matrix
    """
    reference_pc = o3d.geometry.PointCloud()
    reference_pc.points = o3d.utility.Vector3dVector(xyz1)

    test_pc = o3d.geometry.PointCloud()
    test_pc.points = o3d.utility.Vector3dVector(xyz2)

    ref = o3d.registration.Feature()
    ref.data = desc1.T

    test = o3d.registration.Feature()
    test.data = desc2.T

    ref_key = o3d.geometry.PointCloud()
    ref_key.points = o3d.utility.Vector3dVector(pts1)
    test_key = o3d.geometry.PointCloud()
    test_key.points = o3d.utility.Vector3dVector(pts2)

    result_ransac = matching.execute_global_registration(ref_key, test_key, ref, test, 0.5)

    tf_pred = result_ransac.transformation
    matches = np.array(result_ransac.correspondence_set)

    # get matched points and transformed to each frame with gt tf
    import copy
    pts_matched1 = np.asarray(ref_key.points)[matches[:, 0], :]
    ref_key_tmp = copy.deepcopy(ref_key)
    ref_key_tmp.transform(tf_true)
    pts_matched1_t = np.asarray(ref_key_tmp.points)[matches[:, 0], :]
    pts_matched2 = np.asarray(test_key.points)[matches[:, 1], :]
    test_key_tmp = copy.deepcopy(test_key)
    test_key_tmp.transform(np.linalg.inv(tf_true))
    pts_matched2_t = np.asarray(test_key_tmp.points)[matches[:, 1], :]

    # get total number of matches
    N1 = pts_matched1.shape[0]
    N2 = pts_matched2.shape[0]

    # get correct matches
    x1 = pts_matched1
    x2 = pts_matched2_t
    d1 = np.linalg.norm(x1 - x2, axis=1)
    mask1 = d1 < beta
    corr1 = mask1.sum()
    corr2, _, mask2 = corr(pts_matched2, pts_matched1_t, beta)
    x1 = pts_matched2
    x2 = pts_matched1_t
    d2 = np.linalg.norm(x1 - x2, axis=1)
    mask2 = d2 < beta
    corr2 = mask2.sum()

    MS = (corr1 + corr2) / (N1 + N2)
    corr_mask = np.logical_and(mask1, mask2)

    # use icp to refine transformation
    if icp:
        reg_p2p = o3d.registration.registration_icp(
                        reference_pc, test_pc, icp_dist, tf_pred,
                        o3d.registration.TransformationEstimationPointToPoint(),
                        o3d.registration.ICPConvergenceCriteria(max_iteration=500))
        tf_pred = reg_p2p.transformation

    # Plot point clouds after registration
    if show:
        matching.draw_registration_result(reference_pc, test_pc, tf_true)

    # get transformation error
    t_err = np.linalg.norm(tf_true[:3, 3] - tf_pred[:3, 3])
    R_true = tf_true[:3, :3]
    R_pred = tf_pred[:3, :3]
    R_err = np.linalg.norm(np.dot(R_pred, R_true.T) - np.eye(3)) * 180 / pi
    TE = (t_err, R_err)
    return MS, TE, matches, corr_mask, tf_pred

# Use network to extract features from image pairs
def extract_r2d2_features(imgA, imgB, xyzA, xyzB, args, net):
    xysA, scoresA, descA = extract.extract_keypoints(imgA, args, net)
    xysA, scoresA, descA = matching.remove_invalid_keypoints(xysA, scoresA, descA, maskA)
    kptsA = xyzA[xysA[:, 1].astype(int), xysA[:, 0].astype(int)]
    xysB, scoresB, descB = extract.extract_keypoints(imgB, args, net)
    xysB, scoresB, descB = matching.remove_invalid_keypoints(xysB, scoresB, descB, maskB)
    kptsB = xyzB[xysB[:, 1].astype(int), xysB[:, 0].astype(int)]
    return kptsA, descA, xysA, kptsB, descB, xysB

# load precalculated key points and descriptors from D3Feat net
def get_d_three_features(idx):
    root = "/home/dominic/D3Feat.pytorch/geometric_registration/D3Feat09101407/"
    # print('{:03.0f}'.format(2 * idx))
    scoreA = np.load(root + "scores/cloud_bin_" + '{:03.0f}'.format(2 * idx) + ".npy")
    scoreB = np.load(root + "scores/cloud_bin_" + '{:03.0f}'.format(2 * idx + 1) + ".npy")
    kptsA = np.load(root + "keypoints/cloud_bin_" + '{:03.0f}'.format(2 * idx) + ".npy")
    descA = np.load(root + "descriptors/cloud_bin_" + '{:03.0f}'.format(2 * idx) + ".D3Feat.npy")
    kptsB = np.load(root + "keypoints/cloud_bin_" + '{:03.0f}'.format(2 * idx + 1) + ".npy")
    descB = np.load(root + "descriptors/cloud_bin_" + '{:03.0f}'.format(2 * idx + 1) + ".D3Feat.npy")

    sortA = np.argsort(scoreA.squeeze())[-500:-1]
    sortB = np.argsort(scoreB.squeeze())[-500:-1]

    return kptsA[sortA, :], descA[sortA, :], kptsB[sortB, :], descB[sortB, :]


if __name__ == '__main__':
    # initialize arguments
    net_descs = True
    show = False
    icp = False
    if net_descs:
        args = matching.Args()  # feature extraction args
        args.model = "models/true_trained.pt"
        net = extract.load_network(args.model)
        net = net.cuda()

    beta = 0.3
    icp_dist = 0.15

    # initialize lidar db
    root = '/media/dominic/Extreme SSD/datasets/asl_koze/data_both_side/'

    db = LidarPairDataset(root, reproject=False)
    db.pair_dict = pd.read_csv(os.path.join(root, 'tf_bm.csv'), header=0)[::10].reset_index(drop=True)
    print(len(db))
    score_matrix = np.zeros((len(db), 5))

    # loop through benchmarking db
    for idx in range(len(db)):
        # get image pair
        imgA, imgB, meta = db.get_pair(idx)
        maskA = np.load(os.path.join(db.root, 'data', format(db.pair_dict['ts1'][idx], '.0f'), 'valid_mask.npy'))
        xyzA = np.load(os.path.join(db.root, 'data', format(db.pair_dict['ts1'][idx], '.0f'), 'xyz.npy'))
        maskB = np.load(os.path.join(db.root, 'data', format(db.pair_dict['ts2'][idx], '.0f'), 'valid_mask.npy'))
        xyzB = np.load(os.path.join(db.root, 'data', format(db.pair_dict['ts2'][idx], '.0f'), 'xyz.npy'))
        tf_true = db.get_homog_matrix(db.pair_dict.loc[idx, ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].to_numpy())

        # get key points and descriptors
        if net_descs:
            kptsA, descA, xysA, kptsB, descB, xysB = extract_r2d2_features(imgA, imgB, xyzA, xyzB, args, net)
            xyzA = xyzA.reshape((-1, 3))[maskA.reshape(-1)]
            xyzB = xyzB.reshape((-1, 3))[maskB.reshape(-1)]
        else:
            xyzA = xyzA.reshape((-1, 3))[maskA.reshape(-1)]
            xyzB = xyzB.reshape((-1, 3))[maskB.reshape(-1)]
            kptsA, descA, kptsB, descB = get_d_three_features(idx)

        # get key point benchmark metrics
        RS, LE = interest_point_score(kptsA, kptsB, maskA, maskB, tf_true, beta)
        MS, TE, matches, corr_mask, tf = descriptor_score(pts1=kptsA, pts2=kptsB, desc1=descA, desc2=descB,
                                                          xyz1=xyzA, xyz2=xyzB, tf_true=tf_true, beta=beta,
                                                          icp=icp, icp_dist=icp_dist, show=show)
        print("Repeatability Score: ", RS)
        print("Localization Error: ", LE)
        print("Matching Score: ", MS)
        print("Transformation Scores:")
        print("    -Translation Error: ", TE[0], "m")
        print("    -Rotation Error: ", TE[1], "deg")
        score_matrix[idx, :] = [RS, LE, MS, TE[0], TE[1]]

        if show and net_descs:
            def blended(xys, img, matches):
                x = xys[matches, 0].astype(int)
                y = xys[matches, 1].astype(int)
                r, i, s = img.split()
                i = np.array(i)
                i = cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)
                # i = np.array(img)
                for k in range(x.shape[0]):
                    i = cv2.circle(i, (x[k], y[k]), 2, (255, 0, 0), 1)
                return i

            blendA = blended(xysA, imgA, matches[:, 0].astype(int))
            blend = blended(xysB, imgB, matches[:, 1].astype(int))
            stacked = np.vstack((blendA, blend))

            h = imgA.size[1]
            thickness = 1
            lineType = cv2.LINE_AA
            for j in range(matches.shape[0]):
                x1 = xysA[matches[:, 0].astype(int), 0][j]
                y1 = xysA[matches[:, 0].astype(int), 1][j]
                x2 = xysB[matches[:, 1].astype(int), 0][j]
                y2 = xysB[matches[:, 1].astype(int), 1][j] + h
                if corr_mask[j]:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                cv2.line(stacked, (x1, y1), (x2, y2), color, thickness, lineType)

            win_inp = 'Keypoints'
            cv2.namedWindow(win_inp)
            cv2.imshow(win_inp, stacked)
            plt.show()
            cv2.waitKey(0) & 0xFF

    print(score_matrix)
    print(score_matrix.mean(axis=0))
    # save benchmark results
    np.savetxt(root + 'true_pairs_32_bm.csv', score_matrix, delimiter=',')
