import extract
from datasets.lidar_dataset import LidarSynthetic
import numpy as np
import cv2
import tools.point_cloud_utils as pcu
import matplotlib.pyplot as plt
import open3d as o3d
import os

# define arguments needed to extract feautures from images
class Args:
    def __init__(self):
        self.model = None
        self.scale_f = 2 ** 0.25
        self.min_size = 1024
        self.max_size = 2048
        self.min_scale = 1
        self.max_scale = 1

        self.reliability_thr = 0.7
        self.repeatability_thr = 0.7
        self.top_k = 500

        self.gpu = [0]

# estimate transformation based on key points and descriptors of two point clouds with RANSAC
def execute_global_registration(source_down, target_down, reference_desc, target_desc, distance_threshold):
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, reference_desc, target_desc,
        distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
         o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.7)],
        o3d.registration.RANSACConvergenceCriteria(400000, 600))
    return result

# remove invalid key points from extracted xys, scores and desc
def remove_invalid_keypoints(xys, scores, desc, mask):
    valid_mask = mask[xys[:, 1].astype(int), xys[:, 0].astype(int)]
    xys = xys[valid_mask]
    scores = scores[valid_mask]
    desc = desc[valid_mask]
    return xys, scores, desc


# draw two unaligne point clouds after aligning them with transformation
def draw_registration_result(source, target, transformation):
    import copy
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0.8, 0.4, 0])
    target_temp.paint_uniform_color([0, 0.6, 0.6])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == '__main__':
    # initialize arguments for feature extraction
    args = Args()
    args.model = "models/true_trained.pt"
    net = extract.load_network(args.model)
    net = net.cuda()
    show = True
    # initialize lidar db
    root = '/media/dominic/Extreme SSD/datasets/asl_koze/data_both_side/data'
    db = LidarSynthetic(root, skip=(0, -1, 1), crop=False)

    # select two scans from db
    idx1 = 200
    idx2 = idx1 + 30

    # load scan images
    imgA = db.get_image(idx1)
    imgB = db.get_image(idx2)

    # extract keypoints and descriptors A
    maskA = db.get_valid_range_mask(idx1)
    xysA, scoresA, descA = extract.extract_keypoints(imgA, args, net)
    xysA, scoresA, descA = remove_invalid_keypoints(xysA, scoresA, descA, maskA)
    xyzA = db.get_xyz(idx1)
    xyzA_sort = xyzA[xysA[:, 1].astype(int), xysA[:, 0].astype(int)]
    xyzA = xyzA.reshape((-1, 3))[maskA.reshape(-1)]

    # extract keypoints and descriptors B
    maskB = db.get_valid_range_mask(idx2)
    xysB, scoresB, descB = extract.extract_keypoints(imgB, args, net)
    xysB, scoresB, descB = remove_invalid_keypoints(xysB, scoresB, descB, maskB)
    xyzB = db.get_xyz(idx2)
    xyzB_sort = xyzB[xysB[:, 1].astype(int), xysB[:, 0].astype(int)]
    xyzB = xyzB.reshape((-1, 3))[maskB.reshape(-1)]

    # cast data to open3d format
    reference_pc = o3d.geometry.PointCloud()
    reference_pc.points = o3d.utility.Vector3dVector(xyzA)

    test_pc = o3d.geometry.PointCloud()
    test_pc.points = o3d.utility.Vector3dVector(xyzB)

    ref = o3d.registration.Feature()
    ref.data = descA.T

    test = o3d.registration.Feature()
    test.data = descB.T

    ref_key = o3d.geometry.PointCloud()

    ref_key.points = o3d.utility.Vector3dVector(xyzA_sort)
    test_key = o3d.geometry.PointCloud()
    test_key.points = o3d.utility.Vector3dVector(xyzB_sort)

    # get tansformation estimate and set of correct matches with RANSAC
    result_ransac = execute_global_registration(ref_key, test_key, ref, test, 0.75)
    tf = result_ransac.transformation
    matches = np.array(result_ransac.correspondence_set)

    # show results
    if show:
        # Plot point clouds after registration
        reference_pc.paint_uniform_color([0.8, 0.4, 0])
        test_pc.paint_uniform_color([0, 0.6, 0.6])
        match_lines = o3d.geometry.LineSet()
        match_lines.points = o3d.utility.Vector3dVector(np.concatenate((xyzA_sort, xyzB_sort), axis=0))
        matches[:, 1] = matches[:, 1] + xyzA_sort.shape[0] # select matches of second image
        match_lines.lines = o3d.utility.Vector2iVector(matches)
        match_lines.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([reference_pc, test_pc, match_lines])
        draw_registration_result(reference_pc, test_pc, tf)
        matches[:, 1] = matches[:, 1] - xyzA_sort.shape[0] # recorrect matches to standard format

        # draw matches on scan images
        def draw_circles(xys, img, matches):
            x = xys[matches, 0].astype(int)
            y = xys[matches, 1].astype(int)
            r, i, s = img.split()
            i = np.array(i)
            i = cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)
            # i = np.array(img)
            for k in range(x.shape[0]):
                # if not mask[k]: continue
                i = cv2.circle(i, (x[k], y[k]), 2, (0, 0, 255), 1)
            return i
        # draw key point circles
        int_w_circ_A = draw_circles(xysA, imgA, matches[:, 0].astype(int))
        int_w_circ_B = draw_circles(xysB, imgB, matches[:, 1].astype(int))
        stacked = np.vstack((int_w_circ_A, int_w_circ_B))

        # draw match lines
        thickness = 1
        lineType = cv2.LINE_AA
        h = imgA.size[1]
        for j in range(matches.shape[0]):
            x1 = xysA[matches[:, 0].astype(int), 0][j]
            y1 = xysA[matches[:, 0].astype(int), 1][j]
            x2 = xysB[matches[:, 1].astype(int), 0][j]
            y2 = xysB[matches[:, 1].astype(int), 1][j] + h
            color = (0, 255, 0)
            cv2.line(stacked, (x1, y1), (x2, y2), color, thickness, lineType)

        win_inp = 'Keypoints'
        cv2.namedWindow(win_inp)
        cv2.imshow(win_inp, stacked)
        plt.show()
        cv2.waitKey(0)
