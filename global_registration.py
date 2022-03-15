import open3d as o3d
import time
import numpy as np
from sklearn.cluster import KMeans
import hdbscan
from sklearn.neighbors import BallTree
from scipy.spatial.transform import Rotation as R
import pandas as pd

import extract
from datasets.lidar_dataset import LidarSynthetic


# define arguments needed to extract feautures from images
class Args:
    def __init__(self, model=None):
        self.model = model
        self.top_k = 300
        self.scale_f = 2 ** 0.25
        self.min_size = 1024
        self.max_size = 2048
        self.min_scale = 1
        self.max_scale = 1 # only orignal resolution to extract from

        self.reliability_thr = 0.7
        self.repeatability_thr = 0.7

        self.gpu = [0]


# draw two point clouds transfromed into the same frame with transformation
def draw_registration_result(source, target, transformation):
    import copy
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

# estimate transformation based on key points and descriptors of two point clouds with RANSAC
def execute_global_registration(source_down, target_down, reference_desc, target_desc, distance_threshold):
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, reference_desc, target_desc,
        distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
         o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.7)],
        o3d.registration.RANSACConvergenceCriteria(800000, 1000))
    return result

# transformation estimate refined with ICP
def pairwise_registration(pc_source, pc_target, desc_source, desc_target, kpts_source, kpts_target, max_correspondence_distance):
    ref = o3d.registration.Feature()
    ref.data = desc_source.T
    test = o3d.registration.Feature()
    test.data = desc_target.T
    ref_key = o3d.geometry.PointCloud()
    ref_key.points = o3d.utility.Vector3dVector(kpts_source)
    test_key = o3d.geometry.PointCloud()
    test_key.points = o3d.utility.Vector3dVector(kpts_target)

    result_ransac = execute_global_registration(ref_key, test_key, ref, test, 0.5)
    tf = result_ransac.transformation
    fitness = result_ransac.fitness

    # run icp on point clouds to refine rigid transformation
    icp = o3d.registration.registration_icp(
        pc_source, pc_target, max_correspondence_distance, tf, o3d.registration.TransformationEstimationPointToPoint())
    transformation_icp = icp.transformation
    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        pc_source, pc_target, max_correspondence_distance, transformation_icp)
    return transformation_icp, information_icp, fitness


def global_registration(kpts_source, kpts_target, desc_source, desc_target, pc_source, pc_target, ransac_dist_threshold, icp_dist_threshold):
    result_ransac = execute_global_registration(kpts_source, kpts_target, desc_source, desc_target, ransac_dist_threshold)
    tf = result_ransac.transformation
    fitness = result_ransac.fitness

    reg_p2p = o3d.registration.registration_icp(
        pc_source, pc_target, icp_dist_threshold, tf,
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=100))
    tf = reg_p2p.transformation

    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        pc_source, pc_target, icp_dist_threshold, tf)
    return tf, information_icp, fitness


# remove invalid key points from extracted xys, scores and desc
def remove_invalid_keypoints(xys, scores, desc, mask):
    valid_mask = mask[xys[:, 1].astype(int), xys[:, 0].astype(int)]
    xys = xys[valid_mask]
    scores = scores[valid_mask]
    desc = desc[valid_mask]
    return xys, scores, desc

# cluster descriptor list in n_words clusters and calculate the normalized BoVW histogram for each scan
def build_bag_of_words_hist(desc_list, n_words=200):
    n_candidates = len(desc_list)
    desc = np.concatenate(desc_list, axis=0)
    kmeans = KMeans(n_clusters=n_words, max_iter=10000)
    labels = kmeans.fit_predict(desc)
    # build histogram
    hist = np.zeros((n_candidates, n_words), dtype=int)
    # print(hist.shape)
    candidate_it = 0
    for idx in range(n_candidates):
        n_landmarks = desc_list[idx].shape[0]
        candidate_labels = labels[candidate_it:candidate_it+n_landmarks]
        for candidate_label in candidate_labels:
            hist[idx, candidate_label] = hist[idx, candidate_label] + 1
        candidate_it = candidate_it + n_landmarks
    #
    hist = np.array(hist)
    hist = hist/np.linalg.norm(hist, axis=1, keepdims=True)
    return hist


# build search tree with all histograms and query a nearest neighbour search for every histogram(scan). 
# returns ids of nearest neighbours for every scan with min frame dist and max histogram distance
def get_bag_of_words_nearest_neighbours(hist, min_frame_dist, max_hist_dist):
    ball_tree = BallTree(hist)
    ids = ball_tree.query_radius(hist, max_hist_dist)
    n_candidates = len(ids)
    for i in range(n_candidates):
        dist_thresh_neighbour_mask = np.logical_or(i - min_frame_dist > ids[i], ids[i] > i + min_frame_dist)
        ids[i] = ids[i][dist_thresh_neighbour_mask]
    return ids

# run full pose graph estimation algorithm
def full_registration(db, net, net_args, args):
    # initialize
    icp_max_corr_dist = args["icp_max_corr_dist"]
    ransac_max_dist = args["ransac_max_dist"]
    min_frame_dist = args["min_frame_dist"]
    max_hist_dist = args["max_hist_dist"]
    n_words = args["n_words"]

    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
    n_pcds = len(db)

    pcd_list = []
    kpts_list = []
    desc_list = []

    # extract features from first image
    img = db.get_image(0)
    mask = db.get_valid_range_mask(0)
    xys, scores, desc = extract.extract_keypoints(img, net_args, net)
    xys, scores, desc = remove_invalid_keypoints(xys, scores, desc, mask)
    xyz = db.get_xyz(0)
    kpts = xyz[xys[:, 1].astype(int), xys[:, 0].astype(int)]
    desc_list.append(desc)
    kpts_list.append(kpts)

    # extract features for every scan image and add transfromation to pose graph and key points anddscriptors to list
    print("Calculating prior odometry with ICP and Collecting landmarks with r2d2 net")
    for source_id in range(n_pcds-1):
        progress = source_id / (n_pcds-1)
        print("\rProgress {:2.1%}".format(progress), end="")
        target_id = source_id + 1
        xyzA = db.get_xyz(source_id)
        maskA = db.get_valid_range_mask(source_id).reshape(-1)
        pc_source = o3d.geometry.PointCloud()
        pc_source.points = o3d.utility.Vector3dVector(xyzA.reshape(-1, 3)[maskA, :])
        xyzB = db.get_xyz(target_id)
        maskB = db.get_valid_range_mask(target_id).reshape(-1)
        pc_target = o3d.geometry.PointCloud()
        pc_target.points = o3d.utility.Vector3dVector(xyzB.reshape(-1, 3)[maskB, :])

        # collect pointclouds
        pcd_list.append(pc_source)

        # collect landmarks with r2d2 network
        img = db.get_image(target_id)
        mask = db.get_valid_range_mask(target_id)
        xys, scores, desc = extract.extract_keypoints(img, net_args, net)
        xys, scores, desc = remove_invalid_keypoints(xys, scores, desc, mask)
        kpts = xyzB[xys[:, 1].astype(int), xys[:, 0].astype(int)]
        desc_list.append(desc)
        kpts_list.append(kpts)

        # get odometry constraints by icp
        transformation_icp, information_icp, fitness = pairwise_registration(pc_source,
                                                                    pc_target,
                                                                    desc_list[source_id],
                                                                    desc_list[target_id],
                                                                    kpts_list[source_id],
                                                                    kpts_list[target_id],
                                                                    icp_max_corr_dist)
        # draw_registration_result(pc_source, pc_target, transformation_icp)
        odometry = np.dot(transformation_icp, odometry)
        pose_graph.nodes.append(o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
        pose_graph.edges.append(o3d.registration.PoseGraphEdge(source_id,
                                                               target_id,
                                                               transformation_icp,
                                                               information_icp,
                                                               confidence=fitness*0.1,
                                                               uncertain=True))
    pcd_list.append(pc_target)
    print("\rPorgress 100.0%")

    # get loop closure candidates based on BoVW hist neighbours
    print("Building BoVW")
    hist = build_bag_of_words_hist(desc_list, n_words=n_words)
    print("Calculating lc candidates from BoVW")
    nn_ids = get_bag_of_words_nearest_neighbours(hist, min_frame_dist=min_frame_dist, max_hist_dist=max_hist_dist)
    # calculate transformation between lc candidates
    print("Calculating transformation between lc candidates")
    for source_id in range(len(nn_ids)):
        progress = source_id / len(nn_ids)
        print("\rProgress {:2.1%}".format(progress), end="")
        if nn_ids[source_id].shape[0] == 0:
            continue
        desc_source = o3d.registration.Feature()
        desc_source.data = desc_list[source_id].T
        kpts_source = o3d.geometry.PointCloud()
        kpts_source.points = o3d.utility.Vector3dVector(kpts_list[source_id])
        pc_source = pcd_list[source_id]

        for target_id in nn_ids[source_id]:
            desc_target = o3d.registration.Feature()
            desc_target.data = desc_list[target_id].T
            kpts_target = o3d.geometry.PointCloud()
            kpts_target.points = o3d.utility.Vector3dVector(kpts_list[target_id])
            pc_target = pcd_list[target_id]

            tf, icp_info, fitness = global_registration(kpts_source=kpts_source,
                                                        kpts_target=kpts_target,
                                                        desc_source=desc_source,
                                                        desc_target=desc_target,
                                                        pc_source=pc_source,
                                                        pc_target=pc_target,
                                                        ransac_dist_threshold=ransac_max_dist,
                                                        icp_dist_threshold=icp_max_corr_dist)
            # add found loop closure to pose graph if stable enough
            if fitness < 0.4:
                continue
            pose_graph.edges.append(o3d.registration.PoseGraphEdge(source_id,
                                                                   target_id,
                                                                   tf,
                                                                   icp_info,
                                                                   confidence=fitness,
                                                                   uncertain=False))
    print("\rPorgress 100.0%")
    return pose_graph, pcd_list


# vizulaization helper: get lines which connect the poses (trajectory)
def get_line_set_from_poses(pose_graph, color):
    poses = pose_graph.nodes[0].pose[0:3, 3:]
    for i in range(1, len(pose_graph.nodes)):
        poses = np.concatenate((poses, pose_graph.nodes[i].pose[0:3, 3:]), axis=1)
    poses = poses.T
    lines = np.arange(len(pose_graph.nodes) - 1)[..., np.newaxis]
    lines = np.concatenate((lines, lines + 1), axis=1)
    lines.astype(int)
    colors = [color for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(poses)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

# vizulaization helper: get loop closure edges as lines
def get_line_set_from_loop_closures(pose_graph, color):
    poses = pose_graph.nodes[0].pose[0:3, 3:]
    for i in range(1, len(pose_graph.nodes)):
        poses = np.concatenate((poses, pose_graph.nodes[i].pose[0:3, 3:]), axis=1)
    poses = poses.T

    lc_edges = list()
    for i in range(len(pose_graph.nodes), len(pose_graph.edges)):
        edge = pose_graph.edges[i]
        lc_edges.append([edge.source_node_id, edge.target_node_id])

    colors = [color for i in range(len(lc_edges))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(poses)
    line_set.lines = o3d.utility.Vector2iVector(lc_edges)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

# vizulaization helper: draw point cloud alignment for every loop closure edge
def draw_loop_closure_matching(pose_graph, pcd_list):
    for i in range(len(pose_graph.nodes), len(pose_graph.edges)):
        edge = pose_graph.edges[i]
        pc_source = pcd_list[edge.source_node_id]
        pc_target = pcd_list[edge.target_node_id]
        tf = edge.transformation
        draw_registration_result(pc_source, pc_target, tf)

# write poses to csv (with timestamps)
def write_stamped_pose_graph_to_csv(pose_graph, timestamps, file_path):
    pose = pose_graph.nodes[0].pose
    poses = np.concatenate(([timestamps[0]], pose[0:3, 3:].squeeze(), R.from_matrix(pose[0:3, 0:3]).as_quat()))[np.newaxis, ...]
    for i in range(1, len(pose_graph.nodes)):
        pose = pose_graph.nodes[i].pose
        poses = np.concatenate((poses, np.concatenate(([timestamps[i]],
                                                      pose[0:3, 3:].squeeze(),
                                                      R.from_matrix(pose[0:3, 0:3]).as_quat()))[np.newaxis, ...]), axis=0)
    df = pd.DataFrame(poses, columns=['#ts', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    df.to_csv(file_path, index=False, sep=" ")


# initilaize arguments
args = dict()
args["icp_max_corr_dist"] = 0.25
args["ransac_max_dist"] = 0.3
args["min_frame_dist"] = 30
args["max_hist_dist"] = 0.85
args["n_words"] = 180
# args["root"] = '/media/dominic/Extreme SSD/datasets/asl_koze/data_both_side/'
args["root"] = '/media/dominic/Extreme SSD/datasets/lidarmace_data/ethz_outside/'
# args["root"] = '/media/dominic/Extreme SSD/datasets/new_college/long_experiment/'
# # asl_both_side
# args["db_start_idx"] = 32
# args["db_stop_idx"] = -1
# args["db_idx_skip"] = 5
# lee_outside
args["db_start_idx"] = 3
args["db_stop_idx"] = -516
args["db_idx_skip"] = 5
# newer_college
# args["db_start_idx"] = 100
# args["db_stop_idx"] = -1
# args["db_idx_skip"] = 20

args["type"] = "OS0"
args["nn_model_path"] = "models/true_32.pt"
args["csv_export_path"] = args["root"] + "stamped_traj_estimate.txt"
args["csv_export_path_lc"] = args["root"] + "stamped_traj_estimate_w-lc.txt"

db = LidarSynthetic(args["root"] + "/data/", crop=False, skip=(args["db_start_idx"], args["db_stop_idx"], args["db_idx_skip"]), type=args["type"])

# load the network...
net_args = Args(args["nn_model_path"])
net = extract.load_network(args["nn_model_path"])
net = net.cuda()

print("Full registration ...")
pose_graph, pcd_list = full_registration(db, net, net_args=net_args, args=args)

# save poses without loop closure
write_stamped_pose_graph_to_csv(pose_graph, db.img_folders, args["csv_export_path"])

line_set_1 = get_line_set_from_poses(pose_graph, [1, 0, 0])
lc_edges_1 = get_line_set_from_loop_closures(pose_graph, [0, 1, 0])
# draw_loop_closure_matching(pose_graph, pcd_list)

print("Optimizing PoseGraph ...")
option = o3d.registration.GlobalOptimizationOption(
    max_correspondence_distance=0.25,
    edge_prune_threshold=0.1,
    reference_node=0)

o3d.registration.global_optimization(
    pose_graph, o3d.registration.GlobalOptimizationLevenbergMarquardt(),
    o3d.registration.GlobalOptimizationConvergenceCriteria(), option)

# save poses optimized with loop closures
write_stamped_pose_graph_to_csv(pose_graph, db.img_folders, args["csv_export_path_lc"])
line_set_2 = get_line_set_from_poses(pose_graph, [0, 0, 1])

# draw results
o3d.visualization.draw_geometries([line_set_1, lc_edges_1, line_set_2])
o3d.visualization.draw_geometries([line_set_1, lc_edges_1])
o3d.visualization.draw_geometries([line_set_2])
