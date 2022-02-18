
#from pycocotools.cocoeval import COCOeval

import os
import numpy as np
import transforms3d as tf3d
import copy
import cv2
import open3d
from ..utils import ply_loader
from ..utils.anchors import locations_for_shape
from .pose_error import reproj, add, adi, re, te, vsd
import json
import sys
import matplotlib.pyplot as plt
import time
import csv

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


# LineMOD
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899


def toPix_array(translation, fx, fy, cx, cy):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


def load_pcd(data_path, cat):
    # load meshes
    ply_path = os.path.join(data_path, 'meshes', 'obj_' + cat + '.ply')
    pcd_model = open3d.io.read_point_cloud(ply_path)
    model_vsd = {}
    model_vsd['pts'] = np.asarray(pcd_model.points)
    #open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
    #    radius=0.1, max_nn=30))
    # open3d.draw_geometries([pcd_model])
    model_vsd['pts'] = model_vsd['pts'] * 0.001

    return pcd_model, model_vsd
'''

def load_pcd(data_path, cat):
    # load meshes
    ply_path = os.path.join(data_path, 'meshes', 'obj_' + cat + '.ply')
    model_vsd = ply_loader.load_ply(ply_path)
    pcd_model = open3d.geometry.PointCloud()
    pcd_model.points = open3d.utility.Vector3dVector(model_vsd['pts'])
    pcd_model.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    # open3d.draw_geometries([pcd_model])
    model_vsd_mm = copy.deepcopy(model_vsd)
    model_vsd_mm['pts'] = model_vsd_mm['pts'] * 1000.0
    #pcd_model = open3d.read_point_cloud(ply_path)
    #pcd_model = None

    return pcd_model, model_vsd, model_vsd_mm
'''


def create_point_cloud(depth, fx, fy, cx, cy, ds):

    rows, cols = depth.shape

    depRe = depth.reshape(rows * cols)
    zP = np.multiply(depRe, ds)

    x, y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1), indexing='xy')
    yP = y.reshape(rows * cols) - cy
    xP = x.reshape(rows * cols) - cx
    yP = np.multiply(yP, zP)
    xP = np.multiply(xP, zP)
    yP = np.divide(yP, fy)
    xP = np.divide(xP, fx)

    cloud_final = np.transpose(np.array((xP, yP, zP)))
    #cloud_final[cloud_final[:,2]==0] = np.NaN

    return cloud_final


def boxoverlap(a, b):
    a = np.array([a[0], a[1], a[0] + a[2], a[1] + a[3]])
    b = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])

    x1 = np.amax(np.array([a[0], b[0]]))
    y1 = np.amax(np.array([a[1], b[1]]))
    x2 = np.amin(np.array([a[2], b[2]]))
    y2 = np.amin(np.array([a[3], b[3]]))

    wid = x2-x1+1
    hei = y2-y1+1
    inter = wid * hei
    aarea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    # intersection over union overlap
    ovlap = inter / (aarea + barea - inter)
    # set invalid entries to 0 overlap
    maskwid = wid <= 0
    maskhei = hei <= 0
    np.where(ovlap, maskwid, 0)
    np.where(ovlap, maskhei, 0)

    return ovlap


def evaluate_occlusion(generator, model, data_path, threshold=0.5):

    mesh_info = os.path.join(data_path, "meshes/models_info.json")
    threeD_boxes = np.ndarray((31, 8, 3), dtype=np.float32)
    model_dia = np.zeros((31), dtype=np.float32)

    for key, value in json.load(open(mesh_info)).items():
        fac = 0.001
        x_minus = value['min_x'] * fac
        y_minus = value['min_y'] * fac
        z_minus = value['min_z'] * fac
        x_plus = value['size_x'] * fac + x_minus
        y_plus = value['size_y'] * fac + y_minus
        z_plus = value['size_z'] * fac + z_minus
        three_box_solo = np.array([
                                    #[0.0, 0.0, 0.0],
                                    [x_plus, y_plus, z_plus],
                                  [x_plus, y_plus, z_minus],
                                  [x_plus, y_minus, z_minus],
                                  [x_plus, y_minus, z_plus],
                                  [x_minus, y_plus, z_plus],
                                  [x_minus, y_plus, z_minus],
                                  [x_minus, y_minus, z_minus],
                                  [x_minus, y_minus, z_plus]])
        threeD_boxes[int(key), :, :] = three_box_solo
        model_dia[int(key)] = value['diameter'] * fac

    pc1, mv1 = load_pcd(data_path, '000001')
    pc5, mv5 = load_pcd(data_path, '000005')
    pc6, mv6 = load_pcd(data_path, '000006')
    pc8, mv8 = load_pcd(data_path, '000008')
    pc9, mv9 = load_pcd(data_path, '000009')
    pc10, mv10 = load_pcd(data_path, '000010')
    pc11, mv11 = load_pcd(data_path, '000011')
    pc12, mv12 = load_pcd(data_path, '000012')

    allPoses = np.zeros((16), dtype=np.uint32)
    truePoses = np.zeros((16), dtype=np.uint32)
    falsePoses = np.zeros((16), dtype=np.uint32)
    trueDets = np.zeros((16), dtype=np.uint32)
    falseDets = np.zeros((16), dtype=np.uint32)

    eval_img = []
    for index, sample in enumerate(generator):

        print('Proecessing sample ', index)

        image_id = sample[0]
        image = sample[1]
        gt_labels = sample[2].numpy()
        gt_boxes = sample[3].numpy()
        gt_poses = sample[4].numpy()
        gt_calib = sample[5].numpy()

        if gt_labels.size == 0:
            continue

        gt_label_list = []
        gt_poses_list = []
        for obj in range(gt_labels.shape[0]):
            allPoses[int(gt_labels[obj]) + 1] += 1
            gt_label_list.append(int(gt_labels[obj]) + 1)
            gt_poses_list.append(int(gt_labels[obj]) + 1)

        fxkin = gt_calib[0, 0]
        fykin = gt_calib[0, 1]
        cxkin = gt_calib[0, 2]
        cykin = gt_calib[0, 3]

        image_raw = image.numpy()
        image_raw[..., 0] += 103.939
        image_raw[..., 1] += 116.779
        image_raw[..., 2] += 123.68
        image_raw = image_raw.astype(np.uint8)

        image_mask = copy.deepcopy(image)
        image_box = copy.deepcopy(image)
        image_poses = copy.deepcopy(image)

        # run network
        boxes3D, scores, labels, poses, consistency, mask = model.predict_on_batch(np.expand_dims(image, axis=0))

        boxes3D = boxes3D[labels != -1, :]
        scores = scores[labels != -1]
        confs = consistency[labels != -1]
        poses = poses[labels != -1]
        masks = mask[mask != -1]
        labels = labels[labels != -1]

        image_mask = copy.deepcopy(image_raw)
        image_box = copy.deepcopy(image_raw)
        image_poses = copy.deepcopy(image_raw)

        for inv_cls in np.unique(labels):

            true_cls = inv_cls + 1
            cls = true_cls

            pose_votes = boxes3D[labels == inv_cls]
            scores_votes = scores[labels == inv_cls]
            poses_votes = poses[labels == inv_cls, inv_cls]
            confs_votes = confs[labels == inv_cls]
            labels_votes = labels[labels == inv_cls]
            mask_votes = masks[labels == inv_cls]

            col_box = (int(np.random.uniform()*255.0), int(np.random.uniform()*255.0), int(np.random.uniform()*255.0))
            pyramids = np.zeros((6300, 3))
            pyramids[mask_votes, :] = col_box
            P3_mask = np.reshape(pyramids[:4800, :], (60, 80, 3))
            P4_mask = np.reshape(pyramids[4800:6000, :], (30, 40, 3))
            P5_mask = np.reshape(pyramids[6000:, :], (15, 20, 3))
            P3_mask = cv2.resize(P3_mask, (640, 480), interpolation = cv2.INTER_NEAREST)
            P4_mask = cv2.resize(P4_mask, (640, 480), interpolation = cv2.INTER_NEAREST)
            P5_mask = cv2.resize(P5_mask, (640, 480), interpolation = cv2.INTER_NEAREST)
            image_mask = np.where(P3_mask > 0, P3_mask, image_mask)
            image_mask = np.where(P4_mask > 0, P4_mask, image_mask)
            image_mask = np.where(P5_mask > 0, P5_mask, image_mask)

            min_box_x = np.nanmin(pose_votes[:, ::2], axis=1)
            min_box_y = np.nanmin(pose_votes[:, 1::2], axis=1)
            max_box_x = np.nanmax(pose_votes[:, ::2], axis=1)
            max_box_y = np.nanmax(pose_votes[:, 1::2], axis=1)

            pos_anchors = np.stack([min_box_x, min_box_y, max_box_x, max_box_y], axis=1)

            ind_anchors = np.where(labels_votes == inv_cls)[0]

            per_obj_hyps = []
            per_obj_cls = []
            per_obj_poses = []
            per_obj_hyps = []

            while pos_anchors.shape[0] > 0:
                # make sure to separate objects
                start_i = np.random.randint(pos_anchors.shape[0])
                obj_ancs = [pos_anchors[start_i]]
                obj_inds = [ind_anchors[start_i]]
                pos_anchors = np.delete(pos_anchors, start_i, axis=0)
                ind_anchors = np.delete(ind_anchors, start_i, axis=0)
                # print('ind_anchors: ', ind_anchors)
                same_obj = True
                while same_obj == True:
                    # update matrices based on iou
                    same_obj = False
                    indcs2rm = []
                    for adx in range(pos_anchors.shape[0]):
                        # loop through anchors
                        box_b = pos_anchors[adx, :]
                        if not np.all((box_b > 0)):  # need x_max or y_max here? maybe irrelevant due to positivity
                            indcs2rm.append(adx)
                            continue
                        for qdx in range(len(obj_ancs)):
                            # loop through anchors belonging to instance
                            iou = boxoverlap(obj_ancs[qdx], box_b)
                            if iou > 0.4:
                                # print('anc_anchors: ', pos_anchors)
                                # print('ind_anchors: ', ind_anchors)
                                # print('adx: ', adx)
                                obj_ancs.append(box_b)
                                obj_inds.append(ind_anchors[adx])
                                indcs2rm.append(adx)
                                same_obj = True
                                break
                        if same_obj == True:
                            break

                    # print('pos_anchors: ', pos_anchors.shape)
                    # print('ind_anchors: ', len(ind_anchors))
                    # print('indcs2rm: ', indcs2rm)
                    pos_anchors = np.delete(pos_anchors, indcs2rm, axis=0)
                    ind_anchors = np.delete(ind_anchors, indcs2rm, axis=0)

                per_obj_cls.append(cls)
                per_obj_hyps.append(obj_inds)

                for inst, hyps in enumerate(per_obj_hyps):

                    inv_cls = per_obj_cls[inst] - 1
                    true_cls = inv_cls + 1
                    gt_idx = np.argwhere(gt_labels == inv_cls)
                    gt_pose = gt_poses[gt_idx, :]
                    gt_box = gt_boxes[gt_idx, :]

                    if true_cls not in [1, 5, 6, 8, 9, 10, 11, 12] or gt_pose.size == 0: #filter for benchvise, bowl and mug
                        continue

                    print(true_cls)
                    print(gt_pose)
                    gt_pose = gt_pose[0][0]
                    gt_box = gt_box[0][0]

                    box_votes = pose_votes[hyps, :]
                    k_hyp = box_votes.shape[0]

                    # detection
                    min_x = int(np.mean(np.nanmin(box_votes[:, ::2], axis=1), axis=0))
                    min_y = int(np.mean(np.nanmin(box_votes[:, 1::2], axis=1), axis=0))
                    max_x = int(np.mean(np.nanmax(box_votes[:, ::2], axis=1), axis=0))
                    max_y = int(np.mean(np.nanmax(box_votes[:, 1::2], axis=1), axis=0))
                    est_box = np.array([float(min_x), float(min_y), float(max_x), float(max_y)])

                    iou = boxoverlap(est_box, gt_box)

                    if iou > 0.5 and true_cls in gt_label_list:
                        trueDets[true_cls] += 1
                        gt_label_list.remove(true_cls)
                    else:
                        falseDets[true_cls] += 1

                    image_raw = cv2.rectangle(image_raw, (min_x, min_y), (max_x, max_y), (255, 255, 255), 3)

                    ori_points = np.ascontiguousarray(threeD_boxes[true_cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
                    K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

                    est_points = np.ascontiguousarray(box_votes, dtype=np.float32).reshape((int(k_hyp * 8), 1, 2))
                    obj_points = np.repeat(ori_points[np.newaxis, :, :], k_hyp, axis=0)
                    obj_points = obj_points.reshape((int(k_hyp * 8), 1, 3))
                    retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                                       imagePoints=est_points, cameraMatrix=K,
                                                                       distCoeffs=None, rvec=None, tvec=None,
                                                                       useExtrinsicGuess=False, iterationsCount=300,
                                                                       reprojectionError=5.0, confidence=0.99,
                                                                       flags=cv2.SOLVEPNP_EPNP)
                    R_est, _ = cv2.Rodrigues(orvec)
                    t_est = otvec.T
                    t_est = t_est[0, :]
                    t_bop = t_est * 1000.0

                    t_rot = tf3d.quaternions.quat2mat(gt_pose[3:])
                    R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                    t_gt = np.array(gt_pose[:3], dtype=np.float32)
                    t_gt = t_gt * 0.001

                    if true_cls == 1:
                        model_vsd = mv1
                    elif true_cls == 5:
                        model_vsd = mv5
                    elif true_cls == 6:
                        model_vsd = mv6
                    elif true_cls == 8:
                        model_vsd = mv8
                    elif true_cls == 9:
                        model_vsd = mv9
                    elif true_cls == 10:
                        model_vsd = mv10
                    elif true_cls == 11:
                        model_vsd = mv11
                    elif true_cls == 12:
                        model_vsd = mv12

                    if cls == 10 or cls == 11:
                        err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                    else:
                        err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                    if err_add < model_dia[true_cls] * 0.1:
                        if true_cls in gt_poses_list:
                            truePoses[true_cls] += 1
                            gt_poses_list.remove(true_cls)
                    else:
                        falsePoses[true_cls] += 1

                    print(' ')
                    print('error: ', err_add, 'threshold', model_dia[true_cls] * 0.1)

                    tDbox = R_gt.dot(ori_points.T).T
                    tDbox = tDbox + np.repeat(t_gt[:, np.newaxis], 8, axis=1).T
                    box3D = toPix_array(tDbox, fxkin, fykin, cxkin, cykin)
                    tDbox = np.reshape(box3D, (16))
                    tDbox = tDbox.astype(np.uint16)

                    eDbox = R_est.dot(ori_points.T).T
                    eDbox = eDbox + np.repeat(t_est[np.newaxis, :], 8, axis=0)
                    est3D = toPix_array(eDbox, fxkin, fykin, cxkin, cykin)
                    eDbox = np.reshape(est3D, (16))
                    pose = eDbox.astype(np.uint16)
                    colGT = (255, 0, 0)
                    colEst = (0, 204, 0)
                    if err_add > model_dia[true_cls] * 0.1:
                        colEst = (0, 0, 255)

                    image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()), colGT, 2)
                    image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()), colGT, 2)
                    image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()), colGT,
                                         2)
                    image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()), colGT,
                                         2)
                    image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()), colGT,
                                         2)
                    image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()), colGT,
                                         2)
                    image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()), colGT,
                                         2)
                    image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()), colGT,
                                         2)
                    image_raw = cv2.line(image_raw, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                                         colGT,
                                         2)
                    image_raw = cv2.line(image_raw, tuple(tDbox[10:12].ravel()), tuple(tDbox[12:14].ravel()),
                                         colGT,
                                         2)
                    image_raw = cv2.line(image_raw, tuple(tDbox[12:14].ravel()), tuple(tDbox[14:16].ravel()),
                                         colGT,
                                         2)
                    image_raw = cv2.line(image_raw, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                                         colGT,
                                         2)

                    image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 2)
                    image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 2)
                    image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 2)
                    image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 2)
                    image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 2)
                    image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
                    image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
                    image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
                    image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                                         2)
                    image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                                         2)
                    image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                                         2)
                    image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                                         2)


                    direct_votes = poses_votes[hyps, :]

                    '''
                    for pdx in range(direct_votes.shape[0]):
                        # direct pose
                        pose_hyp = direct_votes[pdx, :]
                        R_est = np.eye(3)
                        R_est[:3, 0] = pose_hyp[3:6] / np.linalg.norm(pose_hyp[3:6])
                        R_est[:3, 1] = pose_hyp[6:] / np.linalg.norm(pose_hyp[6:])
                        R3 = np.cross(R_est[:3, 0], pose_hyp[6:])
                        R_est[:3, 2] = R3 / np.linalg.norm(R3)
                        t_est = pose_hyp[:3].T * 0.001
                        t_bop = t_est * 1000.0
                        
                        eDbox = R_est.dot(ori_points.T).T
                        eDbox = eDbox + np.repeat(t_est[np.newaxis, :], 8, axis=0)
                        est3D = toPix_array(eDbox, fxkin, fykin, cxkin, cykin)

                        # bounding box estimation
                        # est3D = pose_votes[pdx, :]

                        # used for viz of both
                        eDbox = np.reshape(est3D, (16))
                        pose = eDbox.astype(np.uint16)
                        colGT = (255, 0, 0)
                        colEst = (0, 204, 0)

                        image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 2)
                        image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 2)
                        image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 2)
                        image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 2)
                        image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 2)
                        image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
                        image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
                        image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
                        image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                                             2)
                        image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                                             2)
                        image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                                             2)
                        image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                                             2)
                    '''

                    eval_line = []
                    R_bop = [str(i) for i in R_est.flatten().tolist()]
                    R_bop = ' '.join(R_bop)
                    eval_line.append(R_bop)
                    t_bop = [str(i) for i in t_bop.flatten().tolist()]
                    t_bop = ' '.join(t_bop)
                    eval_line.append(t_bop)
                    eval_img.append(eval_line)


        name = '/home/stefan/PyraPose_viz/' + 'sample_' + str(index) + '.png'
        # image_row1 = np.concatenate([image, image_mask], axis=0)
        #cv2.imwrite(name, image_raw)

        '''

            anno_ind = np.argwhere(anno['labels'] == cls)
            print(anno_ind)
            t_tra = anno['poses'][anno_ind[0][0]][:3]
            t_rot = anno['poses'][anno_ind[0][0]][3:]

            if true_cls == 1:
                model_vsd = mv1
            elif true_cls == 5:
                model_vsd = mv5
            elif true_cls == 6:
                model_vsd = mv6
            elif true_cls == 8:
                model_vsd = mv8
            elif true_cls == 9:
                model_vsd = mv9
            elif true_cls == 10:
                model_vsd = mv10
            elif true_cls == 11:
                model_vsd = mv11
            elif true_cls == 12:
                model_vsd = mv12

            ori_points = np.ascontiguousarray(threeD_boxes[true_cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
            K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)
            t_rot = tf3d.quaternions.quat2mat(t_rot)
            R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
            t_gt = np.array(t_tra, dtype=np.float32)
            t_gt = t_gt * 0.001

            k_hyp = pose_votes.shape[0]
            # min residual
            # res_idx = np.argmin(res_sum)
            # k_hyp = 1
            # pose_votes = pose_votes[:, res_idx, :]
            # max center
            # centerns = centers[0, cls_indices, 0]
            # centerns = np.squeeze(centerns)
            # max_center = np.argmax(centerns)
            # pose_votes = pose_votes[:, max_center, :]

            #print(pose_votes[:, :, cls, :].shape)
            est_points = np.ascontiguousarray(pose_votes, dtype=np.float32).reshape((int(k_hyp * 8), 1, 2))
            obj_points = np.repeat(ori_points[np.newaxis, :, :], k_hyp, axis=0)
            obj_points = obj_points.reshape((int(k_hyp * 8), 1, 3))
            retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                               imagePoints=est_points, cameraMatrix=K,
                                                               distCoeffs=None, rvec=None, tvec=None,
                                                               useExtrinsicGuess=False, iterationsCount=300,
                                                               reprojectionError=5.0, confidence=0.99,
                                                               flags=cv2.SOLVEPNP_EPNP)
            R_est, _ = cv2.Rodrigues(orvec)
            t_est = otvec.T

            #R_est = tf3d.quaternions.quat2mat(poses_cls[3:])
            #t_est = poses_cls[:3] * 0.001

            #R_est = np.eye(3)
            #R_est[:3, 0] = poses_cls[3:6] / np.linalg.norm(poses_cls[3:6])
            #R_est[:3, 1] = poses_cls[6:] / np.linalg.norm(poses_cls[6:])
            #R3 = np.cross(R_est[:3, 0], poses_cls[6:])
            #R_est[:3, 2] = R3 / np.linalg.norm(R3)
            ##R_est[:3, 1] = np.cross(R_est[:3, 2], R_est[:3, 0])
            #t_est = poses_cls[:3] * 0.001

            t_bop = t_est * 1000.0

            if cls == 10 or cls == 11:
                err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
            else:
                err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
            if err_add < model_dia[true_cls] * 0.1:
                truePoses[true_cls] += 1
            print(' ')
            print('error: ', err_add, 'threshold', model_dia[true_cls] * 0.1)

            
            R_bop = [str(i) for i in R_est.flatten().tolist()]
            R_bop = ' '.join(R_bop)
            eval_line.append(R_bop)
            t_bop = [str(i) for i in t_bop.flatten().tolist()]
            t_bop = ' '.join(t_bop)
            eval_line.append(t_bop)
            eval_img.append(eval_line)

            t_est = t_est.T  # * 0.001
            # print('pose: ', pose)
            # print(t_gt)
            # print(t_est)
            tDbox = R_gt.dot(ori_points.T).T
            tDbox = tDbox + np.repeat(t_gt[:, np.newaxis], 8, axis=1).T
            box3D = toPix_array(tDbox)
            tDbox = np.reshape(box3D, (16))
            tDbox = tDbox.astype(np.uint16)
            eDbox = R_est.dot(ori_points.T).T
            # print(eDbox.shape, np.repeat(t_est, 8, axis=1).T.shape)
            eDbox = eDbox + np.repeat(t_est, 8, axis=1).T
            # eDbox = eDbox + np.repeat(t_est, 8, axis=0)
            # print(eDbox.shape)
            est3D = toPix_array(eDbox)
            # print(est3D)
            eDbox = np.reshape(est3D, (16))
            pose = eDbox.astype(np.uint16)
            colGT = (255, 0, 0)
            colEst = (0, 204, 0)
            image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()), colGT, 2)
            image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()), colGT, 2)
            image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()), colGT,
                                 2)
            image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()), colGT,
                                 2)
            image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()), colGT,
                                 2)
            image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()), colGT,
                                 2)
            image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()), colGT,
                                 2)
            image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()), colGT,
                                 2)
            image_raw = cv2.line(image_raw, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                                 colGT,
                                 2)
            image_raw = cv2.line(image_raw, tuple(tDbox[10:12].ravel()), tuple(tDbox[12:14].ravel()),
                                 colGT,
                                 2)
            image_raw = cv2.line(image_raw, tuple(tDbox[12:14].ravel()), tuple(tDbox[14:16].ravel()),
                                 colGT,
                                 2)
            image_raw = cv2.line(image_raw, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                                 colGT,
                                 2)
            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                                 2)
            image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                                 2)
            image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                                 2)
            image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                                 2)

            image = image_box
            for idx in range(pose_votes.shape[0]):
                image = cv2.circle(image, (pose_votes[idx, 0], pose_votes[idx, 1]), 5, col_box)
                image = cv2.circle(image, (pose_votes[idx, 2], pose_votes[idx, 3]), 5, col_box)
                image = cv2.circle(image, (pose_votes[idx, 4], pose_votes[idx, 5]), 5, col_box)
                image = cv2.circle(image, (pose_votes[idx, 6], pose_votes[idx, 7]), 5, col_box)
                image = cv2.circle(image, (pose_votes[idx, 8], pose_votes[idx, 9]), 5, col_box)
                image = cv2.circle(image, (pose_votes[idx, 10], pose_votes[idx, 11]), 5, col_box)
                image = cv2.circle(image, (pose_votes[idx, 12], pose_votes[idx, 13]), 5, col_box)
                image = cv2.circle(image, (pose_votes[idx, 14], pose_votes[idx, 15]), 5, col_box)

                eDbox = R_est.dot(ori_points.T).T
                # print(eDbox.shape, np.repeat(t_est, 8, axis=1).T.shape)
                eDbox = eDbox + np.repeat(t_est[:, np.newaxis], 8, axis=1).T
                #eDbox = eDbox + np.repeat(t_est, 8, axis=0)
                # print(eDbox.shape)
                est3D = toPix_array(eDbox)
                # print(est3D)
                eDbox = np.reshape(est3D, (16))
                pose = eDbox.astype(np.uint16)
                image_poses = cv2.line(image_poses, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), col_box, 2)
                image_poses = cv2.line(image_poses, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), col_box, 2)
                image_poses = cv2.line(image_poses, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), col_box, 2)
                image_poses = cv2.line(image_poses, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), col_box, 2)
                image_poses = cv2.line(image_poses, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), col_box, 2)
                image_poses = cv2.line(image_poses, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), col_box, 2)
                image_poses = cv2.line(image_poses, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), col_box, 2)
                image_poses = cv2.line(image_poses, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), col_box, 2)
                image_poses = cv2.line(image_poses, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), col_box,
                                     2)
                image_poses = cv2.line(image_poses, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), col_box,
                                     2)
                image_poses = cv2.line(image_poses, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), col_box,
                                     2)
                image_poses = cv2.line(image_poses, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), col_box,
                                     2)

        name_raw = '/home/PyraPose_viz/raw_' + str(index) + '.jpg'
        name_mask = '/home/PyraPose_viz/mask_' + str(index) + '.jpg'
        name_box = '/home/PyraPose_viz/box_' + str(index) + '.jpg'
        name_pose = '/home/PyraPose_viz/poses_' + str(index) + '.jpg'
        cv2.imwrite(name_raw, image_raw)
        cv2.imwrite(name_mask, image_mask)
        cv2.imwrite(name_box, image_box)
        cv2.imwrite(name_pose, image_poses)
        #cv2.imwrite('/home/stefan/occ_viz/pred_mask_' + str(index) + '_.jpg', image_mask)
        #print('break')
        t_eval = time.time()-t_start
        wd_path = os.getcwd()
        #csv_target = os.path.join(wd_path, 'results_occlusion.csv')
        csv_target = os.path.join(wd_path, 'sthalham-pp_lmo-test.csv')

        #line_head = ['scene_id','im_id','obj_id','score','R','t','time']
        #with open(csv_target, 'a') as outfile:
        #    myWriter = csv.writer(outfile, delimiter=',')  # Write out the Headers for the CSV file
        #    myWriter.writerow(line_head)

        #for line_indexed in eval_img:
        #    line_indexed.append(str(t_eval))
        #    with open(csv_target, 'a') as outfile:
        #        myWriter = csv.writer(outfile, delimiter=',')  # Write out the Headers for the CSV file
        #        myWriter.writerow(line_indexed)
        '''

    recall = np.zeros((16), dtype=np.float32)
    precision = np.zeros((16), dtype=np.float32)
    detections = np.zeros((16), dtype=np.float32)
    det_precision = np.zeros((16), dtype=np.float32)
    for i in range(1, (allPoses.shape[0])):
        recall[i] = truePoses[i] / allPoses[i]
        precision[i] = truePoses[i] / (truePoses[i] + falsePoses[i])
        detections[i] = trueDets[i] / allPoses[i]
        det_precision[i] = trueDets[i] / (trueDets[i] + falseDets[i])

        if np.isnan(recall[i]):
            recall[i] = 0.0
        if np.isnan(precision[i]):
            precision[i] = 0.0
        if np.isnan(detections[i]):
            detections[i] = 0.0
        if np.isnan(det_precision[i]):
            det_precision[i] = 0.0

        print('-------------------------------------')
        print('CLS: ', i)
        print('detection recall: ', detections[i])
        print('detection precision: ', detections[i])
        print('poses recall: ', recall[i])
        print('poses precision: ', precision[i])
        print('-------------------------------------')

    filter_indices = [1, 5, 6, 8, 9, 10, 11, 12]
    recall_all = np.sum(np.take(recall, filter_indices, axis=0)) / 8.0
    precision_all = np.sum(np.take(precision, filter_indices, axis=0)) / 8.0
    detections_all = np.sum(np.take(detections, filter_indices, axis=0)) / 8.0
    det_precision_all = np.sum(np.take(det_precision, filter_indices, axis=0)) / 8.0
    print('ALL: ')
    print('mean detection recall: ', detections_all)
    print('mean detection precision: ', det_precision_all)
    print('mean pose recall: ', recall_all)
    print('mean pose precision: ', precision_all)
