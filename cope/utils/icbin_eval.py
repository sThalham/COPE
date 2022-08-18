
import os
import tensorflow.keras as keras
import numpy as np
import json
import math
import transforms3d as tf3d
import copy
import cv2
import open3d
from ..utils import ply_loader
from .pose_error import reproj, add, adi, re, te, vsd
import yaml
import time
import csv


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
    model_vsd['pts'] = model_vsd['pts'] * 0.001

    pcd_down = pcd_model.voxel_down_sample(voxel_size=3)
    model_down = {}
    model_down['pts'] = np.asarray(pcd_down.points) * 0.001

    return pcd_model, model_vsd, model_down


def create_point_cloud(depth, ds):

    rows, cols = depth.shape

    depRe = depth.reshape(rows * cols)
    zP = np.multiply(depRe, ds)

    x, y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1), indexing='xy')
    yP = y.reshape(rows * cols) - cykin
    xP = x.reshape(rows * cols) - cxkin
    yP = np.multiply(yP, zP)
    xP = np.multiply(xP, zP)
    yP = np.divide(yP, fykin)
    xP = np.divide(xP, fxkin)

    cloud_final = np.transpose(np.array((xP, yP, zP)))

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


def evaluate_icbin(generator, model, data_path, threshold=0.5):

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
        norm_pts = np.linalg.norm(np.array([value['size_x'], value['size_y'], value['size_z']]))
        #x_plus = (value['size_x'] / norm_pts) * (value['diameter'] * 0.5)
        #y_plus = (value['size_y'] / norm_pts) * (value['diameter'] * 0.5)
        #z_plus = (value['size_z'] / norm_pts) * (value['diameter'] * 0.5)
        #x_minus = x_plus * -1.0
        #y_minus = y_plus * -1.0
        #z_minus = z_plus * -1.0
        three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                   [x_plus, y_plus, z_minus],
                                   [x_plus, y_minus, z_minus],
                                   [x_plus, y_minus, z_plus],
                                   [x_minus, y_plus, z_plus],
                                   [x_minus, y_plus, z_minus],
                                   [x_minus, y_minus, z_minus],
                                   [x_minus, y_minus, z_plus]])
        threeD_boxes[int(key), :, :] = three_box_solo
        model_dia[int(key)] = value['diameter'] * fac

    # target annotation
    pc1, mv1, md1 = load_pcd(data_path, '000001')
    pc2, mv2, md2 = load_pcd(data_path, '000002')

    allPoses = np.zeros((3), dtype=np.uint32)
    truePoses = np.zeros((3), dtype=np.uint32)
    falsePoses = np.zeros((3), dtype=np.uint32)
    trueDets = np.zeros((3), dtype=np.uint32)
    falseDets = np.zeros((3), dtype=np.uint32)
    times = np.zeros((300), dtype=np.float32)
    times_count = np.zeros((300), dtype=np.float32)

    colors_viz = np.random.randint(255, size=(2, 3))
    colors_viz = np.array([[205, 250, 255], [0, 215, 255]])
    max_gt = 0

    eval_img = []
    eval_det = []
    for index, sample in enumerate(generator):

        scene_id = sample[0].numpy()
        image_id = sample[1].numpy()
        image = sample[2]
        gt_labels = sample[3].numpy()
        gt_boxes = sample[4].numpy()
        gt_poses = sample[5].numpy()
        gt_calib = sample[6].numpy()

        if gt_labels.size == 0:
            continue

        fxkin = gt_calib[0, 0]
        fykin = gt_calib[0, 1]
        cxkin = gt_calib[0, 2]
        cykin = gt_calib[0, 3]

        image_raw = image.numpy()
        image_raw[..., 0] += 103.939
        image_raw[..., 1] += 116.779
        image_raw[..., 2] += 123.68
        image_raw = image_raw.astype(np.uint8)
        image_ori = image_raw.astype(np.uint8)

        image_mask = copy.deepcopy(image_raw)
        image_rep = copy.deepcopy(image_raw)
        image_box = copy.deepcopy(image_raw)
        image_pose = copy.deepcopy(image_raw)
        image_points = copy.deepcopy(image_raw)
        if gt_labels.shape[0] > max_gt:
            max_gt = gt_labels.shape[0]

        for obj in range(gt_labels.shape[0]):
            allPoses[int(gt_labels[obj]) + 1] += 1

            t_rot = tf3d.quaternions.quat2mat(gt_poses[obj, 3:])
            R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
            t_gt = np.array(gt_poses[obj, :3], dtype=np.float32)
            t_gt = t_gt * 0.001

            ori_points = np.ascontiguousarray(threeD_boxes[int(gt_labels[obj]) + 1, :, :], dtype=np.float32)

            tDbox = R_gt.dot(ori_points.T).T
            tDbox = tDbox + np.repeat(t_gt[:, np.newaxis], 8, axis=1).T  # * 0.001
            box3D = toPix_array(tDbox, fxkin, fykin, cxkin, cykin)
            tDbox = np.reshape(box3D, (16))
            tDbox = tDbox.astype(np.uint16)

            colGT = (245, 102, 65)

            image_pose = cv2.line(image_pose, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()), colGT, 2)
            image_pose = cv2.line(image_pose, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()), colGT, 2)
            image_pose = cv2.line(image_pose, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()), colGT,
                                  2)
            image_pose = cv2.line(image_pose, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()), colGT,
                                  2)
            image_pose = cv2.line(image_pose, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()), colGT,
                                  2)
            image_pose = cv2.line(image_pose, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()), colGT,
                                  2)
            image_pose = cv2.line(image_pose, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()), colGT,
                                  2)
            image_pose = cv2.line(image_pose, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()), colGT,
                                  2)
            image_pose = cv2.line(image_pose, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                                  colGT,
                                  2)
            image_pose = cv2.line(image_pose, tuple(tDbox[10:12].ravel()), tuple(tDbox[12:14].ravel()),
                                  colGT,
                                  2)
            image_pose = cv2.line(image_pose, tuple(tDbox[12:14].ravel()), tuple(tDbox[14:16].ravel()),
                                  colGT,
                                  2)
            image_pose = cv2.line(image_pose, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                                  colGT,
                                  2)

            gt_box = gt_boxes[obj, :]
            image_box = cv2.rectangle(image_box, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])),
                                      (245, 17, 50), 2)

        fxkin = gt_calib[0, 0]
        fykin = gt_calib[0, 1]
        cxkin = gt_calib[0, 2]
        cykin = gt_calib[0, 3]

        '''
        # run network
        start_t = time.time()
        t_error = 0
        t_img = 0
        n_img = 0
        boxes_raw, detections_raw, labels_raw, poses_raw = model.predict_on_batch(np.expand_dims(image, axis=0))
        t_img = time.time() - start_t

        labels_cls = np.argmax(labels_raw[0, :, :], axis=1)

        for inv_cls in np.unique(labels_cls):

            true_cls = inv_cls + 1
            n_img += 1

            if inv_cls not in gt_labels or true_cls in [3, 7]:
                continue

            cls_indices = np.where(labels_cls == inv_cls)
            labels_filt = labels_raw[0, cls_indices[0], inv_cls]
            point_votes = boxes_raw[0, cls_indices[0], inv_cls, :]
            detect_votes = detections_raw[0, cls_indices[0], inv_cls, :]
            direct_votes = poses_raw[0, cls_indices[0], inv_cls, :]
            above_thres = np.where(labels_filt > 0.5)

            mask_votes = cls_indices[0][above_thres]

            point_votes = point_votes[above_thres[0], :]
            direct_votes = direct_votes[above_thres[0], :]
            detect_votes = detect_votes[above_thres[0], :]
            labels_votes = labels_cls[cls_indices[0][above_thres[0]]]

            hyps = point_votes.shape[0]

            if hyps < 1:
                continue

            min_box_x = detect_votes[:, 0]
            min_box_y = detect_votes[:, 1]
            max_box_x = detect_votes[:, 2]
            max_box_y = detect_votes[:, 3]

            pos_anchors = np.stack([min_box_x, min_box_y, max_box_x, max_box_y], axis=1)

            # pos_anchors = anchor_params[cls_indices, :]

            ind_anchors = np.where(labels_votes == inv_cls)[0]
            # pos_anchors = pos_anchors[0]
            # ind_anchors = above_thres

            per_obj_hyps = []
            per_obj_cls = []
            per_obj_poses = []
            per_mask_hyps = []

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
                            if iou > 0.8:
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

                per_obj_hyps.append(obj_inds)
                per_obj_cls.append(inv_cls)

            for inst, hyps in enumerate(per_obj_hyps):
                inv_cls = per_obj_cls[inst]
                true_cls = inv_cls + 1
                box_votes = point_votes[hyps, :]
                dp_votes = direct_votes[hyps, :]
                det_votes = detect_votes[hyps, :]
                mask_now = mask_votes[hyps]
                hyps = box_votes.shape[0]

                col_box = (
                    int(np.random.uniform() * 255.0), int(np.random.uniform() * 255.0),
                    int(np.random.uniform() * 255.0))
                pyramids = np.zeros((6300, 3))
                pyramids[mask_now, :] = col_box
                P3_mask = np.reshape(pyramids[:4800, :], (60, 80, 3))
                P4_mask = np.reshape(pyramids[4800:6000, :], (30, 40, 3))
                P5_mask = np.reshape(pyramids[6000:, :], (15, 20, 3))
                P3_mask = cv2.resize(P3_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
                P4_mask = cv2.resize(P4_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
                P5_mask = cv2.resize(P5_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
                image_mask = np.where(P3_mask > 0, P3_mask, image_mask)
                image_mask = np.where(P4_mask > 0, P4_mask, image_mask)
                image_mask = np.where(P5_mask > 0, P5_mask, image_mask)

                ori_points = np.ascontiguousarray(threeD_boxes[true_cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))

                for pdx in range(dp_votes.shape[0]):
                    R_est = np.array(dp_votes[pdx, :9]).reshape((3, 3)).T
                    t_est = np.array(dp_votes[pdx, -3:]) * 0.001

                    eDbox = R_est.dot(ori_points.T).T
                    eDbox = eDbox + np.repeat(t_est[np.newaxis, :], 8, axis=0)  # * 0.001
                    est3D = toPix_array(eDbox, fxkin, fykin, cxkin, cykin)
                    eDbox = np.reshape(est3D, (16))
                    pose = eDbox.astype(np.uint16)
                    colEst = col_box

                    image_poses = cv2.line(image_poses, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 2)
                    image_poses = cv2.line(image_poses, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 2)
                    image_poses = cv2.line(image_poses, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 2)
                    image_poses = cv2.line(image_poses, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 2)
                    image_poses = cv2.line(image_poses, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 2)
                    image_poses = cv2.line(image_poses, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
                    image_poses = cv2.line(image_poses, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
                    image_poses = cv2.line(image_poses, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
                    image_poses = cv2.line(image_poses, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
                    image_poses = cv2.line(image_poses, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
                    image_poses = cv2.line(image_poses, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
                    image_poses = cv2.line(image_poses, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst, 2)

                print(box_votes.shape)
                for bdx in range(box_votes.shape[0]):
                    box = box_votes[bdx, :]
                    print(box.shape)
                    p_idx = 0
                    for jdx in range(box.shape[0]):
                        if p_idx > 14:
                            continue
                        cv2.circle(image_box, (int(box[p_idx]), int(box[p_idx+1])), 3, col_box, 3)
                        p_idx += 2

                image_points = cv2.rectangle(image_points, (int(det_votes[pdx, 0]), int(det_votes[pdx, 1])),
                                          (int(det_votes[pdx, 2]), int(det_votes[pdx, 3])), col_box, 2)

        name = '/home/stefan/PyraPose_viz/' + 'sample_' + str(index) + '.png'
        cv2.imwrite(name, image_raw)

        name = '/home/stefan/PyraPose_viz/' + 'box_' + str(index) + '.png'
        # image_row1 = np.concatenate([image, image_mask], axis=0)
        cv2.imwrite(name, image_box)

        name = '/home/stefan/PyraPose_viz/' + 'pose_' + str(index) + '.png'
        # image_row1 = np.concatenate([image, image_mask], axis=0)
        cv2.imwrite(name, image_poses)

        name = '/home/stefan/PyraPose_viz/' + 'mask_' + str(index) + '.png'
        # image_row1 = np.concatenate([image, image_mask], axis=0)
        cv2.imwrite(name, image_mask)

        name = '/home/stefan/PyraPose_viz/' + 'det_' + str(index) + '.png'
        # image_row1 = np.concatenate([image, image_mask], axis=0)
        cv2.imwrite(name, image_points)

        '''
        # run network
        start_t = time.time()
        t_error = 0
        t_img = 0
        n_img = 0
        scores, labels, poses, mask, boxes = model.predict_on_batch(np.expand_dims(image, axis=0))
        t_img = time.time() - start_t

        scores = scores[labels != -1]
        poses = poses[labels != -1]
        boxes = boxes[labels != -1]
        labels = labels[labels != -1]

        for odx, inv_cls in enumerate(labels):

            box = boxes[odx, :]
            true_cls = inv_cls + 1
            pose = poses[odx, :]
            if inv_cls not in gt_labels:
                continue
            n_img += 1

            R_est = np.array(pose[:9]).reshape((3, 3)).T
            t_est = np.array(pose[-3:]) * 0.001

            eval_line = []
            sc_id = int(scene_id[0])
            eval_line.append(sc_id)
            im_id = int(image_id)
            eval_line.append(im_id)
            obj_id = int(true_cls)
            eval_line.append(obj_id)
            score = float(scores[odx])
            eval_line.append(score)
            R_bop = [str(i) for i in R_est.flatten().tolist()]
            R_bop = ' '.join(R_bop)
            eval_line.append(R_bop)
            t_bop = t_est * 1000.0
            t_bop = [str(i) for i in t_bop.flatten().tolist()]
            t_bop = ' '.join(t_bop)
            eval_line.append(t_bop)
            time_bop = float(t_img)
            eval_line.append(time_bop)
            eval_img.append(eval_line)

            gt_idx = np.argwhere(gt_labels == inv_cls)
            gt_pose = gt_poses[gt_idx, :]
            gt_box = gt_boxes[gt_idx, :]
            gt_pose = gt_pose[0][0]
            gt_box = gt_box[0][0]

            ori_points = np.ascontiguousarray(threeD_boxes[true_cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
            K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

            if true_cls == 1:
                model_vsd = mv1
            elif true_cls == 2:
                model_vsd = mv2

            add_errors = []
            iou_ovlaps = []
            for gtdx in range(gt_poses.shape[0]):
                t_rot = tf3d.quaternions.quat2mat(gt_poses[gtdx, 3:])
                R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                t_gt = np.array(gt_poses[gtdx, :3], dtype=np.float32)
                t_gt = t_gt * 0.001

                err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                add_errors.append(err_add)

                iou = boxoverlap(est_box, gt_boxes[gtdx, :])
                iou_ovlaps.append(iou)

            idx_add = np.argmin(np.array(add_errors))
            err_add = add_errors[idx_add]
            gt_pose = gt_poses[idx_add, :]

            t_rot = tf3d.quaternions.quat2mat(gt_pose[3:])
            R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
            t_gt = np.array(gt_pose[:3], dtype=np.float32)
            t_gt = t_gt  # * 0.001

            if err_add < model_dia[true_cls] * 0.1:
                if np.max(gt_poses[idx_add, :]) != -1:
                    truePoses[true_cls] += 1
                    gt_poses[idx_add, :] = -1
            else:
                falsePoses[true_cls] += 1

            print(' ')
            print('error: ', err_add, 'threshold', model_dia[true_cls] * 0.1)

            # if gt_pose.size == 0:  # filter for benchvise, bowl and mug
            #    continue

            ori_points = np.ascontiguousarray(threeD_boxes[true_cls, :, :], dtype=np.float32)
            eDbox = R_est.dot(ori_points.T).T
            eDbox = eDbox + np.repeat(t_est[np.newaxis, :], 8, axis=0)  # * 0.001
            est3D = toPix_array(eDbox, fxkin, fykin, cxkin, cykin)
            eDbox = np.reshape(est3D, (16))
            pose = eDbox.astype(np.uint16)
            pose = np.where(pose < 3, 3, pose)

            colEst = (50, 205, 50)
            if err_add > model_dia[true_cls] * 0.1:
                colEst = (0, 39, 236)

            image_pose = cv2.line(image_pose, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 2)
            image_pose = cv2.line(image_pose, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 2)
            image_pose = cv2.line(image_pose, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 2)
            image_pose = cv2.line(image_pose, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 2)
            image_pose = cv2.line(image_pose, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 2)
            image_pose = cv2.line(image_pose, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
            image_pose = cv2.line(image_pose, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
            image_pose = cv2.line(image_pose, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
            image_pose = cv2.line(image_pose, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
            image_pose = cv2.line(image_pose, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
            image_pose = cv2.line(image_pose, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
            image_pose = cv2.line(image_pose, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst, 2)

            colEst = (50, 205, 50)
            if err_add > model_dia[true_cls] * 0.1:
                colEst = (25, 119, 242)

            colEst = colors_viz[true_cls -1, :]

            pts = model_vsd["pts"]
            proj_pts = R_est.dot(pts.T).T
            proj_pts = proj_pts + np.repeat(t_est[np.newaxis, :], pts.shape[0], axis=0)
            proj_pts = toPix_array(proj_pts, fxkin, fykin, cxkin, cykin)
            proj_pts = proj_pts.astype(np.uint16)
            proj_pts[:, 0] = np.where(proj_pts[:, 0] > 639, 0, proj_pts[:, 0])
            proj_pts[:, 0] = np.where(proj_pts[:, 0] < 0, 0, proj_pts[:, 0])
            proj_pts[:, 1] = np.where(proj_pts[:, 1] > 479, 0, proj_pts[:, 1])
            proj_pts[:, 1] = np.where(proj_pts[:, 1] < 0, 0, proj_pts[:, 1])
            image_rep[proj_pts[:, 1], proj_pts[:, 0], :] = colEst

            ###########################################
            # Detection
            ###########################################
            est_box = np.array([float(box[0]), float(box[1]), float(box[2]), float(box[3])])

            box_line = {
                "scene_id": sc_id,
                "image_id": im_id,
                "category_id": obj_id,
                "score": score,
                "bbox": [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
                # "segmentation": [],
                "time": time_bop,
            }
            eval_det.append(box_line)

            # bbox visualization
            image_box = cv2.rectangle(image_box, (int(est_box[0]), int(est_box[1])), (int(est_box[2]), int(est_box[3])),
                                      (42, 205, 50), 2)

        if index > 0:
                times[n_img] += t_img
                times_count[n_img] += 1

        if index % 1 == 0:
            #name = '/home/stefan/PyraPose_viz/' + 'icbin_raw_' + str(index) + '.png'
            #cv2.imwrite(name, image_raw)
            #name = '/home/stefan/PyraPose_viz/' + 'icbin_box_' + str(index) + '.png'
            #cv2.imwrite(name, image_box)
            #name = '/home/stefan/PyraPose_viz/' + 'icbin_pose_' + str(index) + '.png'
            #cv2.imwrite(name, image_pose)
            #name = '/home/stefan/PyraPose_viz/' + 'icbin_proj_' + str(index) + '.png'
            #cv2.imwrite(name, image_rep)
            #name = '/home/stefan/PyraPose_viz/' + 'icbin_mask_' + str(index) + '.png'
            #cv2.imwrite(name, image_mask)
            #name = '/home/stefan/PyraPose_viz/' + 'icbin_points_' + str(index) + '.png'
            #cv2.imwrite(name, image_points)

    #times
    print('Number of objects ----- t')
    for tfx in range(1, times.shape[0]):
        t_ins = times[tfx] / times_count[tfx]
        print(tfx, '       ------ ', t_ins, tfx)

    recall = np.zeros((3), dtype=np.float32)
    precision = np.zeros((3), dtype=np.float32)
    detections = np.zeros((3), dtype=np.float32)
    det_precision = np.zeros((3), dtype=np.float32)
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

    recall_all = np.sum(recall) / 2.0
    precision_all = np.sum(precision) / 2.0
    detections_all = np.sum(detections) / 2.0
    det_precision_all = np.sum(det_precision) / 2.0
    print('ALL: ')
    print('mean detection recall: ', detections_all)
    print('mean detection precision: ', det_precision_all)
    print('mean pose recall: ', recall_all)
    print('mean pose precision: ', precision_all)


    wd_path = os.getcwd()
    csv_target = os.path.join(wd_path, 'cope_icbin-test.csv')

    line_head = ['scene_id','im_id','obj_id','score','R','t','time']
    with open(csv_target, 'a') as outfile:
        myWriter = csv.writer(outfile, delimiter=',')  # Write out the Headers for the CSV file
        myWriter.writerow(line_head)

    for line_indexed in eval_img:
        with open(csv_target, 'a') as outfile:
            myWriter = csv.writer(outfile, delimiter=',')  # Write out the Headers for the CSV file
            myWriter.writerow(line_indexed)

    json_target = os.path.join(wd_path, 'cope_icbin-test.json')
    with open(json_target, 'w') as f:
        json.dump(eval_det, f)