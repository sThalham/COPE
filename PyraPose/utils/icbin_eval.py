
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

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


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
        model_dia[int(key)] = value['diameter'] #* fac

    # target annotation
    pc1, mv1 = load_pcd(data_path, '000001')
    pc2, mv2 = load_pcd(data_path, '000002')
    '''
    pc3, mv3 = load_pcd(data_path, '000003')
    pc4, mv4 = load_pcd(data_path, '000004')
    pc5, mv5 = load_pcd(data_path, '000005')
    pc6, mv6 = load_pcd(data_path, '000006')
    pc7, mv7 = load_pcd(data_path, '000007')
    pc8, mv8 = load_pcd(data_path, '000008')
    pc9, mv9 = load_pcd(data_path, '000009')
    pc10, mv10 = load_pcd(data_path, '000010')
    pc11, mv11 = load_pcd(data_path, '000011')
    pc12, mv12 = load_pcd(data_path, '000012')
    pc13, mv13 = load_pcd(data_path, '000013')
    pc14, mv14 = load_pcd(data_path, '000014')
    pc15, mv15 = load_pcd(data_path, '000015')
    pc16, mv16 = load_pcd(data_path, '000016')
    pc17, mv17 = load_pcd(data_path, '000017')
    pc18, mv18 = load_pcd(data_path, '000018')
    pc19, mv19 = load_pcd(data_path, '000019')
    pc20, mv20 = load_pcd(data_path, '000020')
    pc21, mv21 = load_pcd(data_path, '000021')
    pc22, mv22 = load_pcd(data_path, '000022')
    pc23, mv23 = load_pcd(data_path, '000023')
    pc24, mv24 = load_pcd(data_path, '000024')
    pc25, mv25 = load_pcd(data_path, '000025')
    pc26, mv26 = load_pcd(data_path, '000026')
    pc27, mv27 = load_pcd(data_path, '000027')
    pc28, mv28 = load_pcd(data_path, '000028')
    pc29, mv29 = load_pcd(data_path, '000029')
    pc30, mv30 = load_pcd(data_path, '000030')
    '''

    allPoses = np.zeros((31), dtype=np.uint32)
    truePoses = np.zeros((31), dtype=np.uint32)
    falsePoses = np.zeros((31), dtype=np.uint32)
    trueDets = np.zeros((31), dtype=np.uint32)
    falseDets = np.zeros((31), dtype=np.uint32)
    times = np.zeros((20), dtype=np.uint32)
    times_count = np.zeros((20), dtype=np.uint32)

    eval_img = []
    #for index in progressbar.progressbar(range(generator.size()), prefix='Tless evaluation: '):
    for index, sample in enumerate(generator):

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
        image_ori = image_raw.astype(np.uint8)

        image_mask = copy.deepcopy(image_raw)
        image_box = copy.deepcopy(image_raw)
        image_poses = copy.deepcopy(image_raw)

        # run network
        start_t = time.time()
        t_error = 0
        t_img = 0
        n_img = 0
        boxes3D, scores, labels, poses, consistency, mask = model.predict_on_batch(np.expand_dims(image, axis=0))
        print('t: ', time.time()- start_t)


        scores = scores[labels != -1]
        poses = poses[labels != -1]
        labels = labels[labels != -1]

        print('labels: ', labels)

        for odx, inv_cls in enumerate(labels):

            gt_idx = np.argwhere(gt_labels == inv_cls)
            gt_pose = gt_poses[gt_idx, :]

            if gt_pose.size == 0:  # filter for benchvise, bowl and mug
                continue

            true_cls = inv_cls + 1

            ori_points = np.ascontiguousarray(threeD_boxes[true_cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
            K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

            pose = poses[odx, :]

            R_est = np.array(pose[:9]).reshape((3, 3)).T
            t_est = np.array(pose[-3:])
            print('R: ', R_est)
            print('t: ', t_est)

            t_e_now = time.time()
            if inv_cls == 1:
                model_vsd = mv1
            elif inv_cls == 2:
                model_vsd = mv2

            add_errors = []
            for gtdx in range(gt_pose.shape[0]):
                t_rot = tf3d.quaternions.quat2mat(gt_pose[gtdx, 0, 3:])
                R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                t_gt = np.array(gt_pose[gtdx, 0, :3], dtype=np.float32)
                t_gt = t_gt  # * 0.001

                err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                add_errors.append(err_add)

            idx_add = np.argmin(np.array(add_errors))
            err_add = add_errors[idx_add]
            gt_pose = gt_pose[idx_add, 0, :]

            t_rot = tf3d.quaternions.quat2mat(gt_pose[3:])
            R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
            t_gt = np.array(gt_pose[:3], dtype=np.float32)
            t_gt = t_gt  # * 0.001

            # if gt_pose.size == 0:  # filter for benchvise, bowl and mug
            #    continue
            if err_add < model_dia[true_cls] * 0.1:
                if true_cls in gt_poses_list:
                    truePoses[true_cls] += 1
                    gt_poses_list.remove(true_cls)
            else:
                falsePoses[true_cls] += 1

            print(' ')
            # print('error: ', err_add, 'threshold', model_dia[true_cls] * 0.1)

            tDbox = R_gt.dot(ori_points.T).T
            tDbox = tDbox + np.repeat(t_gt[:, np.newaxis], 8, axis=1).T * 0.001
            box3D = toPix_array(tDbox, fxkin, fykin, cxkin, cykin)
            tDbox = np.reshape(box3D, (16))
            tDbox = tDbox.astype(np.uint16)

            eDbox = R_est.dot(ori_points.T).T
            eDbox = eDbox + np.repeat(t_est[np.newaxis, :], 8, axis=0) * 0.001
            est3D = toPix_array(eDbox, fxkin, fykin, cxkin, cykin)
            eDbox = np.reshape(est3D, (16))
            pose = eDbox.astype(np.uint16)
            colEst1 = (0, 145, 195)
            colEst = (0, 204, 0)
            if err_add > model_dia[true_cls] * 0.1:
                colEst = (0, 0, 255)

            # image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()), colGT, 2)
            # image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()), colGT, 2)
            # image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()), colGT,
            #                     2)
            # image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()), colGT,
            #                     2)
            # image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()), colGT,
            #                     2)
            # image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()), colGT,
            #                     2)
            # image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()), colGT,
            #                     2)
            # image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()), colGT,
            #                     2)
            # image_raw = cv2.line(image_raw, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
            #                     colGT,
            #                     2)
            # image_raw = cv2.line(image_raw, tuple(tDbox[10:12].ravel()), tuple(tDbox[12:14].ravel()),
            #                     colGT,
            #                     2)
            # image_raw = cv2.line(image_raw, tuple(tDbox[12:14].ravel()), tuple(tDbox[14:16].ravel()),
            #                     colGT,
            #                     2)
            # image_raw = cv2.line(image_raw, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
            #                     colGT,
            #                     2)

            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst, 3)

            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst, 3)
            image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst, 3)

        '''
        for inv_cls in np.unique(labels):

            true_cls = inv_cls + 1
            cls = true_cls

            pose_votes = boxes3D[labels == inv_cls]
            scores_votes = scores[labels == inv_cls]
            poses_votes = poses[labels == inv_cls, inv_cls]
            confs_votes = confs[labels == inv_cls]
            labels_votes = labels[labels == inv_cls]
            mask_votes = masks[labels == inv_cls]

            col_box = (
            int(np.random.uniform() * 255.0), int(np.random.uniform() * 255.0), int(np.random.uniform() * 255.0))
            pyramids = np.zeros((6300, 3))
            pyramids[mask_votes, :] = col_box
            P3_mask = np.reshape(pyramids[:4800, :], (60, 80, 3))
            P4_mask = np.reshape(pyramids[4800:6000, :], (30, 40, 3))
            P5_mask = np.reshape(pyramids[6000:, :], (15, 20, 3))
            P3_mask = cv2.resize(P3_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
            P4_mask = cv2.resize(P4_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
            P5_mask = cv2.resize(P5_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
            image_mask = np.where(P3_mask > 0, P3_mask, image_mask)
            image_mask = np.where(P4_mask > 0, P4_mask, image_mask)
            image_mask = np.where(P5_mask > 0, P5_mask, image_mask)

            #name = '/home/stefan/PyraPose_viz/' + 'sample_' + str(index) + '_' + str(cls) + '.png'
            #image_row1 = np.concatenate([image, image_mask], axis=0)
            #cv2.imwrite(name, image_row1)

            min_box_x = np.nanmin(pose_votes[:, ::2], axis=1)
            min_box_y = np.nanmin(pose_votes[:, 1::2], axis=1)
            max_box_x = np.nanmax(pose_votes[:, ::2], axis=1)
            max_box_y = np.nanmax(pose_votes[:, 1::2], axis=1)

            pos_anchors = np.stack([min_box_x, min_box_y, max_box_x, max_box_y], axis=1)

            # pos_anchors = anchor_params[cls_indices, :]

            ind_anchors = np.where(labels_votes == inv_cls)[0]
            # pos_anchors = pos_anchors[0]

            per_obj_hyps = []
            per_obj_cls = []
            per_obj_poses = []
            per_obj_hyps = []

            t_ins = time.time()
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
                per_obj_cls.append(cls)
            print('instance splitting time: ', t_ins - time.time())

            for inst, hyps in enumerate(per_obj_hyps):
                n_img += 1

                inv_cls = per_obj_cls[inst] - 1
                true_cls = inv_cls + 1
                gt_idx = np.argwhere(gt_labels == inv_cls)
                gt_pose = gt_poses[gt_idx, :]
                gt_box = gt_boxes[gt_idx, :]

                if gt_pose.size == 0:  # filter for benchvise, bowl and mug
                    continue

                col_box = (
                    int(np.random.uniform() * 255.0), int(np.random.uniform() * 255.0),
                    int(np.random.uniform() * 255.0))
                pyramids = np.zeros((6300, 3))
                pyramids[mask_votes[hyps], :] = col_box
                P3_mask = np.reshape(pyramids[:4800, :], (60, 80, 3))
                P4_mask = np.reshape(pyramids[4800:6000, :], (30, 40, 3))
                P5_mask = np.reshape(pyramids[6000:, :], (15, 20, 3))
                P3_mask = cv2.resize(P3_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
                P4_mask = cv2.resize(P4_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
                P5_mask = cv2.resize(P5_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
                image_mask = np.where(P3_mask > 0, P3_mask, image_mask)
                image_mask = np.where(P4_mask > 0, P4_mask, image_mask)
                image_mask = np.where(P5_mask > 0, P5_mask, image_mask)

                #gt_pose = gt_pose[0][0]
                #gt_box = gt_box[0][0]

                box_votes = pose_votes[hyps, :]
                k_hyp = box_votes.shape[0]

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
                t_est = t_est[0, :] * 1000.0
                #print(t_est)
                t_bop = t_est * 1000.0
                

                #t_rot = tf3d.quaternions.quat2mat(gt_pose[3:])
                #R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                #t_gt = np.array(gt_pose[:3], dtype=np.float32)
                #t_gt = t_gt  # * 0.001


                # direct pose regression
                direct_votes = poses_votes[hyps, :]
                direct_confs = confs_votes[hyps]

                n_hyps = 3
                if direct_confs.shape[0] < n_hyps:
                    n_hyps = direct_confs.shape[0]
                conf_ranks = np.argsort(direct_confs)
                poses_cls = np.mean(direct_votes[conf_ranks[:n_hyps], :], axis=0)

                # R6d
                R_est = np.eye(3)
                R_est[:3, 0] = poses_cls[3:6] / np.linalg.norm(poses_cls[3:6])
                R_est[:3, 1] = poses_cls[6:] / np.linalg.norm(poses_cls[6:])
                R3 = np.cross(R_est[:3, 0], poses_cls[6:])
                R_est[:3, 2] = R3 / np.linalg.norm(R3)
                t_est = poses_cls[:3]  # * 0.001

                t_e_now = time.time()
                if cls == 1:
                    model_vsd = mv1
                elif cls == 2:
                    model_vsd = mv2

                add_errors = []
                for gtdx in range(gt_pose.shape[0]):
                    t_rot = tf3d.quaternions.quat2mat(gt_pose[gtdx, 0, 3:])
                    R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                    t_gt = np.array(gt_pose[gtdx, 0, :3], dtype=np.float32)
                    t_gt = t_gt #* 0.001

                    err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                    add_errors.append(err_add)

                idx_add = np.argmin(np.array(add_errors))
                err_add = add_errors[idx_add]
                gt_pose = gt_pose[idx_add, 0, :]

                t_rot = tf3d.quaternions.quat2mat(gt_pose[3:])
                R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                t_gt = np.array(gt_pose[:3], dtype=np.float32)
                t_gt = t_gt  # * 0.001

                if err_add < model_dia[true_cls] * 0.1:
                    if true_cls in gt_poses_list:
                        truePoses[true_cls] += 1
                        gt_poses_list.remove(true_cls)
                else:
                    falsePoses[true_cls] += 1

                print(' ')
                #print('error: ', err_add, 'threshold', model_dia[true_cls] * 0.1)

                tDbox = R_gt.dot(ori_points.T).T
                tDbox = tDbox + np.repeat(t_gt[:, np.newaxis], 8, axis=1).T * 0.001
                box3D = toPix_array(tDbox, fxkin, fykin, cxkin, cykin)
                tDbox = np.reshape(box3D, (16))
                tDbox = tDbox.astype(np.uint16)

                eDbox = R_est.dot(ori_points.T).T
                eDbox = eDbox + np.repeat(t_est[np.newaxis, :], 8, axis=0) * 0.001
                est3D = toPix_array(eDbox, fxkin, fykin, cxkin, cykin)
                eDbox = np.reshape(est3D, (16))
                pose = eDbox.astype(np.uint16)
                colEst1 = (0, 145, 195)
                colEst = (0, 204, 0)
                #if err_add > model_dia[true_cls] * 0.1:
                #    colEst = (0, 0, 255)

                #image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()), colGT, 2)
                #image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()), colGT, 2)
                #image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()), colGT,
                #                     2)
                #image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()), colGT,
                #                     2)
                #image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()), colGT,
                #                     2)
                #image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()), colGT,
                #                     2)
                #image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()), colGT,
                #                     2)
                #image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()), colGT,
                #                     2)
                #image_raw = cv2.line(image_raw, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                #                     colGT,
                #                     2)
                #image_raw = cv2.line(image_raw, tuple(tDbox[10:12].ravel()), tuple(tDbox[12:14].ravel()),
                #                     colGT,
                #                     2)
                #image_raw = cv2.line(image_raw, tuple(tDbox[12:14].ravel()), tuple(tDbox[14:16].ravel()),
                #                     colGT,
                #                     2)
                #image_raw = cv2.line(image_raw, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                #                     colGT,
                #                     2)

                image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst, 3)

                image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst, 3)
                image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst, 3)

                for pdx, box_vote in enumerate(pose_votes[hyps, :]):

                    # direct pose
                    #pose_hyp = direct_votes[pdx, :]
                    #R_est = np.eye(3)
                    #R_est[:3, 0] = pose_hyp[3:6] / np.linalg.norm(pose_hyp[3:6])
                    #R_est[:3, 1] = pose_hyp[6:] / np.linalg.norm(pose_hyp[6:])
                    #R3 = np.cross(R_est[:3, 0], pose_hyp[6:])
                    #R_est[:3, 2] = R3 / np.linalg.norm(R3)
                    #t_est = pose_hyp[:3].T * 0.001
                    #t_bop = t_est * 1000.0

                    #print('tra: ', t_est)

                    #eDbox = R_est.dot(ori_points.T).T
                    #eDbox = eDbox + np.repeat(t_est[np.newaxis, :], 8, axis=0)
                    #est3D = toPix_array(eDbox, fxkin, fykin, cxkin, cykin)

                    # bounding box estimation
                    est3D = box_vote

                    # used for viz of both
                    eDbox = np.reshape(est3D, (16))
                    pose = eDbox.astype(np.uint16)
                    colGT = (255, 0, 0)
                    #colEst = (0, 204, 0)
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
                    image_poses = cv2.line(image_poses, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,2)
                    image_poses = cv2.line(image_poses, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,2)
                    image_poses = cv2.line(image_poses, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,2)

                eval_line = []
                R_bop = [str(i) for i in R_est.flatten().tolist()]
                R_bop = ' '.join(R_bop)
                eval_line.append(R_bop)
                t_bop = t_est * 1000.0
                t_bop = [str(i) for i in t_bop.flatten().tolist()]
                t_bop = ' '.join(t_bop)
                eval_line.append(t_bop)
                eval_img.append(eval_line)

                t_error += time.time() - t_e_now

        t_img = time.time() - start_t - t_error
        times[n_img] = t_img
        times_count[n_img] += 1
        print('t_now', t_img, n_img)
        '''

        name = '/home/stefan/PyraPose_viz/' + 'sample_' + str(index) + '.png'
        #image_row1 = np.concatenate([image_ori, image_raw], axis=1)
        #image_row2 = np.concatenate([image_mask, image_poses], axis=1)
        #image_rows = np.concatenate([image_row1, image_row2], axis=0)
        #cv2.imwrite(name, image_rows)
        cv2.imwrite(name, image_raw)

    #times
    print('Number of objects ----- t')
    for tfx in range(1, times.shape[0]):
        t_ins = times[tfx] / times_count[tfx]
        print(tfx, '       ------ ', t_ins)


    wd_path = os.getcwd()
    csv_target = os.path.join(wd_path, 'sthalham-pp_tless-test.csv')

    line_head = ['scene_id','im_id','obj_id','score','R','t','time']
    with open(csv_target, 'a') as outfile:
        myWriter = csv.writer(outfile, delimiter=',')  # Write out the Headers for the CSV file
        myWriter.writerow(line_head)

    for line_indexed in eval_img:
        line_indexed.append(str(t_eval))
        with open(csv_target, 'a') as outfile:
            myWriter = csv.writer(outfile, delimiter=',')  # Write out the Headers for the CSV file
            myWriter.writerow(line_indexed)

        '''
        if cls == 1:
            model_vsd = mv1
        elif cls == 2:
            model_vsd = mv2
        elif cls == 3:
            model_vsd = mv3
        elif cls == 4:
            model_vsd = mv4
        elif cls == 5:
            model_vsd = mv5
        elif cls == 6:
            model_vsd = mv6
        elif cls == 7:
            model_vsd = mv7
        elif cls == 8:
            model_vsd = mv8
        elif cls == 9:
            model_vsd = mv9
        elif cls == 10:
            model_vsd = mv10
        elif cls == 11:
            model_vsd = mv11
        elif cls == 12:
            model_vsd = mv12
        elif cls == 13:
            model_vsd = mv13
        elif cls == 14:
            model_vsd = mv14
        elif cls == 15:
            model_vsd = mv15
        elif cls == 16:
            model_vsd = mv16
        elif cls == 17:
            model_vsd = mv17
        elif cls == 18:
            model_vsd = mv18
        elif cls == 19:
            model_vsd = mv19
        elif cls == 20:
            model_vsd = mv20
        elif cls == 21:
            model_vsd = mv21
        elif cls == 22:
            model_vsd = mv22
        elif cls == 23:
            model_vsd = mv23
        elif cls == 24:
            model_vsd = mv24
        elif cls == 25:
            model_vsd = mv25
        elif cls == 26:
            model_vsd = mv26
        elif cls == 27:
            model_vsd = mv27
        elif cls == 28:
            model_vsd = mv28
        elif cls == 29:
            model_vsd = mv29
        elif cls == 30:
            model_vsd = mv30


                        cur_pose = gt_poses[odx[0]]
                        t_rot = cur_pose[0][3:]
                        t_tra = cur_pose[0][:3]

                        t_rot = tf3d.euler.euler2mat(t_rot[0], t_rot[1], t_rot[2])
                        R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                        t_gt = np.array(t_tra, dtype=np.float32) * 0.001

                        rd = re(R_gt, R_est)
                        xyz = te(t_gt, t_est.T)
                        #print(control_points)

                        if Visualization is True:
                            tDbox = R_gt.dot(obj_points.T).T
                            tDbox = tDbox + np.repeat(t_gt[np.newaxis, :], 8, axis=0)
                            box3D = toPix_array(tDbox, calib[0], calib[1], calib[2], calib[3])
                            tDbox = np.reshape(box3D, (16))
                            #print(tDbox)

                            #img = image
                            pose = est_points.reshape((16)).astype(np.int16)
                            bb = b1

                            colGT = (0, 128, 0)
                            colEst = (255, 0, 0)

                            image = cv2.line(image, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()), colGT, 5)
                            image = cv2.line(image, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()), colGT, 5)
                            image = cv2.line(image, tuple(tDbox[4:6].ravel()), tuple(pose[6:8].ravel()), colGT,
                            5)
                            image = cv2.line(image, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()), colGT,
                            5)
                            image = cv2.line(image, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()), colGT,
                            5)
                            image = cv2.line(image, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()), colGT,
                            5)
                            image = cv2.line(image, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()), colGT,
                            5)
                            image = cv2.line(image, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()), colGT,
                            5)
                            image = cv2.line(image, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                                         colGT,
                            5)
                            image = cv2.line(image, tuple(tDbox[10:12].ravel()), tuple(tDbox[12:14].ravel()),
                                         colGT,
                            5)
                            image = cv2.line(image, tuple(tDbox[12:14].ravel()), tuple(tDbox[14:16].ravel()),
                                         colGT,
                            5)
                            image = cv2.line(image, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                                         colGT,
                            5)
                            image = cv2.line(image, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 5)
                            image = cv2.line(image, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 5)
                            image = cv2.line(image, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 5)
                            image = cv2.line(image, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 5)
                            image = cv2.line(image, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 5)
                            image = cv2.line(image, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 5)
                            image = cv2.line(image, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 5)
                            image = cv2.line(image, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 5)
                            image = cv2.line(image, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                            5)
                            image = cv2.line(image, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                            5)
                            image = cv2.line(image, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                            5)
                            image = cv2.line(image, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                            5)
                            font = cv2.FONT_HERSHEY_COMPLEX
                            bottomLeftCornerOfText = (int(bb[0]) + 5, int(bb[3]) - 5)
                            fontScale = 0.5
                            fontColor = (0, 128, 0)
                            fontthickness = 2
                            lineType = 2

                            gtText = 'obj_' + str(cls)
                            # gtText = cate + " / " + str(detSco[i])
                            fontColor2 = (0, 0, 255)
                            fontthickness2 = 4
                            cv2.putText(image, gtText,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor2,
                            fontthickness2,
                            lineType)
                            cv2.putText(image, gtText,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            fontthickness,
                            lineType)

                        if cls == 1:
                            model_vsd = mv1
                            model_vsd_mm = mv1_mm
                        elif cls == 2:
                            model_vsd = mv2
                            model_vsd_mm = mv2_mm
                        elif cls == 3:
                            model_vsd = mv3
                            model_vsd_mm = mv3_mm
                        elif cls == 4:
                            model_vsd = mv4
                            model_vsd_mm = mv4_mm
                        elif cls == 5:
                            model_vsd = mv5
                            model_vsd_mm = mv5_mm
                        elif cls == 6:
                            model_vsd = mv6
                            model_vsd_mm = mv6_mm
                        elif cls == 7:
                            model_vsd = mv7
                            model_vsd_mm = mv7_mm
                        elif cls == 8:
                            model_vsd = mv8
                            model_vsd_mm = mv8_mm
                        elif cls == 9:
                            model_vsd = mv9
                            model_vsd_mm = mv9_mm
                        elif cls == 10:
                            model_vsd = mv10
                            model_vsd_mm = mv10_mm
                        elif cls == 11:
                            model_vsd = mv11
                            model_vsd_mm = mv11_mm
                        elif cls == 12:
                            model_vsd = mv12
                            model_vsd_mm = mv12_mm
                        elif cls == 13:
                            model_vsd = mv13
                            model_vsd_mm = mv13_mm
                        elif cls == 14:
                            model_vsd = mv14
                            model_vsd_mm = mv14_mm
                        elif cls == 15:
                            model_vsd = mv15
                            model_vsd_mm = mv15_mm
                        elif cls == 16:
                            model_vsd = mv16
                            model_vsd_mm = mv16_mm
                        elif cls == 17:
                            model_vsd = mv17
                            model_vsd_mm = mv17_mm
                        elif cls == 18:
                            model_vsd = mv18
                            model_vsd_mm = mv18_mm
                        elif cls == 19:
                            model_vsd = mv19
                            model_vsd_mm = mv19_mm
                        elif cls == 20:
                            model_vsd = mv20
                            model_vsd_mm = mv20_mm
                        elif cls == 21:
                            model_vsd = mv21
                            model_vsd_mm = mv21_mm
                        elif cls == 22:
                            model_vsd = mv22
                            model_vsd_mm = mv22_mm
                        elif cls == 23:
                            model_vsd = mv23
                            model_vsd_mm = mv23_mm
                        elif cls == 24:
                            model_vsd = mv24
                            model_vsd_mm = mv24_mm
                        elif cls == 25:
                            model_vsd = mv25
                            model_vsd_mm = mv25_mm
                        elif cls == 26:
                            model_vsd = mv26
                            model_vsd_mm = mv26_mm
                        elif cls == 27:
                            model_vsd = mv27
                            model_vsd_mm = mv27_mm
                        elif cls == 28:
                            model_vsd = mv28
                            model_vsd_mm = mv28_mm
                        elif cls == 29:
                            model_vsd = mv29
                            model_vsd_mm = mv29_mm
                        elif cls == 30:
                            model_vsd = mv30
                            model_vsd_mm = mv30_mm

                        if not math.isnan(rd):
                            if rd < 5.0 and xyz < 0.05:
                                less5[cls - 1] += 1

                        err_repr = reproj(K, R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                        #print('err_repr: ', err_repr)

                        if not math.isnan(err_repr):
                            if err_repr < 5.0:
                                rep_less5[cls - 1] += 1

                        image_dep = cv2.imread(image_dep_path, -1)
                        err_vsd = vsd(R_est, t_est*1000.0, R_gt, t_gt*1000.0, model_vsd, image_dep, K, 0.3, 20.0)
                        if not math.isnan(err_vsd):
                            if err_vsd < 0.3:
                                vsd_less_t[cls] += 1
                                vsd_true = True

                        err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                        #print('err_add: ', err_add, 'thres: ', model_dia[cls - 1] * 0.1)

                        if not math.isnan(err_add):
                            if err_add < (model_dia[cls - 1] * 0.05):
                                add_less_d005[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.1):
                                add_less_d[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.15):
                                add_less_d015[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.2):
                                add_less_d02[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.25):
                                add_less_d025[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.3):
                                add_less_d03[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.35):
                                add_less_d035[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.4):
                                add_less_d04[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.45):
                                add_less_d045[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.5):
                                add_less_d05[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.55):
                                add_less_d055[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.6):
                                add_less_d06[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.65):
                                add_less_d065[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.7):
                                add_less_d07[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.75):
                                add_less_d075[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.8):
                                add_less_d08[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.85):
                                add_less_d085[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.9):
                                add_less_d09[cls] += 1

                            if err_add < (model_dia[cls - 1] * 0.95):
                                add_less_d095[cls] += 1

                            if err_add < (model_dia[cls - 1] ):
                                add_less_d1[cls] += 1

                        if not math.isnan(err_add):
                            if err_add < (model_dia[cls - 1] * 0.15):
                                tp_add[cls] += 1
                                fn_add[cls] -= 1
                    else:
                        fp[cls] += 1
                        fp_add[cls] += 1

                        fp55[cls] += 1
                        fp6[cls] += 1
                        fp65[cls] += 1
                        fp7[cls] += 1
                        fp75[cls] += 1
                        fp8[cls] += 1
                        fp85[cls] += 1
                        fp9[cls] += 1
                        fp925[cls] += 1
                        fp95[cls] += 1
                        fp975[cls] += 1
                else:
                    fp[cls] += 1
                    fp_add[cls] += 1

                    fp55[cls] += 1
                    fp6[cls] += 1
                    fp65[cls] += 1
                    fp7[cls] += 1
                    fp75[cls] += 1
                    fp8[cls] += 1
                    fp85[cls] += 1
                    fp9[cls] += 1
                    fp925[cls] += 1
                    fp95[cls] += 1
                    fp975[cls] += 1

#1293 1307 1361
        if Visualization is True:
            if vsd_true == True:
                name = '/home/sthalham/visTests/detected_Tless.jpg'
                cv2.imwrite(name, image)
                print('stop')

        # append image to list of processed images
        image_ids.append(generator.image_ids[index])
        image_indices.append(index)
        idx += 1

    print(len(image_ids))

    if not len(results):
        return

    # write output
    json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
    #json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

    detPre = [0.0] * 31
    detRec = [0.0] * 31
    detPre_add = [0.0] * 31
    detRec_add = [0.0] * 31
    F1_add = [0.0] * 31
    less_55 = [0.0] * 31
    less_repr_5 = [0.0] * 31
    less_add_d = [0.0] * 31
    less_vsd_t = [0.0] * 31

    less_add_d005 = [0.0] * 31
    less_add_d015 = [0.0] * 31
    less_add_d02 = [0.0] * 31
    less_add_d025 = [0.0] * 31
    less_add_d03 = [0.0] * 31
    less_add_d035 = [0.0] * 31
    less_add_d04 = [0.0] * 31
    less_add_d045 = [0.0] * 31
    less_add_d05 = [0.0] * 31
    less_add_d055 = [0.0] * 31
    less_add_d06 = [0.0] * 31
    less_add_d065 = [0.0] * 31
    less_add_d07 = [0.0] * 31
    less_add_d075 = [0.0] * 31
    less_add_d08 = [0.0] * 31
    less_add_d085 = [0.0] * 31
    less_add_d09 = [0.0] * 31
    less_add_d095 = [0.0] * 31
    less_add_d1 = [0.0] * 31

    np.set_printoptions(precision=2)
    print('')
    for ind in range(1, 31):
        if ind == 0:
            continue

        else:
            detRec[ind] = tp[ind] / (tp[ind] + fn[ind]) * 100.0
            detPre[ind] = tp[ind] / (tp[ind] + fp[ind]) * 100.0
            detRec_add[ind] = tp_add[ind] / (tp_add[ind] + fn_add[ind]) * 100.0
            detPre_add[ind] = tp_add[ind] / (tp_add[ind] + fp_add[ind]) * 100.0
            F1_add[ind] = 2 * ((detPre_add[ind] * detRec_add[ind])/(detPre_add[ind] + detRec_add[ind]))
            less_55[ind] = (less5[ind]) / (rotD[ind]) * 100.0
            less_repr_5[ind] = (rep_less5[ind]) / (rep_e[ind]) * 100.0
            less_add_d[ind] = (add_less_d[ind]) / (add_e[ind]) * 100.0
            less_vsd_t[ind] = (vsd_less_t[ind]) / (vsd_e[ind]) * 100.0

            less_add_d005[ind] = (add_less_d005[ind]) / (add_e[ind]) * 100.0
            less_add_d015[ind] = (add_less_d015[ind]) / (add_e[ind]) * 100.0
            less_add_d02[ind] = (add_less_d02[ind]) / (add_e[ind]) * 100.0
            less_add_d025[ind] = (add_less_d025[ind]) / (add_e[ind]) * 100.0
            less_add_d03[ind] = (add_less_d03[ind]) / (add_e[ind]) * 100.0
            less_add_d035[ind] = (add_less_d035[ind]) / (add_e[ind]) * 100.0
            less_add_d04[ind] = (add_less_d04[ind]) / (add_e[ind]) * 100.0
            less_add_d045[ind] = (add_less_d045[ind]) / (add_e[ind]) * 100.0
            less_add_d05[ind] = (add_less_d05[ind]) / (add_e[ind]) * 100.0
            less_add_d055[ind] = (add_less_d055[ind]) / (add_e[ind]) * 100.0
            less_add_d06[ind] = (add_less_d06[ind]) / (add_e[ind]) * 100.0
            less_add_d065[ind] = (add_less_d065[ind]) / (add_e[ind]) * 100.0
            less_add_d07[ind] = (add_less_d07[ind]) / (add_e[ind]) * 100.0
            less_add_d075[ind] = (add_less_d075[ind]) / (add_e[ind]) * 100.0
            less_add_d08[ind] = (add_less_d08[ind]) / (add_e[ind]) * 100.0
            less_add_d085[ind] = (add_less_d085[ind]) / (add_e[ind]) * 100.0
            less_add_d09[ind] = (add_less_d09[ind]) / (add_e[ind]) * 100.0
            less_add_d095[ind] = (add_less_d095[ind]) / (add_e[ind]) * 100.0
            less_add_d1[ind] = (add_less_d1[ind]) / (add_e[ind]) * 100.0

            print('cat', ind)
            print('add < 0.05: ', less_add_d005[ind])
            print('add < 0.1: ', less_add_d[ind])
            print('add < 0.15: ', less_add_d015[ind])
            print('add < 0.2: ', less_add_d02[ind])
            print('add < 0.25: ', less_add_d025[ind])
            print('add < 0.3: ', less_add_d03[ind])
            print('add < 0.35: ', less_add_d035[ind])
            print('add < 0.4: ', less_add_d04[ind])
            print('add < 0.45: ', less_add_d045[ind])
            print('add < 0.5: ', less_add_d05[ind])
            print('add < 0.55: ', less_add_d055[ind])
            print('add < 0.6: ', less_add_d06[ind])
            print('add < 0.65: ', less_add_d065[ind])
            print('add < 0.7: ', less_add_d07[ind])
            print('add < 0.75: ', less_add_d075[ind])
            print('add < 0.8: ', less_add_d08[ind])
            print('add < 0.85: ', less_add_d085[ind])
            print('add < 0.9: ', less_add_d09[ind])
            print('add < 0.95: ', less_add_d095[ind])
            print('add < 1: ', less_add_d1[ind])

        print('cat ', ind, ' rec ', detPre[ind], ' pre ', detRec[ind], ' less5 ', less_55[ind], ' repr ',
                  less_repr_5[ind], ' add ', less_add_d[ind], ' vsd ', less_vsd_t[ind], ' F1 add 0.15d ', F1_add[ind])

    dataset_recall = sum(tp) / (sum(tp) + sum(fp)) * 100.0
    dataset_precision = sum(tp) / (sum(tp) + sum(fn)) * 100.0
    dataset_recall_add = sum(tp_add) / (sum(tp_add) + sum(fp_add)) * 100.0
    dataset_precision_add = sum(tp_add) / (sum(tp_add) + sum(fn_add)) * 100.0
    F1_add_all = 2 * ((dataset_precision_add * dataset_recall_add)/(dataset_precision_add + dataset_recall_add))
    less_55 = sum(less5) / sum(rotD) * 100.0
    less_repr_5 = sum(rep_less5) / sum(rep_e) * 100.0
    less_add_d = sum(add_less_d) / sum(add_e) * 100.0
    less_vsd_t = sum(vsd_less_t) / sum(vsd_e) * 100.0

    print('IoU 05: ', sum(tp) / (sum(tp) + sum(fp)) * 100.0, sum(tp) / (sum(tp) + sum(fn)) * 100.0)
    print('IoU 055: ', sum(tp55) / (sum(tp55) + sum(fp55)) * 100.0, sum(tp55) / (sum(tp55) + sum(fn55)) * 100.0)
    print('IoU 06: ', sum(tp6) / (sum(tp6) + sum(fp6)) * 100.0, sum(tp6) / (sum(tp6) + sum(fn6)) * 100.0)
    print('IoU 065: ', sum(tp65) / (sum(tp65) + sum(fp65)) * 100.0, sum(tp65) / (sum(tp65) + sum(fn65)) * 100.0)
    print('IoU 07: ', sum(tp7) / (sum(tp7) + sum(fp7)) * 100.0, sum(tp7) / (sum(tp7) + sum(fn7)) * 100.0)
    print('IoU 075: ', sum(tp75) / (sum(tp75) + sum(fp75)) * 100.0, sum(tp75) / (sum(tp75) + sum(fn75)) * 100.0)
    print('IoU 08: ', sum(tp8) / (sum(tp8) + sum(fp8)) * 100.0, sum(tp8) / (sum(tp8) + sum(fn8)) * 100.0)
    print('IoU 085: ', sum(tp85) / (sum(tp85) + sum(fp85)) * 100.0, sum(tp85) / (sum(tp85) + sum(fn85)) * 100.0)
    print('IoU 09: ', sum(tp9) / (sum(tp9) + sum(fp9)) * 100.0, sum(tp9) / (sum(tp9) + sum(fn9)) * 100.0)
    print('IoU 0975: ', sum(tp925) / (sum(tp925) + sum(fp925)) * 100.0, sum(tp925) / (sum(tp925) + sum(fn925)) * 100.0)
    print('IoU 095: ', sum(tp95) / (sum(tp95) + sum(fp95)) * 100.0, sum(tp95) / (sum(tp95) + sum(fn95)) * 100.0)
    print('IoU 0975: ', sum(tp975) / (sum(tp975) + sum(fp975)) * 100.0, sum(tp975) / (sum(tp975) + sum(fn975)) * 100.0)

    return dataset_recall, dataset_precision, less_55, less_vsd_t, less_repr_5, less_add_d, F1_add_all
'''