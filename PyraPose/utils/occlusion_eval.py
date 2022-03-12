
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
    model_vsd['pts'] = np.asarray(pcd_model.points) * 0.001

    pcd_down = pcd_model.voxel_down_sample(voxel_size=3)
    model_down = {}
    model_down['pts'] = np.asarray(pcd_down.points) * 0.001

    return pcd_model, model_vsd, model_down
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

    inv_key = 1
    for key, value in json.load(open(mesh_info)).items():
        if key not in [1, 5, 6, 8, 9, 10, 11, 12]:
            continue
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
        threeD_boxes[int(inv_key), :, :] = three_box_solo
        model_dia[int(inv_key)] = value['diameter'] * fac
        inv_key += 1

    pc1, mv1, md1 = load_pcd(data_path, '000001')
    pc5, mv5, md5 = load_pcd(data_path, '000005')
    pc6, mv6, md6 = load_pcd(data_path, '000006')
    pc8, mv8, md8 = load_pcd(data_path, '000008')
    pc9, mv9, md9 = load_pcd(data_path, '000009')
    pc10, mv10, md10 = load_pcd(data_path, '000010')
    pc11, mv11, md11 = load_pcd(data_path, '000011')
    pc12, mv12, md12 = load_pcd(data_path, '000012')

    allPoses = np.zeros((16), dtype=np.uint32)
    truePoses = np.zeros((16), dtype=np.uint32)
    falsePoses = np.zeros((16), dtype=np.uint32)
    trueDets = np.zeros((16), dtype=np.uint32)
    falseDets = np.zeros((16), dtype=np.uint32)
    times = np.zeros((40), dtype=np.float32)
    times_count = np.zeros((40), dtype=np.float32)

    colors_viz = np.random.randint(255, size=(15, 3))
    inv_keys = [1, 5, 6, 8, 9, 10, 11, 12]

    eval_img = []
    for index, sample in enumerate(generator):

        scene_id = sample[0].numpy()
        image_id = sample[1].numpy()
        image = sample[2]
        gt_labels = sample[3].numpy()
        gt_boxes = sample[4].numpy()
        gt_poses = sample[5].numpy()
        gt_calib = sample[6].numpy()
        allLabels = copy.deepcopy(gt_labels)

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
        image_box = copy.deepcopy(image_raw)
        image_poses = copy.deepcopy(image_raw)

        for obj in range(gt_labels.shape[0]):
            allPoses[int(gt_labels[obj]) + 1] += 1

            t_rot = tf3d.quaternions.quat2mat(gt_poses[obj, 3:])
            R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
            t_gt = np.array(gt_poses[obj, :3], dtype=np.float32)
            t_gt = t_gt * 0.001

            '''
            if int(gt_labels[obj]) != 5:
                continue

            if int(gt_labels[obj]) + 1 == 1:
                model_vsd = md1
            elif int(gt_labels[obj]) + 1 == 5:
                model_vsd = md5
            elif int(gt_labels[obj]) + 1 == 6:
                model_vsd = md6
            elif int(gt_labels[obj]) + 1 == 8:
                model_vsd = md8
            elif int(gt_labels[obj]) + 1 == 9:
                model_vsd = md9
            elif int(gt_labels[obj]) + 1 == 10:
                model_vsd = md10
            elif int(gt_labels[obj]) + 1 == 11:
                model_vsd = md11
            elif int(gt_labels[obj]) + 1 == 12:
                model_vsd = md12

            pts = model_vsd["pts"]
            print(pts.shape)
            proj_pts = R_gt.dot(pts.T).T
            proj_pts = proj_pts + np.repeat(t_gt[np.newaxis, :], pts.shape[0], axis=0)
            proj_pts = toPix_array(proj_pts, fxkin, fykin, cxkin, cykin)
            proj_pts[:, 0] = np.where(proj_pts[:, 0] > 639, 0, proj_pts[:, 0])
            proj_pts[:, 0] = np.where(proj_pts[:, 0] < 0, 0, proj_pts[:, 0])
            proj_pts[:, 1] = np.where(proj_pts[:, 1] > 479, 0, proj_pts[:, 1])
            proj_pts[:, 1] = np.where(proj_pts[:, 1] < 0, 0, proj_pts[:, 1])
            proj_pts = proj_pts.astype(np.uint16)
            image_raw[proj_pts[:, 1], proj_pts[:, 0], :] = (245, 102, 65)

            
            '''
            ori_points = np.ascontiguousarray(threeD_boxes[int(gt_labels[obj])+1, :, :], dtype=np.float32)
            tDbox = R_gt.dot(ori_points.T).T
            tDbox = tDbox + np.repeat(t_gt[:, np.newaxis], 8, axis=1).T  # * 0.001
            box3D = toPix_array(tDbox, fxkin, fykin, cxkin, cykin)
            tDbox = np.reshape(box3D, (16))
            tDbox = tDbox.astype(np.uint16)
            tDbox = np.where(tDbox < 3, 3, tDbox)

            colGT = (245, 102, 65)

            #image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()), colGT, 2)
            #image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()), colGT, 2)
            #image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()), colGT,
            #                    2)
            #image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()), colGT,
            #                    2)
            #image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()), colGT,
            #                    2)
            #image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()), colGT,
            #                    2)
            #image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()), colGT,
            #                    2)
            #image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()), colGT,
            #                    2)
            #image_raw = cv2.line(image_raw, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
            #                    colGT,
            #                    2)
            #image_raw = cv2.line(image_raw, tuple(tDbox[10:12].ravel()), tuple(tDbox[12:14].ravel()),
            #                    colGT,
            #                    2)
            #image_raw = cv2.line(image_raw, tuple(tDbox[12:14].ravel()), tuple(tDbox[14:16].ravel()),
            #                    colGT,
            #                    2)
            #image_raw = cv2.line(image_raw, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
            #                    colGT,
            #                    2)

        # run network
        start_t = time.time()
        t_error = 0
        t_img = 0
        n_img = 0

        boxes_raw, labels_raw, poses_raw, scores, labels, poses, mask = model.predict_on_batch(np.expand_dims(image, axis=0))
        t_img = time.time() - start_t

        '''
        #################################
        # viz error cases
        labels_cls = np.argmax(labels_raw[0, :, :], axis=1)

        for inv_cls in np.unique(labels_cls):

            true_cls = inv_cls + 1
            n_img += 1

            if inv_cls not in gt_labels or true_cls in [3, 7]:
                continue

            cls_indices = np.where(labels_cls == inv_cls)
            labels_filt = labels_raw[0, cls_indices[0], inv_cls]
            point_votes = boxes_raw[0, cls_indices[0], inv_cls, :]
            direct_votes = poses_raw[0, cls_indices[0], inv_cls, :]
            above_thres = np.where(labels_filt > 0.25)

            mask_votes = cls_indices[0][above_thres]

            point_votes = point_votes[above_thres[0], :]
            direct_votes = direct_votes[above_thres[0], :]
            labels_votes = labels_cls[cls_indices[0][above_thres[0]]]

            hyps = point_votes.shape[0]

            if hyps < 1:
                continue

            min_box_x = np.nanmin(point_votes[:, ::2], axis=1)
            min_box_y = np.nanmin(point_votes[:, 1::2], axis=1)
            max_box_x = np.nanmax(point_votes[:, ::2], axis=1)
            max_box_y = np.nanmax(point_votes[:, 1::2], axis=1)

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
                    image_poses = cv2.line(image_poses, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                                           2)
                    image_poses = cv2.line(image_poses, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                                           2)
                    image_poses = cv2.line(image_poses, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                                           2)
                    image_poses = cv2.line(image_poses, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                                           2)

                print(box_votes.shape)
                for bdx in range(box_votes.shape[0]):
                    box = box_votes[bdx, :]
                    print(box.shape)
                    p_idx = 0
                    for jdx in range(box.shape[0]):
                        if p_idx > 14:
                            continue
                        cv2.circle(image_box, (int(box[p_idx]), int(box[p_idx + 1])), 3, col_box, 3)
                        p_idx += 2

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

        # end of error case viz
        ########################################
        '''

        scores = scores[labels != -1]
        poses = poses[labels != -1]
        labels = labels[labels != -1]
        print('est: ', labels)
        print('gt: ', gt_labels)

        for odx, inv_cls in enumerate(labels):

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


            # detection
            min_x = int(np.nanmin(pose[::2], axis=0))
            min_y = int(np.nanmin(pose[1::2], axis=0))
            max_x = int(np.nanmax(pose[::2], axis=0))
            max_y = int(np.nanmax(pose[1::2], axis=0))
            est_box = np.array([float(min_x), float(min_y), float(max_x), float(max_y)])
            
            t_rot = tf3d.quaternions.quat2mat(gt_pose[3:])
            R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
            t_gt = np.array(gt_pose[:3], dtype=np.float32)
            t_gt = t_gt * 0.001
            
            if true_cls == 1:
                model_vsd = mv1
            elif true_cls == 2: #5:
                model_vsd = mv5
            elif true_cls == 3: #6:
                model_vsd = mv6
            elif true_cls == 4: #8:
                model_vsd = mv8
            elif true_cls == 5: #9:
                model_vsd = mv9
            elif true_cls == 6: #10:
                model_vsd = mv10
            elif true_cls == 7: #11:
                model_vsd = mv11
            elif true_cls == 8: #12:
                model_vsd = mv12

            add_errors = []
            iou_ovlaps = []

            #if true_cls == 10 or true_cls == 11:
            if true_cls == 6 or true_cls == 7:
                err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
            else:
                err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

            if err_add < model_dia[true_cls] * 0.1:
                if np.max(gt_poses[gt_idx, :]) != -1:
                    truePoses[true_cls] += 1
                    gt_poses[gt_idx, :] = -1
            else:
                falsePoses[true_cls] += 1
                
            if inv_cls in allLabels:
                trueDets[true_cls] +=1
                allLabels[gt_idx] = -1

            print(' ')
            print('error: ', err_add, 'threshold', model_dia[true_cls] * 0.1)

            # if gt_pose.size == 0:  # filter for benchvise, bowl and mug
            #    continue

            #idx_iou = np.argmax(np.array(iou_ovlaps))
            #iou_ov = iou_ovlaps[idx_iou]

            #if iou_ov > 0.7 and np.max(gt_boxes[idx_iou, :]) != -1:
            #    trueDets[true_cls] += 1
            #    gt_boxes[idx_add, :] = -1
            #else:
            #    falseDets[true_cls] += 1

            ori_points = np.ascontiguousarray(threeD_boxes[true_cls, :, :], dtype=np.float32)
            eDbox = R_est.dot(ori_points.T).T
            eDbox = eDbox + np.repeat(t_est[np.newaxis, :], 8, axis=0) #* 0.001
            est3D = toPix_array(eDbox, fxkin, fykin, cxkin, cykin)
            eDbox = np.reshape(est3D, (16))
            pose = eDbox.astype(np.uint16)
            pose = np.where(pose < 3, 3, pose)

            colEst = (50, 205, 50)
            if err_add > model_dia[true_cls] * 0.1:
                colEst = (0, 39, 236)

            #image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 2)
            #image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 2)
            #image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 2)
            #image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 2)
            #image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 2)
            #image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
            #image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
            #image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
            #image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
            #image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
            #image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
            #image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst, 2)

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
            image_ori[proj_pts[:, 1], proj_pts[:, 0], :] = colEst
        '''

        boxes3D, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        t_img = time.time() - start_t

        labels_cls = np.argmax(labels[0, :, :], axis=1)

        for inv_cls in np.unique(labels_cls):

            true_cls = inv_cls + 1
            n_img += 1

            if inv_cls not in gt_labels or true_cls in [3, 7]:
                continue

            cls_indices = np.where(labels_cls == inv_cls)
            print(labels.shape)
            print(cls_indices[0].shape)
            labels_filt = labels[0, cls_indices[0], inv_cls]
            pose_votes = boxes3D[0, cls_indices[0], inv_cls, :]
            above_thres = np.where(labels_filt > 0.25)

            pose_votes = pose_votes[above_thres[0], :]
            labels_votes = labels_cls[cls_indices[0][above_thres[0]]]

            hyps = pose_votes.shape[0]
            print(pose_votes.shape)
            print(labels_votes.shape)
            if hyps < 1:
                continue

            min_box_x = np.nanmin(pose_votes[:, ::2], axis=1)
            min_box_y = np.nanmin(pose_votes[:, 1::2], axis=1)
            max_box_x = np.nanmax(pose_votes[:, ::2], axis=1)
            max_box_y = np.nanmax(pose_votes[:, 1::2], axis=1)

            pos_anchors = np.stack([min_box_x, min_box_y, max_box_x, max_box_y], axis=1)

            # pos_anchors = anchor_params[cls_indices, :]

            ind_anchors = np.where(labels_votes == inv_cls)[0]
            # pos_anchors = pos_anchors[0]
            #ind_anchors = above_thres

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
                            if iou > 0.5:
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
                box_votes = pose_votes[hyps, :]
                hyps = box_votes.shape[0]

                print(inv_cls)
                print(box_votes)

                ori_points = np.ascontiguousarray(threeD_boxes[true_cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
                obj_points = np.repeat(ori_points[np.newaxis, :, :], hyps, axis=0)
                obj_points = obj_points.reshape((int(hyps * 8), 1, 3))
                K = np.float64([float(fxkin), 0., float(cxkin), 0., float(fykin), float(cykin), 0., 0., 1.]).reshape(3, 3)

                est_points = np.ascontiguousarray(box_votes, dtype=np.float32).reshape((hyps * 8, 2))
                # est_points = pose_votes.reshape((hyps * 8, 1, 2))

                retval, orvec, otvec, _ = cv2.solvePnPRansac(objectPoints=obj_points,
                                                             imagePoints=est_points, cameraMatrix=K,
                                                             distCoeffs=None, rvec=None, tvec=None,
                                                             useExtrinsicGuess=False, iterationsCount=300,
                                                             reprojectionError=5.0, confidence=0.99,
                                                             flags=cv2.SOLVEPNP_EPNP)
                R_est, _ = cv2.Rodrigues(orvec)
                t_est = otvec[:, 0]

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

                add_errors = []
                iou_ovlaps = []

                #for gtdx in range(gt_poses.shape[0]):
                #    t_rot = tf3d.quaternions.quat2mat(gt_poses[gtdx, 3:])
                #    R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                #    t_gt = np.array(gt_poses[gtdx, :3], dtype=np.float32)
                #    t_gt = t_gt * 0.001

                #    if true_cls == 10 or true_cls == 11:
                #        err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                #    else:
                #        err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                #    add_errors.append(err_add)

                    #iou = boxoverlap(est_box, gt_boxes[gtdx, :])
                    #iou_ovlaps.append(iou)

                #idx_add = np.argmin(np.array(add_errors))
                idx_add = np.where(gt_labels==inv_cls)
                #err_add = add_errors[idx_add]
                gt_pose = gt_poses[idx_add, :][0][0]

                t_rot = tf3d.quaternions.quat2mat(gt_pose[3:])
                R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                t_gt = np.array(gt_pose[:3], dtype=np.float32)
                t_gt = t_gt * 0.001

                if true_cls == 10 or true_cls == 11:
                    err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                else:
                    err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                if err_add < model_dia[true_cls] * 0.1:
                    if np.max(gt_poses[idx_add, :]) != -1:
                        truePoses[true_cls] += 1
                        gt_poses[idx_add, :] = -1
                else:
                    falsePoses[true_cls] += 1

                print(' ')
                print('error: ', err_add, 'threshold', model_dia[true_cls] * 0.1)

                #print('ori: ', ori_points.T.shape)
                #eDbox = R_est.dot(ori_points.T).T
                #eDbox = eDbox + np.repeat(t_est[np.newaxis, :], 8, axis=0) #* 0.001
                #print('edbox: ', eDbox.shape)
                #est3D = toPix_array(eDbox, fxkin, fykin, cxkin, cykin)
                #eDbox = np.reshape(est3D, (16))
                #pose = eDbox.astype(np.uint16)

                #colEst = (50, 205, 50)
                #if err_add > model_dia[true_cls] * 0.1:
                #    colEst = (0, 39, 236)

                #image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 1)
                #image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 1)
                #image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 1)
                #image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 1)
                #image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 1)
                #image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 1)
                #image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 1)
                #image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 1)
                #image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst, 1)
                #image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst, 1)
                #image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst, 1)
                #image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst, 1)

                if true_cls != 6:
                    continue

                if true_cls == 1:
                    model_vsd = md1
                elif true_cls == 5:
                    model_vsd = md5
                elif true_cls == 6:
                    model_vsd = md6
                elif true_cls == 8:
                    model_vsd = md8
                elif true_cls == 9:
                    model_vsd = md9
                elif true_cls == 10:
                    model_vsd = md10
                elif true_cls == 11:
                    model_vsd = md11
                elif true_cls == 12:
                    model_vsd = md12

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
                image_ori[proj_pts[:, 1], proj_pts[:, 0], :] = colEst
                '''

        if index > 0:
            times[n_img] += t_img
            times_count[n_img] += 1

        #name = '/home/stefan/PyraPose_viz/' + 'sample_' + str(index) + '.png'
        #image_row1 = np.concatenate([image_ori, image_raw], axis=1)
        #image_row2 = np.concatenate([image_mask, image_poses], axis=1)
        #image_rows = np.concatenate([image_row1, image_row2], axis=0)
        #cv2.imwrite(name, image_rows)
        #cv2.imwrite(name, image_raw)
        name = '/home/stefan/PyraPose_viz/' + 'ori_' + str(index) + '.png'
        # cv2.imwrite(name, image_rows)
        #cv2.imwrite(name, image_ori)

    #times
    print('Number of objects ----- t')
    for tfx in range(1, times.shape[0]):
        t_ins = times[tfx] / times_count[tfx]
        print(tfx, '       ------ ', t_ins, tfx)

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

    wd_path = os.getcwd()
    csv_target = os.path.join(wd_path, 'cope_lmo-test.csv')

    line_head = ['scene_id','im_id','obj_id','score','R','t','time']
    with open(csv_target, 'a') as outfile:
        myWriter = csv.writer(outfile, delimiter=',')  # Write out the Headers for the CSV file
        myWriter.writerow(line_head)

    for line_indexed in eval_img:
        with open(csv_target, 'a') as outfile:
            myWriter = csv.writer(outfile, delimiter=',')  # Write out the Headers for the CSV file
            myWriter.writerow(line_indexed)
