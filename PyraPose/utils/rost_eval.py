
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


def evaluate_rost(generator, model, data_path, threshold=0.5):

    mesh_info = os.path.join(data_path, "meshes/models_info.json")
    threeD_boxes = np.ndarray((7, 8, 3), dtype=np.float32)
    model_dia = np.zeros((7), dtype=np.float32)

    for key, value in json.load(open(mesh_info)).items():
        #if int(key) not in [1, 5, 6, 8, 9, 10, 11, 12]:
        #    continue
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
        threeD_boxes[int(key), :, :] = three_box_solo
        model_dia[int(key)] = value['diameter'] * fac

    pc1, mv1, md1 = load_pcd(data_path, '000001')
    pc2, mv2, md2 = load_pcd(data_path, '000002')
    pc3, mv3, md3 = load_pcd(data_path, '000003')
    pc4, mv4, md4 = load_pcd(data_path, '000004')
    pc5, mv5, md5 = load_pcd(data_path, '000005')
    pc6, mv6, md6 = load_pcd(data_path, '000006')

    allPoses = np.zeros((7), dtype=np.uint32)
    truePoses = np.zeros((7), dtype=np.uint32)
    falsePoses = np.zeros((7), dtype=np.uint32)
    trueDets = np.zeros((7), dtype=np.uint32)
    falseDets = np.zeros((7), dtype=np.uint32)
    times = np.zeros((100), dtype=np.float32)
    times_count = np.zeros((100), dtype=np.float32)

    colors_viz = np.random.randint(255, size=(15, 3))

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

        # run network
        start_t = time.time()
        t_error = 0
        t_img = 0
        n_img = 0

        scores, labels, poses, mask = model.predict_on_batch(np.expand_dims(image, axis=0))
        t_img = time.time() - start_t

        scores = scores[labels != -1]
        poses = poses[labels != -1]
        labels = labels[labels != -1]

        print('gt_labels: ', gt_labels)
        print('labels: ', labels)

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
            elif true_cls == 2:
                model_vsd = mv2
            elif true_cls == 3:
                model_vsd = mv3
            elif true_cls == 4:
                model_vsd = mv4
            elif true_cls == 5:
                model_vsd = mv5
            elif true_cls == 6:
                model_vsd = mv6

            add_errors = []
            iou_ovlaps = []

            if true_cls == 1 or true_cls == 2 or true_cls == 4:
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
                trueDets[true_cls] += 1
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

            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst, 2)

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
            image_raw[proj_pts[:, 1], proj_pts[:, 0], :] = colEst

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
        cv2.imwrite(name, image_raw)

    #times
    print('Number of objects ----- t')
    for tfx in range(1, times.shape[0]):
        t_ins = times[tfx] / times_count[tfx]
        print(tfx, '       ------ ', t_ins, tfx)

    recall = np.zeros((7), dtype=np.float32)
    precision = np.zeros((7), dtype=np.float32)
    detections = np.zeros((7), dtype=np.float32)
    det_precision = np.zeros((7), dtype=np.float32)
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

    recall_all = np.sum(recall) / 6.0
    precision_all = np.sum(precision) / 6.0
    detections_all = np.sum(detections) / 6.0
    det_precision_all = np.sum(det_precision) / 6.0
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
