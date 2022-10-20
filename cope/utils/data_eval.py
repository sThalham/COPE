"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import os
import numpy as np
import transforms3d as tf3d
import copy
import cv2
import open3d
from .pose_error import reproj, add, adi, re, te, vsd
import json
import time
import csv


def toPix_array(translation, fx, fy, cx, cy):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


def load_pcd(data_path, cat):
    # load meshes
    ply_path = os.path.join(data_path, cat + '.ply')
    pcd_model = open3d.io.read_point_cloud(ply_path)

    factor = 0.001
    if np.nanmax(pcd_model.points) < 10.0:
        factor = 1.0
    model_vsd = {}
    model_vsd['pts'] = np.asarray(pcd_model.points)
    model_vsd['pts'] = model_vsd['pts'] * factor

    voxel_size = 0.003 / factor

    pcd_down = pcd_model.voxel_down_sample(voxel_size=voxel_size)
    model_down = {}
    model_down['pts'] = np.asarray(pcd_down.points) * factor

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


def evaluate_data(generator, model, dataset_name, data_path, threshold=0.3):

    mesh_info = os.path.join(data_path, "meshes/models_info.json")
    threeD_boxes = np.ndarray((99, 8, 3), dtype=np.float32)
    model_dia = np.zeros((99), dtype=np.float32)
    avg_dimension = np.ndarray((99), dtype=np.float32)

    num_classes = 0
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
        avg_dimension[int(key)] = ((value['size_x'] + value['size_y'] + value['size_z'])/3) * fac
        num_classes += 1

    mesh_path = os.path.join(data_path, "meshes")
    meshes = [None] * (num_classes + 1)
    for mesh_name in os.listdir(mesh_path):
        print(mesh_name)
        if mesh_name.endswith('.ply'):
            if mesh_name[:4] != 'obj_':
                mesh_name = 'obj_' + mesh_name
            pc, mv, md = load_pcd(mesh_path, mesh_name[:-4])
            meshes[int(mesh_name[4:-4])] = mv

    allPoses = np.zeros((num_classes+1), dtype=np.uint32)
    truePoses = np.zeros((num_classes+1), dtype=np.uint32)
    falsePoses = np.zeros((num_classes+1), dtype=np.uint32)
    trueDets = np.zeros((num_classes+1), dtype=np.uint32)
    falseDets = np.zeros((num_classes+1), dtype=np.uint32)
    times = np.zeros((num_classes+1), dtype=np.float32)
    times_count = np.zeros((num_classes+1), dtype=np.float32)

    colors_viz = np.random.randint(255, size=(num_classes, 3))

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

        if gt_labels.size == 0: #or int(gt_labels[0]) in [2, 6]:
            continue

        fx = gt_calib[0, 0]
        fy = gt_calib[0, 1]
        cx = gt_calib[0, 2]
        cy = gt_calib[0, 3]

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

            ori_points = np.ascontiguousarray(threeD_boxes[int(gt_labels[obj]) + 1, :, :], dtype=np.float32)
            tDbox = R_gt.dot(ori_points.T).T
            tDbox = tDbox + np.repeat(t_gt[:, np.newaxis], 8, axis=1).T  # * 0.001
            box3D = toPix_array(tDbox, fx, fy, cx, cy)
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

        print(np.expand_dims(np.array([fx, fy, cx, cy]), axis=0).shape, np.expand_dims(np.array([fx, fy, cx, cy]), axis=0))
        scores, labels, poses, mask, boxes = model.predict_on_batch((np.expand_dims(image, axis=0), np.expand_dims(np.array([fx, fy, cx, cy]), axis=0)))
        t_img = time.time() - start_t

        scores = scores[labels != -1]
        poses = poses[labels != -1]
        boxes = boxes[labels != -1]
        labels = labels[labels != -1]

        for odx, inv_cls in enumerate(labels):

            true_cls = inv_cls + 1
            pose = poses[odx, :]
            box = boxes[odx, :]

            #if inv_cls not in gt_labels:
            #    continue
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

            if inv_cls in gt_labels:

                gt_idx = np.argwhere(gt_labels == inv_cls)
                gt_pose = gt_poses[gt_idx, :]
                gt_box = gt_boxes[gt_idx, :]
                gt_pose = gt_pose[0][0]
                gt_box = gt_box[0][0]

                t_rot = tf3d.quaternions.quat2mat(gt_pose[3:])
                R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                t_gt = np.array(gt_pose[:3], dtype=np.float32)
                t_gt = t_gt * 0.001

                model_vsd = meshes[true_cls - 1]

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

            ori_points = np.ascontiguousarray(threeD_boxes[true_cls, :, :], dtype=np.float32)
            eDbox = R_est.dot(ori_points.T).T
            eDbox = eDbox + np.repeat(t_est[np.newaxis, :], 8, axis=0)  # * 0.001
            est3D = toPix_array(eDbox, fx, fy, cx, cy)
            eDbox = np.reshape(est3D, (16))
            pose = eDbox.astype(np.int16)
            pose = np.where(pose < 3, 3, pose)

            colEst = (50, 205, 50)
            #if err_add > model_dia[true_cls] * 0.1:
            #    colEst = (0, 39, 236)

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

            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (int(box[0] + 0.5 * (box[2] - box[0])), int(box[1] + 0.5 * (box[3] - box[1])))
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2
            image_raw = cv2.putText(image_raw, str(true_cls), org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

            image_box = cv2.putText(image_box, str(true_cls), org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
            est_box = np.array([float(box[0]), float(box[1]), float(box[2]), float(box[3])])

            image_box = cv2.rectangle(image_box, (int(est_box[0]), int(est_box[1])), (int(est_box[2]), int(est_box[3])), (42, 205, 50), 2)


            #colEst = (50, 205, 50)
            #if err_add > model_dia[true_cls] * 0.1:
            #    colEst = (25, 119, 242)

            #pts = model_vsd["pts"]
            #proj_pts = R_est.dot(pts.T).T
            #proj_pts = proj_pts + np.repeat(t_est[np.newaxis, :], pts.shape[0], axis=0)
            #proj_pts = toPix_array(proj_pts, fxkin, fykin, cxkin, cykin)
            #proj_pts = proj_pts.astype(np.uint16)
            #proj_pts[:, 0] = np.where(proj_pts[:, 0] > 639, 0, proj_pts[:, 0])
            #proj_pts[:, 0] = np.where(proj_pts[:, 0] < 0, 0, proj_pts[:, 0])
            #proj_pts[:, 1] = np.where(proj_pts[:, 1] > 479, 0, proj_pts[:, 1])
            #proj_pts[:, 1] = np.where(proj_pts[:, 1] < 0, 0, proj_pts[:, 1])
            #image_raw[proj_pts[:, 1], proj_pts[:, 0], :] = colEst

        #if index > 0:
        #    times[n_img] += t_img
        #    times_count[n_img] += 1

        name = '/home/stefan/PyraPose_viz/' + 'sample_' + str(index) + '.png'
        name_box = '/home/stefan/PyraPose_viz/' + 'box_' + str(index) + '.png'
        #image_row1 = np.concatenate([image_ori, image_raw], axis=1)
        #image_row2 = np.concatenate([image_mask, image_poses], axis=1)
        #image_rows = np.concatenate([image_row1, image_row2], axis=0)
        #cv2.imwrite(name_box, image_box)
        cv2.imwrite(name, image_raw)

        #name = '/home/stefan/PyraPose_viz/' + 'ori_' + str(index) + '.png'
        #cv2.imwrite(name, image_ori)

    #times
    print('Number of objects ----- t')
    for tfx in range(1, times.shape[0]):
        t_ins = times[tfx] / times_count[tfx]
        print(tfx, '       ------ ', t_ins, tfx)

    recall = np.zeros((num_classes+1), dtype=np.float32)
    precision = np.zeros((num_classes+1), dtype=np.float32)
    detections = np.zeros((num_classes+1), dtype=np.float32)
    det_precision = np.zeros((num_classes+1), dtype=np.float32)
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

    recall_all = np.sum(recall) / float(num_classes)
    precision_all = np.sum(precision) / float(num_classes)
    detections_all = np.sum(detections) / float(num_classes)
    det_precision_all = np.sum(det_precision) / float(num_classes)
    print('ALL: ')
    print('mean detection recall: ', detections_all)
    print('mean detection precision: ', det_precision_all)
    print('mean pose recall: ', recall_all)
    print('mean pose precision: ', precision_all)

    wd_path = os.getcwd()
    csv_target = os.path.join(wd_path, 'sthalham-cope-' + str(dataset_name) + '-test.csv')

    line_head = ['scene_id','im_id','obj_id','score','R','t','time']
    with open(csv_target, 'a') as outfile:
        myWriter = csv.writer(outfile, delimiter=',')  # Write out the Headers for the CSV file
        myWriter.writerow(line_head)

    for line_indexed in eval_img:
        with open(csv_target, 'a') as outfile:
            myWriter = csv.writer(outfile, delimiter=',')  # Write out the Headers for the CSV file
            myWriter.writerow(line_indexed)
