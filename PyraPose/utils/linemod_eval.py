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

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


# LineMOD
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899


def get_evaluation_kiru(pcd_temp_,pcd_scene_,inlier_thres,tf,final_th, model_dia):#queue
    tf_pcd =np.eye(4)
    pcd_temp_.transform(tf)

    mean_temp = np.mean(np.array(pcd_temp_.points)[:, 2])
    mean_scene = np.median(np.array(pcd_scene_.points)[:, 2])
    pcd_diff = mean_scene - mean_temp

    #open3d.draw_geometries([pcd_temp_])
    # align model with median depth of scene
    new_pcd_trans = []
    for i, point in enumerate(pcd_temp_.points):
        poi = np.asarray(point)
        poi = poi + [0.0, 0.0, pcd_diff]
        new_pcd_trans.append(poi)
    tf = np.array(tf)
    tf[2, 3] = tf[2, 3] + pcd_diff
    pcd_temp_.points = open3d.Vector3dVector(np.asarray(new_pcd_trans))
    open3d.estimate_normals(pcd_temp_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=5.0, max_nn=10))

    pcd_min = mean_scene - (model_dia * 2)
    pcd_max = mean_scene + (model_dia * 2)
    new_pcd_scene = []
    for i, point in enumerate(pcd_scene_.points):
        if point[2] > pcd_min or point[2] < pcd_max:
            new_pcd_scene.append(point)
    pcd_scene_.points = open3d.Vector3dVector(np.asarray(new_pcd_scene))
    #open3d.draw_geometries([pcd_scene_])
    open3d.estimate_normals(pcd_scene_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=5.0, max_nn=10))

    reg_p2p = open3d.registration.registration_icp(pcd_temp_,pcd_scene_ , inlier_thres, np.eye(4),
                                                   open3d.registration.TransformationEstimationPointToPoint(),
                                                   open3d.registration.ICPConvergenceCriteria(max_iteration = 5)) #5?
    tf = np.matmul(reg_p2p.transformation,tf)
    tf_pcd = np.matmul(reg_p2p.transformation,tf_pcd)
    pcd_temp_.transform(reg_p2p.transformation)

    open3d.estimate_normals(pcd_temp_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=2.0, max_nn=30))
    #open3d.draw_geometries([pcd_scene_])
    points_unfiltered = np.asarray(pcd_temp_.points)
    last_pcd_temp = []
    for i, normal in enumerate(pcd_temp_.normals):
        if normal[2] < 0:
            last_pcd_temp.append(points_unfiltered[i, :])
    if not last_pcd_temp:
        normal_array = np.asarray(pcd_temp_.normals) * -1
        pcd_temp_.normals = open3d.Vector3dVector(normal_array)
        points_unfiltered = np.asarray(pcd_temp_.points)
        last_pcd_temp = []
        for i, normal in enumerate(pcd_temp_.normals):
            if normal[2] < 0:
                last_pcd_temp.append(points_unfiltered[i, :])
    #print(np.asarray(last_pcd_temp))
    pcd_temp_.points = open3d.Vector3dVector(np.asarray(last_pcd_temp))

    open3d.estimate_normals(pcd_temp_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=5.0, max_nn=30))

    hyper_tresh = inlier_thres
    for i in range(4):
        inlier_thres = reg_p2p.inlier_rmse*2
        hyper_thres = hyper_tresh * 0.75
        if inlier_thres < 1.0:
            inlier_thres = hyper_tresh * 0.75
            hyper_tresh = inlier_thres
        reg_p2p = open3d.registration.registration_icp(pcd_temp_,pcd_scene_ , inlier_thres, np.eye(4),
                                                       open3d.registration.TransformationEstimationPointToPlane(),
                                                       open3d.registration.ICPConvergenceCriteria(max_iteration = 1)) #5?
        tf = np.matmul(reg_p2p.transformation,tf)
        tf_pcd = np.matmul(reg_p2p.transformation,tf_pcd)
        pcd_temp_.transform(reg_p2p.transformation)
    inlier_rmse = reg_p2p.inlier_rmse

    #open3d.draw_geometries([pcd_temp_, pcd_scene_])

    ##Calculate fitness with depth_inlier_th
    if(final_th>0):

        inlier_thres = final_th #depth_inlier_th*2 #reg_p2p.inlier_rmse*3
        reg_p2p = open3d.registration.registration_icp(pcd_temp_,pcd_scene_, inlier_thres, np.eye(4),
                                                       open3d.registration.TransformationEstimationPointToPlane(),
                                                       open3d.registration.ICPConvergenceCriteria(max_iteration = 1)) #5?
        tf = np.matmul(reg_p2p.transformation, tf)
        tf_pcd = np.matmul(reg_p2p.transformation, tf_pcd)
        pcd_temp_.transform(reg_p2p.transformation)

    #open3d.draw_geometries([last_pcd_temp_, pcd_scene_])

    if( np.abs(np.linalg.det(tf[:3,:3])-1)>0.001):
        tf[:3,0]=tf[:3,0]/np.linalg.norm(tf[:3,0])
        tf[:3,1]=tf[:3,1]/np.linalg.norm(tf[:3,1])
        tf[:3,2]=tf[:3,2]/np.linalg.norm(tf[:3,2])
    if( np.linalg.det(tf) < 0) :
        tf[:3,2]=-tf[:3,2]

    return tf,inlier_rmse,tf_pcd,reg_p2p.fitness


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


def denorm_box(locations, regression, obj_diameter):
    mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #std = [150, 150,  150,  150,  150,  150,  150,  150,  150,  150,  150, 150, 150, 150, 150, 150]
    std = np.full(16, 0.65)
    #std = np.full(18, 0.95)

    #regression = np.where(regression > 0, np.log(regression + 1.0), regression)
    #regression = np.where(regression < 0, -np.log(-regression + 1.0), regression)

    obj_diameter = obj_diameter * 1000.0

    x1 = locations[:, :, 0] - (regression[:, :, 0] * (std[0] * obj_diameter) + mean[0])
    y1 = locations[:, :, 1] - (regression[:, :, 1] * (std[1] * obj_diameter) + mean[1])
    x2 = locations[:, :, 0] - (regression[:, :, 2] * (std[2] * obj_diameter) + mean[2])
    y2 = locations[:, :, 1] - (regression[:, :, 3] * (std[3] * obj_diameter) + mean[3])
    x3 = locations[:, :, 0] - (regression[:, :, 4] * (std[4] * obj_diameter) + mean[4])
    y3 = locations[:, :, 1] - (regression[:, :, 5] * (std[5] * obj_diameter) + mean[5])
    x4 = locations[:, :, 0] - (regression[:, :, 6] * (std[6] * obj_diameter) + mean[6])
    y4 = locations[:, :, 1] - (regression[:, :, 7] * (std[7] * obj_diameter) + mean[7])
    x5 = locations[:, :, 0] - (regression[:, :, 8] * (std[8] * obj_diameter) + mean[8])
    y5 = locations[:, :, 1] - (regression[:, :, 9] * (std[9] * obj_diameter) + mean[9])
    x6 = locations[:, :, 0] - (regression[:, :, 10] * (std[10] * obj_diameter) + mean[10])
    y6 = locations[:, :, 1] - (regression[:, :, 11] * (std[11] * obj_diameter) + mean[11])
    x7 = locations[:, :, 0] - (regression[:, :, 12] * (std[12] * obj_diameter) + mean[12])
    y7 = locations[:, :, 1] - (regression[:, :, 13] * (std[13] * obj_diameter) + mean[13])
    x8 = locations[:, :, 0] - (regression[:, :, 14] * (std[14] * obj_diameter) + mean[14])
    y8 = locations[:, :, 1] - (regression[:, :, 15] * (std[15] * obj_diameter) + mean[15])
    #x9 = locations[:, :, 0] - (regression[:, :, 16] * (std[16] * obj_diameter) + mean[0])
    #y9 = locations[:, :, 1] - (regression[:, :, 17] * (std[17] * obj_diameter) + mean[1])

    pred_boxes = np.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8], axis=2)
    #pred_boxes = np.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9], axis=2)

    return pred_boxes


def evaluate_linemod(generator, model, data_path, threshold=0.3):

    mesh_info = os.path.join(data_path, "meshes/models_info.json")
    threeD_boxes = np.ndarray((31, 8, 3), dtype=np.float32)
    model_dia = np.zeros((31), dtype=np.float32)
    avg_dimension = np.ndarray((16), dtype=np.float32)

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

    pc1, mv1 = load_pcd(data_path,'000001')
    pc2, mv2 = load_pcd(data_path,'000002')
    pc4, mv4 = load_pcd(data_path,'000004')
    pc5, mv5 = load_pcd(data_path,'000005')
    pc6, mv6 = load_pcd(data_path,'000006')
    pc8, mv8 = load_pcd(data_path,'000008')
    pc9, mv9 = load_pcd(data_path,'000009')
    pc10, mv10 = load_pcd(data_path,'000010')
    pc11, mv11 = load_pcd(data_path,'000011')
    pc12, mv12 = load_pcd(data_path,'000012')
    pc13, mv13 = load_pcd(data_path,'000013')
    pc14, mv14 = load_pcd(data_path,'000014')
    pc15, mv15 = load_pcd(data_path,'000015')


    allPoses = np.zeros((16), dtype=np.uint32)
    truePoses = np.zeros((16), dtype=np.uint32)
    falsePoses = np.zeros((16), dtype=np.uint32)
    trueDets = np.zeros((16), dtype=np.uint32)
    falseDets = np.zeros((16), dtype=np.uint32)
    times = np.zeros((30), dtype=np.float32)
    times_count = np.zeros((30), dtype=np.float32)

    eval_img = []
    for index, sample in enumerate(generator):

        scene_id = sample[0].numpy()
        image_id = sample[1].numpy()
        image = sample[2]
        gt_labels = sample[3].numpy()
        gt_boxes = sample[4].numpy()
        gt_poses = sample[5].numpy()
        gt_calib = sample[6].numpy()

        if gt_labels.size == 0 or int(gt_labels[0]) in [2, 6]:
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

            ori_points = np.ascontiguousarray(threeD_boxes[int(gt_labels[obj])+1, :, :], dtype=np.float32)

            tDbox = R_gt.dot(ori_points.T).T
            tDbox = tDbox + np.repeat(t_gt[:, np.newaxis], 8, axis=1).T  # * 0.001
            box3D = toPix_array(tDbox, fxkin, fykin, cxkin, cykin)
            tDbox = np.reshape(box3D, (16))
            tDbox = tDbox.astype(np.uint16)

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
        #scores, labels, poses, mask = model.predict_on_batch(np.expand_dims(image, axis=0))
        #t_img = time.time() - start_t

        #scores = scores[labels != -1]
        #poses = poses[labels != -1]
        #labels = labels[labels != -1]

        #scores = scores[labels != -1]
        #poses = poses[labels != -1]
        #labels = labels[labels != -1]

        #for odx, inv_cls in enumerate(labels):

            #true_cls = inv_cls + 1
            # pose = poses[odx, :]
            # if inv_cls not in gt_labels:
            #    continue
            #n_img += 1

            #if inv_cls != int(gt_labels[0]) or true_cls in [3, 7]:
            #    continue

            #R_est = np.array(pose[:9]).reshape((3, 3)).T
            #t_est = np.array(pose[-3:]) * 0.001

        boxes3D, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        t_img = time.time() - start_t

        labels_cls = np.argmax(labels[0, :, :], axis=1)

        for inv_cls in np.unique(labels_cls):

            true_cls = inv_cls + 1
            n_img += 1

            print('inv_cls: ', inv_cls)
            print('gt: ', gt_labels)

            if inv_cls != int(gt_labels[0]) or true_cls in [3, 7]:
                continue

            cls_indices = np.where(labels_cls == inv_cls)
            labels_filt = labels[0, cls_indices[0], inv_cls]
            pose_votes = boxes3D[0, cls_indices[0], inv_cls, :]
            above_thres = np.where(labels_filt > 0.25)

            pose_votes = pose_votes[above_thres[0], :]

            hyps = pose_votes.shape[0]
            if hyps < 1:
                continue

            ori_points = np.ascontiguousarray(threeD_boxes[true_cls, :, :], dtype=np.float32) # .reshape((8, 1, 3))
            obj_points = np.repeat(ori_points[np.newaxis, :, :], hyps, axis=0)
            obj_points = obj_points.reshape((int(hyps * 8), 1, 3))
            K = np.float64([float(fxkin), 0., float(cxkin), 0., float(fykin), float(cykin), 0., 0., 1.]).reshape(3, 3)

            est_points = np.ascontiguousarray(pose_votes, dtype=np.float32).reshape((hyps * 8, 2))
            #est_points = pose_votes.reshape((hyps * 8, 1, 2))

            retval, orvec, otvec, _ = cv2.solvePnPRansac(objectPoints=obj_points,
                                                               imagePoints=est_points, cameraMatrix=K,
                                                               distCoeffs=None, rvec=None, tvec=None,
                                                               useExtrinsicGuess=False, iterationsCount=300,
                                                               reprojectionError=5.0, confidence=0.99,
                                                               flags=cv2.SOLVEPNP_EPNP)
            R_est, _ = cv2.Rodrigues(orvec)
            t_est = otvec[:, 0]

            '''
            eval_line = []
            sc_id = int(scene_id)
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
            '''

            #gt_idx = np.argwhere(gt_labels == inv_cls)
            #gt_pose = gt_poses[gt_idx, :]
            #gt_box = gt_boxes[gt_idx, :]
            #gt_pose = gt_pose[0][0]
            #gt_box = gt_box[0][0]


            # detection
            #min_x = int(np.nanmin(pose[::2], axis=0))
            #min_y = int(np.nanmin(pose[1::2], axis=0))
            #max_x = int(np.nanmax(pose[::2], axis=0))
            #max_y = int(np.nanmax(pose[1::2], axis=0))
            #est_box = np.array([float(min_x), float(min_y), float(max_x), float(max_y)])

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
            for gtdx in range(gt_poses.shape[0]):
                t_rot = tf3d.quaternions.quat2mat(gt_poses[gtdx, 3:])
                R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                t_gt = np.array(gt_poses[gtdx, :3], dtype=np.float32)
                t_gt = t_gt * 0.001

                print('t_gt: ', t_gt)

                if true_cls == 10 or true_cls == 11:
                    err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                else:
                    err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                add_errors.append(err_add)

                #iou = boxoverlap(est_box, gt_boxes[gtdx, :])
                #iou_ovlaps.append(iou)

            idx_add = np.argmin(np.array(add_errors))
            err_add = add_errors[idx_add]
            gt_pose = gt_poses[idx_add, :]

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

            #idx_iou = np.argmax(np.array(iou_ovlaps))
            #iou_ov = iou_ovlaps[idx_iou]

            #if iou_ov > 0.7 and np.max(gt_boxes[idx_iou, :]) != -1:
            #    trueDets[true_cls] += 1
            #    gt_boxes[idx_add, :] = -1
            #else:
            #    falseDets[true_cls] += 1

            eDbox = R_est.dot(ori_points.T).T
            eDbox = eDbox + np.repeat(t_est[np.newaxis, :], 8, axis=0) #* 0.001
            est3D = toPix_array(eDbox, fxkin, fykin, cxkin, cykin)
            eDbox = np.reshape(est3D, (16))
            pose = eDbox.astype(np.uint16)
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

        #if index > 0:
        #    times[n_img] += t_img
        #    times_count[n_img] += 1

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
    recall_all = np.sum(recall) / 13.0
    precision_all = np.sum(precision) / 13.0
    detections_all = np.sum(detections) / 13.0
    det_precision_all = np.sum(det_precision) / 13.0
    print('ALL: ')
    print('mean detection recall: ', detections_all)
    print('mean detection precision: ', det_precision_all)
    print('mean pose recall: ', recall_all)
    print('mean pose precision: ', precision_all)

    wd_path = os.getcwd()
    csv_target = os.path.join(wd_path, 'sthalham-cope-lm-test.csv')

    line_head = ['scene_id','im_id','obj_id','score','R','t','time']
    with open(csv_target, 'a') as outfile:
        myWriter = csv.writer(outfile, delimiter=',')  # Write out the Headers for the CSV file
        myWriter.writerow(line_head)

    for line_indexed in eval_img:
        with open(csv_target, 'a') as outfile:
            myWriter = csv.writer(outfile, delimiter=',')  # Write out the Headers for the CSV file
            myWriter.writerow(line_indexed)