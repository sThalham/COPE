
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
import yaml
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


def toPix_array(translation):

    xpix = ((translation[:, 0] * fxkin) / translation[:, 2]) + cxkin
    ypix = ((translation[:, 1] * fykin) / translation[:, 2]) + cykin
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


def load_pcd(data_path, cat):
    # load meshes
    ply_path = os.path.join(data_path, 'meshes', 'obj_' + cat + '.ply')
    pcd_model = open3d.io.read_point_cloud(ply_path)
    model_vsd = {}
    model_vsd['pts'] = np.asarray(pcd_model.points)
    open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
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


def evaluate_occlusion(generator, model, data_path, threshold=0.3):

    mesh_info = os.path.join(data_path, "meshes/models_info.yml")
    threeD_boxes = np.ndarray((31, 8, 3), dtype=np.float32)
    model_dia = np.zeros((31), dtype=np.float32)
    avg_dimension = np.ndarray((16), dtype=np.float32)

    image_locations = locations_for_shape((480, 640))

    for key, value in yaml.load(open(mesh_info)).items():
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

    pc1, mv1 = load_pcd(data_path,'000001')
    pc5, mv5 = load_pcd(data_path,'000005')
    pc6, mv6 = load_pcd(data_path,'000006')
    pc8, mv8 = load_pcd(data_path,'000008')
    pc9, mv9 = load_pcd(data_path,'000009')
    pc10, mv10 = load_pcd(data_path,'000010')
    pc11, mv11 = load_pcd(data_path,'000011')
    pc12, mv12 = load_pcd(data_path,'000012')


    allPoses = np.zeros((16), dtype=np.uint32)
    truePoses = np.zeros((16), dtype=np.uint32)
    falsePoses = np.zeros((16), dtype=np.uint32)
    trueDets = np.zeros((16), dtype=np.uint32)

    for index in progressbar.progressbar(range(generator.size()), prefix='LineMOD evaluation: '):
        image_raw = generator.load_image(index)
        image = generator.preprocess_image(image_raw)
        image, scale = generator.resize_image(image)

        anno = generator.load_annotations(index)

        if len(anno['labels']) < 1:
            continue

        checkLab = anno['labels']  # +1 to real_class
        for idx, lab in enumerate(checkLab):
            allPoses[int(lab) + 1] += 1
            checkLab[idx] += 1

        # run network
        t_start = time.time()
        boxes3D, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print('forward: ', time.time() - t_start)

        boxes3D = boxes3D[labels != -1, :]
        scores = scores[labels != -1]
        labels = labels[labels != -1]

        image = image_raw

        for idx, lab in enumerate(checkLab):
            if lab == 1:
                continue
            t_tra = anno['poses'][idx][:3]
            t_rot = anno['poses'][idx][3:]
            t_rot = tf3d.quaternions.quat2mat(t_rot)
            R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
            t_gt = np.array(t_tra, dtype=np.float32)
            t_gt = t_gt * 0.001

            ori_points = np.ascontiguousarray(threeD_boxes[int(lab), :, :], dtype=np.float32)
            colGT = (255, 0, 0)
            tDbox = R_gt.dot(ori_points.T).T
            tDbox = tDbox + np.repeat(t_gt[:, np.newaxis], 8, axis=1).T
            box3D = toPix_array(tDbox)
            tDbox = np.reshape(box3D, (16))
            tDbox = tDbox.astype(np.uint16)

            image = cv2.line(image, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()), colGT, 2)
            image = cv2.line(image, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()), colGT, 2)
            image = cv2.line(image, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()), colGT,
                             2)
            image = cv2.line(image, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()), colGT,
                             2)
            image = cv2.line(image, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()), colGT,
                             2)
            image = cv2.line(image, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()), colGT,
                             2)
            image = cv2.line(image, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()), colGT,
                             2)
            image = cv2.line(image, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()), colGT,
                             2)
            image = cv2.line(image, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                             colGT, 2)
            image = cv2.line(image, tuple(tDbox[10:12].ravel()), tuple(tDbox[12:14].ravel()),
                             colGT, 2)
            image = cv2.line(image, tuple(tDbox[12:14].ravel()), tuple(tDbox[14:16].ravel()),
                             colGT, 2)
            image = cv2.line(image, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                             colGT, 2)

        for cls in np.unique(labels):

            true_cls = cls + 1
            pose_votes = boxes3D[labels == cls, :]
            scores_votes = scores[labels == cls]
            labels_votes = labels[labels == cls]

            if len(labels_votes) < 1:
                continue
            else:
                trueDets[true_cls] += 1

            if true_cls not in checkLab or true_cls == 2:
                # falsePoses[int(cls)] += 1
                continue

            '''
            print(pose_votes.shape)
            start_anc = time.time()
            min_box_x = np.nanmin(pose_votes[:, ::2], axis=1)
            min_box_y = np.nanmin(pose_votes[:, 1::2], axis=1)
            max_box_x = np.nanmax(pose_votes[:, ::2], axis=1)
            max_box_y = np.nanmax(pose_votes[:, 1::2], axis=1)

            pos_anchors = np.stack([min_box_x, min_box_y, max_box_x, max_box_y], axis=1)
            print(pos_anchors.shape)

            #pos_anchors = anchor_params[cls_indices, :]

            ind_anchors = scores[labels == cls]
            print(ind_anchors.shape)
            #pos_anchors = pos_anchors[0]
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

                print('obj_inds per instance: ', obj_inds)
                per_obj_hyps.append(obj_inds)

            print('mult_hyp: ', time.time() - start_anc)
            '''

            anno_ind = np.argwhere(anno['labels'] == true_cls)
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

            est_points = np.ascontiguousarray(pose_votes, dtype=np.float32).reshape((int(k_hyp * 8), 1, 2))
            obj_points = np.repeat(ori_points[np.newaxis, :, :], k_hyp, axis=0)
            obj_points = obj_points.reshape((int(k_hyp * 8), 1, 3))
            retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                               imagePoints=est_points, cameraMatrix=K,
                                                               distCoeffs=None, rvec=None, tvec=None,
                                                               useExtrinsicGuess=False, iterationsCount=300,
                                                               reprojectionError=5.0, confidence=0.99,
                                                               flags=cv2.SOLVEPNP_ITERATIVE)
            R_est, _ = cv2.Rodrigues(orvec)
            t_est = otvec.T
            if cls == 10 or cls == 11:
                err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
            else:
                err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
            if err_add < model_dia[true_cls] * 0.1:
                truePoses[true_cls] += 1
            print(' ')
            print('error: ', err_add, 'threshold', model_dia[true_cls] * 0.1)

            '''
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
            '''


            '''
            idx = 0
            for i in range(k_hyp):
                image = cv2.circle(image, (est_points[idx, 0, 0], est_points[idx, 0, 1]), 3, (13, 243, 207), -2)
                image = cv2.circle(image, (est_points[idx+1, 0, 0], est_points[idx+1, 0, 1]), 3, (251, 194, 213), -2)
                image = cv2.circle(image, (est_points[idx+2, 0, 0], est_points[idx+2, 0, 1]), 3, (222, 243, 41), -2)
                image = cv2.circle(image, (est_points[idx+3, 0, 0], est_points[idx+3, 0, 1]), 3, (209, 31, 201), -2)
                image = cv2.circle(image, (est_points[idx+4, 0, 0], est_points[idx+4, 0, 1]), 3, (8, 62, 53), -2)                                 
                image = cv2.circle(image, (est_points[idx+5, 0, 0], est_points[idx+5, 0, 1]), 3, (13, 243, 207), -2)
                image = cv2.circle(image, (est_points[idx+6, 0, 0], est_points[idx+6, 0, 1]), 3, (215, 41, 29), -2)
                image = cv2.circle(image, (est_points[idx+7, 0, 0], est_points[idx+7, 0, 1]), 3, (78, 213, 16), -2)
                idx = idx+8
            '''

        #name = '/home/stefan/PyraPose_viz/detection_' + str(index) + '.jpg'
        #cv2.imwrite(name, image)
        #cv2.imwrite('/home/stefan/occ_viz/pred_mask_' + str(index) + '_.jpg', image_mask)
        #print('break')

    recall = np.zeros((16), dtype=np.float32)
    precision = np.zeros((16), dtype=np.float32)
    detections = np.zeros((16), dtype=np.float32)
    for i in range(1, (allPoses.shape[0])):
        recall[i] = truePoses[i] / allPoses[i]
        #precision[i] = truePoses[i] / (truePoses[i] + falsePoses[i])
        detections[i] = trueDets[i] / allPoses[i]
        precision[i] = recall[i]/detections[i]

        if np.isnan(recall[i]):
            recall[i] = 0.0
        if np.isnan(precision[i]):
            precision[i] = 0.0

        print('CLS: ', i)
        print('true detections: ', detections[i])
        print('recall: ', recall[i])
        print('precision: ', precision[i])

    recall_all = np.sum(recall[1:]) / 8.0
    precision_all = np.sum(precision[1:]) / 8.0
    detections_all = np.sum(detections[1:]) / 8.0
    print('ALL: ')
    print('true detections: ', detections_all)
    print('recall: ', recall_all)
    print('precision: ', precision_all)
