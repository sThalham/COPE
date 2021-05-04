
#from pycocotools.cocoeval import COCOeval

import keras
import numpy as np
import json
import pyquaternion
import math
import transforms3d as tf3d
import geometry
import os
import copy
import cv2
import open3d
from ..utils import ply_loader
from .pose_error import reproj, add, adi, re, te, vsd
import yaml

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def get_evaluation(pcd_temp_, pcd_scene_, inlier_thres, tf, final_th=0, n_iter=5):#queue
    tf_pcd =np.eye(4)

    reg_p2p = open3d.registration_icp(pcd_temp_, pcd_scene_ , inlier_thres, np.eye(4),
              open3d.TransformationEstimationPointToPoint(),
              open3d.ICPConvergenceCriteria(max_iteration=1)) #5?
    tf = np.matmul(reg_p2p.transformation, tf)
    tf_pcd = np.matmul(reg_p2p.transformation,tf_pcd)
    pcd_temp_.transform(reg_p2p.transformation)

    for i in range(4):
        inlier_thres = reg_p2p.inlier_rmse*3
        if inlier_thres == 0:
            continue

        reg_p2p = open3d.registration_icp(pcd_temp_,pcd_scene_ , inlier_thres, np.eye(4),
                  open3d.TransformationEstimationPointToPlane(),
                  open3d.ICPConvergenceCriteria(max_iteration=1)) #5?
        tf = np.matmul(reg_p2p.transformation, tf)
        tf_pcd = np.matmul(reg_p2p.transformation, tf_pcd)
        pcd_temp_.transform(reg_p2p.transformation)
    inlier_rmse = reg_p2p.inlier_rmse

    ##Calculate fitness with depth_inlier_th
    if(final_th>0):
        inlier_thres = final_th #depth_inlier_th*2 #reg_p2p.inlier_rmse*3
        reg_p2p = registration_icp(pcd_temp_,pcd_scene_, inlier_thres, np.eye(4),
                  TransformationEstimationPointToPlane(),
                  ICPConvergenceCriteria(max_iteration = 1)) #5?

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


def load_pcd(cat):
    # load meshes
    mesh_path = "/home/sthalham/data/Meshes/tless_BOP/"
    #mesh_path = "/home/stefan/data/val_linemod_cc_rgb/models_ply/"
    ply_path = mesh_path + 'obj_0000' + cat + '.ply'
    model_vsd = ply_loader.load_ply(ply_path)
    pcd_model = open3d.PointCloud()
    pcd_model.points = open3d.Vector3dVector(model_vsd['pts'])
    open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    # open3d.draw_geometries([pcd_model])
    model_vsd_mm = copy.deepcopy(model_vsd)
    model_vsd_mm['pts'] = model_vsd_mm['pts'] * 1000.0
    pcd_model = open3d.read_point_cloud(ply_path)

    return pcd_model, model_vsd, model_vsd_mm


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


def evaluate_tless(generator, model, threshold=0.05):
    threshold = 0.5
    """ Use the pycocotools to evaluate a COCO model on a dataset.
    Args
        generator : The generator for generating the evaluation data.
        model     : The model to evaluate.
        threshold : The score threshold to use.
    """
    # start collecting results
    results = []
    image_ids = []
    image_indices = []
    idx = 0
    Visualization = True

    mesh_info = '/home/sthalham/data/Meshes/tless_BOP/models_info.json'

    threeD_boxes = np.ndarray((31, 8, 3), dtype=np.float32)
    model_dia = np.zeros((31), dtype=np.float32)

    for key, value in yaml.load(open(mesh_info)).items():
        fac = 0.001
        x_minus = value['min_x'] * fac
        y_minus = value['min_y'] * fac
        z_minus = value['min_z'] * fac
        x_plus = value['size_x'] * fac + x_minus
        y_plus = value['size_y'] * fac + y_minus
        z_plus = value['size_z'] * fac + z_minus
        three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                   [x_plus, y_plus, z_minus],
                                   [x_plus, y_minus, z_minus],
                                   [x_plus, y_minus, z_plus],
                                   [x_minus, y_plus, z_plus],
                                   [x_minus, y_plus, z_minus],
                                   [x_minus, y_minus, z_minus],
                                   [x_minus, y_minus, z_plus]])
        threeD_boxes[int(key), :, :] = three_box_solo
        model_dia[int(key)] = value['diameter']

    tp = np.zeros((31), dtype=np.uint32)
    fp = np.zeros((31), dtype=np.uint32)
    fn = np.zeros((31), dtype=np.uint32)

    # interlude
    tp55 = np.zeros((31), dtype=np.uint32)
    fp55 = np.zeros((31), dtype=np.uint32)
    fn55 = np.zeros((31), dtype=np.uint32)

    tp6 = np.zeros((31), dtype=np.uint32)
    fp6 = np.zeros((31), dtype=np.uint32)
    fn6 = np.zeros((31), dtype=np.uint32)

    tp65 = np.zeros((31), dtype=np.uint32)
    fp65 = np.zeros((31), dtype=np.uint32)
    fn65 = np.zeros((31), dtype=np.uint32)

    tp7 = np.zeros((31), dtype=np.uint32)
    fp7 = np.zeros((31), dtype=np.uint32)
    fn7 = np.zeros((31), dtype=np.uint32)

    tp75 = np.zeros((31), dtype=np.uint32)
    fp75 = np.zeros((31), dtype=np.uint32)
    fn75 = np.zeros((31), dtype=np.uint32)

    tp8 = np.zeros((31), dtype=np.uint32)
    fp8 = np.zeros((31), dtype=np.uint32)
    fn8 = np.zeros((31), dtype=np.uint32)

    tp85 = np.zeros((31), dtype=np.uint32)
    fp85 = np.zeros((31), dtype=np.uint32)
    fn85 = np.zeros((31), dtype=np.uint32)

    tp9 = np.zeros((31), dtype=np.uint32)
    fp9 = np.zeros((31), dtype=np.uint32)
    fn9 = np.zeros((31), dtype=np.uint32)

    tp925 = np.zeros((31), dtype=np.uint32)
    fp925 = np.zeros((31), dtype=np.uint32)
    fn925 = np.zeros((31), dtype=np.uint32)

    tp95 = np.zeros((31), dtype=np.uint32)
    fp95 = np.zeros((31), dtype=np.uint32)
    fn95 = np.zeros((31), dtype=np.uint32)

    tp975 = np.zeros((31), dtype=np.uint32)
    fp975 = np.zeros((31), dtype=np.uint32)
    fn975 = np.zeros((31), dtype=np.uint32)
    # interlude end

    tp_add = np.zeros((31), dtype=np.uint32)
    fp_add = np.zeros((31), dtype=np.uint32)
    fn_add = np.zeros((31), dtype=np.uint32)

    rotD = np.zeros((31), dtype=np.uint32)
    less5 = np.zeros((31), dtype=np.uint32)
    rep_e = np.zeros((31), dtype=np.uint32)
    rep_less5 = np.zeros((31), dtype=np.uint32)
    add_e = np.zeros((31), dtype=np.uint32)
    add_less_d = np.zeros((31), dtype=np.uint32)
    vsd_e = np.zeros((31), dtype=np.uint32)
    vsd_less_t = np.zeros((31), dtype=np.uint32)

    add_less_d005 = np.zeros((31), dtype=np.uint32)
    add_less_d015 = np.zeros((31), dtype=np.uint32)
    add_less_d02 = np.zeros((31), dtype=np.uint32)
    add_less_d025 = np.zeros((31), dtype=np.uint32)
    add_less_d03 = np.zeros((31), dtype=np.uint32)
    add_less_d035 = np.zeros((31), dtype=np.uint32)
    add_less_d04 = np.zeros((31), dtype=np.uint32)
    add_less_d045 = np.zeros((31), dtype=np.uint32)
    add_less_d05 = np.zeros((31), dtype=np.uint32)
    add_less_d055 = np.zeros((31), dtype=np.uint32)
    add_less_d06 = np.zeros((31), dtype=np.uint32)
    add_less_d065 = np.zeros((31), dtype=np.uint32)
    add_less_d07 = np.zeros((31), dtype=np.uint32)
    add_less_d075 = np.zeros((31), dtype=np.uint32)
    add_less_d08 = np.zeros((31), dtype=np.uint32)
    add_less_d085 = np.zeros((31), dtype=np.uint32)
    add_less_d09 = np.zeros((31), dtype=np.uint32)
    add_less_d095 = np.zeros((31), dtype=np.uint32)
    add_less_d1 = np.zeros((31), dtype=np.uint32)

    # target annotation
    pc1, mv1, mv1_mm = load_pcd('01')
    pc2, mv2, mv2_mm = load_pcd('02')
    pc3, mv3, mv3_mm = load_pcd('03')
    pc4, mv4, mv4_mm = load_pcd('04')
    pc5, mv5, mv5_mm = load_pcd('05')
    pc6, mv6, mv6_mm = load_pcd('06')
    pc7, mv7, mv7_mm = load_pcd('07')
    pc8, mv8, mv8_mm = load_pcd('08')
    pc9, mv9, mv9_mm = load_pcd('09')
    pc10, mv10, mv10_mm = load_pcd('10')
    pc11, mv11, mv11_mm = load_pcd('11')
    pc12, mv12, mv12_mm = load_pcd('12')
    pc13, mv13, mv13_mm = load_pcd('13')
    pc14, mv14, mv14_mm = load_pcd('14')
    pc15, mv15, mv15_mm = load_pcd('15')
    pc16, mv16, mv16_mm = load_pcd('16')
    pc17, mv17, mv17_mm = load_pcd('17')
    pc18, mv18, mv18_mm = load_pcd('18')
    pc19, mv19, mv19_mm = load_pcd('19')
    pc20, mv20, mv20_mm = load_pcd('20')
    pc21, mv21, mv21_mm = load_pcd('21')
    pc22, mv22, mv22_mm = load_pcd('22')
    pc23, mv23, mv23_mm = load_pcd('23')
    pc24, mv24, mv24_mm = load_pcd('24')
    pc25, mv25, mv25_mm = load_pcd('25')
    pc26, mv26, mv26_mm = load_pcd('26')
    pc27, mv27, mv27_mm = load_pcd('27')
    pc28, mv28, mv28_mm = load_pcd('28')
    pc29, mv29, mv29_mm = load_pcd('29')
    pc30, mv30, mv30_mm = load_pcd('30')

    for index in progressbar.progressbar(range(generator.size()), prefix='Tless evaluation: '):
        image_raw = generator.load_image(index)
        image = generator.preprocess_image(image_raw)
        image, scale = generator.resize_image(image)
        image_dep_path = generator.load_image_dep(index)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        anno = generator.load_annotations(index)

        t_cat = anno['labels'].astype(np.int8) + 1
        obj_name = []
        for idx, obj_temp in enumerate(t_cat):
            if obj_temp < 10:
                obj_name.append('0' + str(obj_temp))
            else:
                obj_name.append(str(obj_temp))
        t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)
        gt_poses = anno['poses']
        gt_calib = anno['K']

        # run network
        boxes, boxes3D, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        # correct boxes for image scale
        boxes /= scale

        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        rotD[t_cat] += 1
        rep_e[t_cat] += 1
        add_e[t_cat] += 1
        vsd_e[t_cat] += 1
        fn[t_cat] += 1
        fn55[t_cat] += 1
        fn6[t_cat] += 1
        fn65[t_cat] += 1
        fn7[t_cat] += 1
        fn75[t_cat] += 1
        fn8[t_cat] += 1
        fn85[t_cat] += 1
        fn9[t_cat] += 1
        fn925[t_cat] += 1
        fn95[t_cat] += 1
        fn975[t_cat] += 1

        # end interlude
        fn_add[t_cat] += 1
        fnit = np.zeros((31), dtype=np.uint32)
        fnit[t_cat] +=1

        vsd_true = False

        # compute predicted labels and scores
        for box, box3D, score, label in zip(boxes[0], boxes3D[0], scores[0], labels[0]):
            # scores are sorted, so we can break
            if score < threshold:
                continue

            if label < 0:
                continue

            cls = generator.label_to_inv_label(label)
            control_points = box3D
            #print(cls)
            #print(control_points)

            # append detection for each positively labeled class
            image_result = {
                'image_id'    : generator.image_ids[index],
                'category_id' : generator.label_to_inv_label(label),
                'score'       : float(score),
                'bbox'        : box.tolist(),
                'pose'        : control_points.tolist()
            }

            # append detection to results
            results.append(image_result)

            if cls in t_cat:
                b1 = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
                odx = np.where(t_cat==cls)

                b2 = np.array([t_bbox[odx[0]][0][0], t_bbox[odx[0]][0][1], t_bbox[odx[0]][0][2], t_bbox[odx[0]][0][3]])

                IoU = boxoverlap(b1, b2)
                # occurences of 2 or more instances not possible in LINEMOD
                if IoU > 0.5:
                    if fnit[cls] > 0:
                        # interlude
                        if IoU > 0.55:
                            tp55[cls] += 1
                            fn55[cls] -= 1
                        else:
                            fp55[cls] += 1
                        if IoU > 0.6:
                            tp6[cls] += 1
                            fn6[cls] -= 1
                        else:
                            fp6[cls] += 1
                        if IoU > 0.65:
                            tp65[cls] += 1
                            fn65[cls] -= 1
                        else:
                            fp65[cls] += 1
                        if IoU > 0.7:
                            tp7[cls] += 1
                            fn7[cls] -= 1
                        else:
                            fp7[cls] += 1
                        if IoU > 0.75:
                            tp75[cls] += 1
                            fn75[cls] -= 1
                        else:
                            fp75[cls] += 1
                        if IoU > 0.8:
                            tp8[cls] += 1
                            fn8[cls] -= 1
                        else:
                            fp8[cls] += 1
                        if IoU > 0.85:
                            tp85[cls] += 1
                            fn85[cls] -= 1
                        else:
                            fp85[cls] += 1
                        if IoU > 0.9:
                            tp9[cls] += 1
                            fn9[cls] -= 1
                        else:
                            fp9[cls] += 1
                        if IoU > 0.925:
                            tp925[cls] += 1
                            fn925[cls] -= 1
                        else:
                            fp925[cls] += 1
                        if IoU > 0.95:
                            tp95[cls] += 1
                            fn95[cls] -= 1
                        else:
                            fp95[cls] += 1
                        if IoU > 0.975:
                            tp975[cls] += 1
                            fn975[cls] -= 1
                        else:
                            fp975[cls] += 1

                        # interlude end

                        tp[cls] += 1
                        fn[cls] -= 1
                        fnit[cls] -= 1

                        obj_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32) #.reshape((8, 1, 3))
                        est_points = np.ascontiguousarray(control_points.T, dtype=np.float32).reshape((8, 1, 2))

                        calib = gt_calib[odx][0]
                        #print(calib)
                        K = np.float32([calib[0], 0., calib[2], 0., calib[1], calib[3], 0., 0., 1.]).reshape(3, 3)

                        #retval, orvec, otvec = cv2.solvePnP(obj_points, est_points, K, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
                        retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                                           imagePoints=est_points, cameraMatrix=K,
                                                                           distCoeffs=None, rvec=None, tvec=None,
                                                                           useExtrinsicGuess=False, iterationsCount=100,
                                                                           reprojectionError=5.0, confidence=0.99,
                                                                           flags=cv2.SOLVEPNP_ITERATIVE)

                        R_est, _ = cv2.Rodrigues(orvec)
                        t_est = otvec

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
