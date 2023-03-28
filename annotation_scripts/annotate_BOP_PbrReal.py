#!/usr/bin/env python

import sys
import os
import subprocess
import cv2
import numpy as np
import json
import math
import datetime
import copy
import transforms3d as tf3d
import time
from pathlib import Path

depSca = 1.0
resX = 640
resY = 480
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899
depthCut = 2000.0


def matang(A, B):

    r_oa_t = np.transpose(A)
    r_ab = np.multiply(r_oa_t, B)

    thetrace = (np.trace(r_ab) -1) / 2
    #if thetrace < 0.0:
    #    thetrace *= -1
    if thetrace < 0.0:
        while thetrace < 1:
            thetrace += 1
    if thetrace > 0.0:
        while thetrace > 1:
            thetrace -= 1
    return np.rad2deg(np.arccos(thetrace))


def get_cont_sympose(rota, sym):

    rot_pose = np.eye((4), dtype=np.float32)
    rot_pose[:3, :3] = tf3d.quaternions.quat2mat(rota[3:])
    rot_pose[:3, 3] = rota[:3]

    cam_in_obj = np.dot(np.linalg.inv(rot_pose), (0, 0, 0, 1))
    if sym[0] == 1:
        alpha = math.atan2(cam_in_obj[2], cam_in_obj[1])
        rot_pose[:3, :3] = np.dot(rot_pose[:3, :3], tf3d.euler.euler2mat(alpha, 0.0, 0.0, 'sxyz'))
    elif sym[1] == 1:
        alpha = math.atan2(cam_in_obj[0], cam_in_obj[2])
        rot_pose[:3, :3] = np.dot(rot_pose[:3, :3], tf3d.euler.euler2mat(0.0, alpha, 0.0, 'sxyz'))
    elif sym[2] == 1:
        alpha = math.atan2(cam_in_obj[1], cam_in_obj[0])
        rot_pose[:3, :3] = np.dot(rot_pose[:3, :3], tf3d.euler.euler2mat(0.0, 0.0, alpha, 'sxyz'))

    rota[3:] = tf3d.quaternions.mat2quat(rot_pose[:3, :3])
    rota[:3] = rot_pose[:3, 3]

    return rota


def get_disc_sympose(rota, sym):

    rot_pose = np.eye((4), dtype=np.float32)
    rot_pose[:3, :3] = tf3d.quaternions.quat2mat(rota[3:])
    rot_pose[:3, 3] = rota[:3]
    #print('rot_pose: ', rot_pose)

    rot_sym = np.dot(rot_pose, sym)
    base_dir = np.dot(sym[:3, :3], (0, 0, 1))
    pose_dir = np.dot(rot_sym[:3, :3], (0, 0, 1))

    ang2z = np.arccos(np.dot(pose_dir, base_dir))

    if ang2z > math.pi * 0.5:
        rot_pose = rot_sym

    rota[3:] = tf3d.quaternions.mat2quat(rot_pose[:3, :3])
    rota[:3] = rot_pose[:3, 3]

    return rota


def draw_axis(img, cam_R, cam_T, K):
    # unit is mm
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)

    rotMat = tf3d.quaternions.quat2mat(cam_R)
    rot, _ = cv2.Rodrigues(rotMat)

    tra = cam_T

    #K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3,3)
    K = np.float32(K).reshape(3,3)

    axisPoints, _ = cv2.projectPoints(points, rot, tra, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img


def toPix(translation):

    xpix = ((translation[0] * fxkin) / translation[2]) + cxkin
    ypix = ((translation[1] * fykin) / translation[2]) + cykin
    #zpix = translation[2] * 0.001 * fxkin

    return [xpix, ypix]


def toPix_array(translation, fx=None, fy=None, cx=None, cy=None):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


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

    return cloud_final


def create_BB(rgb):

    imgray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    mask = imgray > 25

    oneA = np.ones(imgray.shape)
    masked = np.where(mask, oneA, 0)

    kernel = np.ones((9, 9), np.uint8)
    mask_dil = cv2.dilate(masked, kernel, iterations=1)

    im2, contours, hier = cv2.findContours(np.uint8(mask_dil), 1, 2)

    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    area = cv2.contourArea(box)

    # cv2.drawContours(rgb, [box], -1, (170, 160, 0), 2)
    # cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
    bb = [int(x),int(y),int(w),int(h)]

    return cnt, bb, area, mask_dil


if __name__ == "__main__":

    dataset = 'ycbv'
    traintestval = 'train'
    visu = False

    root = "/home/stefan/data/bop_data/ycbv"  # path to train samples, depth + rgb
    target = '/home/stefan/data/train_data/ycbv_train/'

    if dataset == 'linemod':
        mesh_info = '/hdd/bop_datasets/lm/models_eval/models_info.json'
        num_objects = 15
    elif dataset == 'occlusion':
        mesh_info = '/hdd/bop_datasets/lmo/models_eval/models_info.json'
        num_objects = 15
    elif dataset == 'ycbv':
        mesh_info = '/home/stefan/data/bop_data/ycbv/models/models_info.json'
        num_objects = 21
    elif dataset == 'tless':
        mesh_info = '/home/stefan/data/bop_datasets/tless/models_eval/models_info.json'
        num_objects = 30
    elif dataset == 'homebrewed':
        mesh_info = '/home/stefan/data/BOP_datasets/hb/models_eval/models_info.json'
        num_objects = 33
    elif dataset == 'icbin':
        mesh_info = '/hdd/bop_datasets/icbin/models_eval/models_info.json'
        num_objects = 2
    elif dataset == 'canister':
        mesh_info = '/home/stefan/data/datasets/canister/models/models_info.json'
        num_objects = 1
    else:
        print('unknown dataset')

    threeD_boxes = np.ndarray((num_objects + 1, 8, 3), dtype=np.float32)
    sym_cont = np.ndarray((num_objects + 1, 3), dtype=np.float32)
    sym_disc = np.ndarray((num_objects + 1, 4, 4), dtype=np.float32)

    for key, value in json.load(open(mesh_info)).items():
    #for key, value in json.load(open(mesh_info)).items():
        fac = 0.001
        x_minus = value['min_x']
        y_minus = value['min_y']
        z_minus = value['min_z']
        x_plus = value['size_x'] + x_minus
        y_plus = value['size_y'] + y_minus
        z_plus = value['size_z'] + z_minus
        three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                   [x_plus, y_plus, z_minus],
                                   [x_plus, y_minus, z_minus],
                                   [x_plus, y_minus, z_plus],
                                   [x_minus, y_plus, z_plus],
                                   [x_minus, y_plus, z_minus],
                                   [x_minus, y_minus, z_minus],
                                   [x_minus, y_minus, z_plus]])
        threeD_boxes[int(key), :, :] = three_box_solo

        if "symmetries_continuous" in value:
            sym_cont[int(key), :] = np.asarray(value['symmetries_continuous'][0]['axis'], dtype=np.float32)
        elif "symmetries_discrete" in value:
            syms = value['symmetries_discrete']
            # Obj 27 has 3 planes of symmetry
            if len(syms) > 1:
                if dataset == 'ycbv' and int(key) == 16:
                    pass
                elif dataset == 'tless' and int(key) == 27:
                    #sym_disc[int(key), :, :] = np.asarray(syms[0], dtype=np.float32).reshape((4, 4))
                    #sym_disc[31, :, :] = np.asarray(syms[1], dtype=np.float32).reshape((4, 4))
                    #sym_disc[32, :, :] = np.asarray(syms[2], dtype=np.float32).reshape((4, 4))
                    pass
            else:
                sym_disc[int(key), :, :] = np.asarray(syms[0], dtype=np.float32).reshape((4, 4))
        else:
            pass

    now = datetime.datetime.now()
    dateT = str(now)

    dict = {"info": {
                "description": dataset,
                "url": "cmp.felk.cvut.cz/t-less/",
                "version": "1.0",
                "year": 2023,
                "contributor": "Stefan Thalhammer",
                "date_created": dateT
                    },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
            }

    annoID = 0
    count = 0

    parent_root = copy.deepcopy(root)
    domain_dir = os.listdir(parent_root)
    domain_idx = 0
    for domain_path in domain_dir:

        if domain_path!='train_pbr' and domain_path!='train_real':
            continue

        root = os.path.join(parent_root, domain_path)
        syns = os.listdir(root)

        for sub_set in syns:
            set_root = os.path.join(root, sub_set)

            scene_id = str(sub_set)

            print(scene_id)

            if scene_id == '000044' and domain_path == 'train_pbr':
                continue

            rgbPath = set_root + "/rgb/"
            depPath = set_root + "/depth/"
            masPath = set_root + "/mask/"
            visPath = set_root + "/mask_visib/"
            camPath = set_root + "/scene_camera.json"
            gtPath = set_root + "/scene_gt.json"
            infoPath = set_root + "/scene_gt_info.json"
            print(gtPath)

            with open(camPath, 'r') as streamCAM:
                camjson = json.load(streamCAM)

            with open(gtPath, 'r') as streamGT:
                scenejson = json.load(streamGT)

            with open(infoPath, 'r') as streamINFO:
                gtjson = json.load(streamINFO)

            print('managed')

            img_idx=0
            for samp in os.listdir(rgbPath):

                #if img_idx > 1:
                #    continue
                #img_idx += 1

                BOP_im_id = str(samp[:-4])

                imgname = samp
                rgbImgPath = rgbPath + samp
                depImgPath = depPath + samp[:-4] + '.png'
                visImgPath = visPath + samp[:-4] + '.png'

                if samp.startswith('00000'):
                    samp = samp[5:]
                elif samp.startswith('0000'):
                    samp = samp[4:]
                elif samp.startswith('000'):
                    samp = samp[3:]
                elif samp.startswith('00'):
                    samp = samp[2:]
                elif samp.startswith('0'):
                    samp = samp[1:]

                samp = samp[:-4]
                print('samp: ', samp)

                calib = camjson.get(str(samp))
                K = calib["cam_K"]
                depSca = calib["depth_scale"]
                fxca = K[0]
                fyca = K[4]
                cxca = K[2]
                cyca = K[5]

                # GraspA intermezzo
                gtPose = scenejson.get(str(samp))
                gtImg = gtjson.get(str(samp))

                #########################
                # Prepare the stuff
                #########################

                # read images and create mask
                # read images and create mask
                rgbImg = cv2.imread(rgbImgPath)
                depImg = cv2.imread(depImgPath, cv2.IMREAD_UNCHANGED)

                if rgbImg is None or depImg is None:
                    continue
                rows, cols = depImg.shape
                depImg = np.multiply(depImg, depSca)

                # create image number and name
                template_subset = '000000'
                template_samp = '000000'
                imgNum = str(domain_idx) + template_subset[:-len(sub_set)] + sub_set + template_samp[:-len(samp)] + samp
                img_id = int(imgNum)
                imgNam = imgNum + '.png'
                iname = str(imgNam)

                bbox_vis = []
                cat_vis = []
                camR_vis = []
                camT_vis = []
                calib_K = []
                # if rnd == 1:

                if dataset == 'tless':

                    # tless train
                    fxkin = 1075.65091572
                    fykin = 1073.90347929
                    cxkin = 360
                    cykin = 240

                    shift_x = (fxca / fxkin) * 320
                    shift_y = (fyca / fykin) * 240

                    sha_y, sha_x, _ = rgbImg.shape
                    pad_img = np.zeros((sha_y * 2, sha_x * 2, 3), dtype=np.uint8)
                    pad_img[int(sha_y*0.5):-int(sha_y*0.5), int(sha_x*0.5):-int(sha_x*0.5), :] = rgbImg
                    rgbImg = pad_img[int((sha_y*0.5)+cyca-shift_y):int((sha_y*0.5)+cyca+shift_y), int((sha_x*0.5)+cxca-shift_x):int((sha_x*0.5)+cxca+shift_x), :]

                    #rgbImg = rgbImg[int(sha_y*0.5):-int(sha_y*0.5), int(sha_x*0.5):-int(sha_x*0.5), :]

                    fxca = fxkin
                    fyca = fykin
                    cxvan = cxca
                    cyvan = cyca
                    cxca = 320.0
                    cyca = 240.0
                    rgbImg = cv2.resize(rgbImg, (640, 480))

                fileName = target + 'images/' + traintestval + '/' + imgNam[:-4] + '_rgb.png'
                myFile = Path(fileName)
                if myFile.exists():
                    print('File exists, skip encoding, ', fileName)
                else:
                    imgI = depImg.astype(np.uint16)

                    rgb_name = fileName[:-8] + '_rgb.png'
                    cv2.imwrite(fileName, rgbImg)
                    print("storing image in : ", fileName)

                mask_ind = 0
                #if dataset == 'tless':
                #    mask_img = np.zeros((540, 720), dtype=np.uint8)
               # else:
                mask_img = np.zeros((480, 640), dtype=np.uint8)
                bbvis = []
                bbox_vis = []
                cnt = 0
                # bbsca = 720.0 / 640.0
                for i in range(len(gtImg)):
                    mask_name = '000000'[:-len(samp)] + samp + '_000000'[:-len(str(mask_ind))] + str(mask_ind) + '.png'
                    mask_path = os.path.join(visPath, mask_name)
                    obj_mask = cv2.imread(mask_path)
                    if obj_mask is None:
                        continue
                    obj_mask = obj_mask[:, :, 0]
                    if dataset == 'tless':

                        pad_img = np.zeros((sha_y * 2, sha_x * 2), dtype=np.uint8)
                        pad_img[int(sha_y * 0.5):-int(sha_y * 0.5), int(sha_x * 0.5):-int(sha_x * 0.5)] = obj_mask
                        obj_mask = pad_img[int((sha_y * 0.5) + cyvan - shift_y):int((sha_y * 0.5) + cyvan + shift_y),
                                   int((sha_x * 0.5) + cxvan - shift_x):int((sha_x * 0.5) + cxvan + shift_x)]

                        obj_mask = cv2.resize(obj_mask, (640, 480))

                    mask_id = mask_ind + 1
                    mask_img = np.where(obj_mask > 0, mask_id, mask_img)
                    mask_ind = mask_ind + 1

                    curlist = gtImg[i]
                    #obj_bb = curlist["bbox_visib"]
                    obj_bb = curlist["bbox_obj"]

                    obj_id = gtPose[i]['obj_id']
                    if dataset == 'canister':
                        obj_id += 1

                    if dataset == 'linemod':
                        if obj_id == 7 or obj_id == 3:
                            continue

                    R = gtPose[i]["cam_R_m2c"]
                    T = gtPose[i]["cam_t_m2c"]
                    cat_vis.append(obj_id)

                    # pose [x, y, z, roll, pitch, yaw]
                    R = np.asarray(R, dtype=np.float32)
                    rot = tf3d.quaternions.mat2quat(R.reshape(3, 3))
                    rot = np.asarray(rot, dtype=np.float32)
                    tra = np.asarray(T, dtype=np.float32)

                    if dataset == 'canister':
                        tra *= 1000.0
                        offset = np.array([-1.396, 2.799, 54.302])
                        offset = R.reshape(3, 3).dot(offset.T)
                        tra = tra + offset

                    # interlude for rotating d435
                    if scene_id == '000001' and dataset == 'canister':
                        trans = np.eye(4)
                        trans[:3, :3] = R.reshape(3, 3)
                        trans[:3, 3] = tra
                        mod_ori = np.eye(4)
                        mod_ori[:3, :3] = tf3d.euler.euler2mat(math.pi, 0.0, 0.0, 'szyx')
                        trans = mod_ori @ trans
                        tra = trans[:3, 3]
                        # tra[:2] *= -1
                        rot = tf3d.quaternions.mat2quat(trans[:3, :3])

                    pose = [tra[0], tra[1], tra[2], rot[0], rot[1], rot[2], rot[3]]

                    visib_fract = float(curlist["visib_fract"])
                    area = obj_bb[2] * obj_bb[3]

                    trans = np.asarray([pose[0], pose[1], pose[2]], dtype=np.float32)
                    R = tf3d.quaternions.quat2mat(np.asarray([pose[3], pose[4], pose[5], pose[6]], dtype=np.float32))
                    tDbox = R.reshape(3, 3).dot(threeD_boxes[obj_id, :, :].T).T
                    tDbox = tDbox + np.repeat(trans[np.newaxis, :], 8, axis=0)
                    box3D = toPix_array(tDbox, fx=fxca, fy=fyca, cx=cxca, cy=cyca)
                    box3D = np.reshape(box3D, (16))
                    box3D = box3D.tolist()

                    pose = [np.asscalar(pose[0]), np.asscalar(pose[1]), np.asscalar(pose[2]),
                            np.asscalar(pose[3]), np.asscalar(pose[4]), np.asscalar(pose[5]), np.asscalar(pose[6])]

                    #if obj_id in [10, 11, 14]:
                    bbox_vis.append(obj_bb)
                    bbvis.append(box3D)
                    camR_vis.append(np.asarray([pose[3], pose[4], pose[5], pose[6]], dtype=np.float32))
                    camT_vis.append(np.asarray([pose[0], pose[1], pose[2]], dtype=np.float32))
                    calib_K.append(K)

                    nx1 = obj_bb[0]
                    ny1 = obj_bb[1]
                    nx2 = nx1 + obj_bb[2]
                    ny2 = ny1 + obj_bb[3]
                    npseg = np.array([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2])
                    cont = npseg.tolist()

                    annoID = annoID + 1
                    tempTA = {
                        "scene_id": scene_id,
                        "im_id": BOP_im_id,
                        "id": annoID,
                        "image_id": img_id,
                        "category_id": obj_id,
                        "bbox": obj_bb,
                        "pose": pose,
                        "segmentation": box3D,
                        "mask_id": mask_id,
                        "area": area,
                        "iscrowd": 0,
                        "feature_visibility": visib_fract
                    }

                    dict["annotations"].append(tempTA)
                    count = count + 1

                tempTL = {
                    "url": "https://bop.felk.cvut.cz/home/",
                    "id": img_id,
                    "name": iname,
                }
                dict["licenses"].append(tempTL)

                if dataset == 'canister':
                    #mask_img = cv2.warpAffine(mask_img, warp_mat[:2, :], dsize=(rgbImg.shape[1], rgbImg.shape[0]),
                    #                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE, borderValue=0)

                    if scene_id == '000001':
                        mask_img = np.flip(mask_img, axis=0)
                        mask_img = np.flip(mask_img, axis=1)

                # mask_img = cv2.resize(mask_img, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_NEAREST)
                mask_safe_path = fileName[:-8] + '_mask.png'
                cv2.imwrite(mask_safe_path, mask_img)

                tempTV = {
                    "license": 2,
                    "url": "https://bop.felk.cvut.cz/home/",
                    "file_name": iname,
                    "height": resY,
                    "width": resX,
                    "fx": fxca,
                    "fy": fyca,
                    "cx": cxca,
                    "cy": cyca,
                    "date_captured": dateT,
                    "id": img_id,
                }
                dict["images"].append(tempTV)

                if visu is True:
                    img = rgbImg
                    for i, bb in enumerate(bbvis):

                        bb = np.array(bb)
                        bbox = np.array(bbox_vis[i])

                        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                                                  (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (42, 205, 50), 2)

                        cv2.imwrite(mask_safe_path, (mask_img*10).astype(np.uint8))

                        phler = True
                        if phler:
                            pose = np.asarray(bbvis[i], dtype=np.int16)

                            colR = 250
                            colG = 25
                            colB = 175

                            img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), (130, 245, 13), 2)
                            img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), (50, 112, 220), 2)
                            img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), (50, 112, 220), 2)
                            img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), (50, 112, 220), 2)
                            img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), (colR, colG, colB),
                                           2)
                            img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()),
                                           (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()),
                                           (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()),
                                           (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()),
                                           (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()),
                                           (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()),
                                           (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()),
                                           (colR, colG, colB), 2)

                    #print(camR_vis[i], camT_vis[i])
                    #draw_axis(img, camR_vis[i], camT_vis[i], K)
                    cv2.imwrite(rgb_name, img)

                    print('STOP')

        domain_idx += 1

    for s in range(1, num_objects + 1):
        objName = str(s)
        tempC = {
            "id": s,
            "name": objName,
            "supercategory": "object"
        }
        dict["categories"].append(tempC)

    valAnno = target + 'annotations/instances_' + traintestval + '.json'

    with open(valAnno, 'w') as fpT:
        json.dump(dict, fpT)

    print('everythings done')


