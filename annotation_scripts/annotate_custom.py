import os
import sys
import yaml
import cv2
import numpy as np
import datetime
import copy
import transforms3d as tf3d
import time
import random
import json

from pathlib import Path

# Import bop_renderer and bop_toolkit.
# ------------------------------------------------------------------------------
#bop_renderer_path = '/home/stefan/bop_renderer/build'
#sys.path.append(bop_renderer_path)

#import bop_renderer


def lookAt(eye, target, up):
    # eye is from
    # target is to
    # expects numpy arrays
    f = eye - target
    f = f/np.linalg.norm(f)

    s = np.cross(up, f)
    s = s/np.linalg.norm(s)
    u = np.cross(f, s)
    u = u/np.linalg.norm(u)

    tx = np.dot(s, eye.T)
    ty = np.dot(u, eye.T)
    tz = np.dot(f, eye.T)

    m = np.zeros((4, 4), dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = f
    m[:, 3] = [tx, ty, tz, 1]

    #m[0, :-1] = s
    #m[1, :-1] = u
    #m[2, :-1] = -f
    #m[-1, -1] = 1.0

    return m

def m3dLookAt(eye, target, up):
    mz = normalize(eye - target) # inverse line of sight
    mx = normalize( cross( up, mz ) )
    my = normalize( cross( mz, mx ) )
    tx =  dot( mx, eye )
    ty =  dot( my, eye )
    tz = -dot( mz, eye )
    return np.array([mx[0], my[0], mz[0], 0, mx[1], my[1], mz[1], 0, mx[2], my[2], mz[2], 0, tx, ty, tz, 1])


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


def toPix_array(translation, fx=None, fy=None, cx=None, cy=None):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


if __name__ == "__main__":

    dataset = 'canister'
    base_path = '/home/stefan/data/canister_pix2pose'
    root = os.path.join(base_path, dataset, 'train')
    background = '/home/stefan/data/EvalMESH/COCO/val2017'
    target = '/home/stefan/data/EvalMESH/renderings/CAD/ROST'

    if dataset == 'ROST':
        mesh_path = os.path.join(base_path, 'Meshes', 'CAD', dataset)
    elif dataset == 'canister':
        mesh_path = os.path.join(base_path, 'models')

    visu = False

    resX = 2208
    resY = 1242
    fx = 1359.9708251953125
    fy = 1359.9708251953125
    cx = 1072.132568359375
    cy = 601.8897705078125
    #a_x = 57°
    #a_y = 43°
    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

    #ren = bop_renderer.Renderer()
    #ren.init(resX, resY)
    #mesh_id = 1
    #categories = []

    #for mesh_now in os.listdir(mesh_path):
    #    mesh_path_now = os.path.join(mesh_path, mesh_now)
    #    if mesh_now[-4:] != '.ply':
    #        continue
    #    mesh_id = int(mesh_now[:-4])
    #    ren.add_object(mesh_id, mesh_path_now)
    #    categories.append(mesh_id)

    # interlude for debugging
    mesh_info = os.path.join(mesh_path, 'models_info.json')
    threeD_boxes = np.ndarray((2, 8, 3), dtype=np.float32)
    sym_cont = np.ndarray((2, 3), dtype=np.float32)
    sym_disc = np.ndarray((2, 4, 4), dtype=np.float32)

    max_box = [0, 0, 0, 0]
    max_box_area = 0
    min_box = [0, 0, 0, 0]
    min_box_area = 300 * 300

    #for key, value in yaml.load(open(mesh_info)).items():
    for key, value in json.load(open(mesh_info)).items():
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

        threeD_boxes[int(key), :, :] = three_box_solo * fac

    if "symmetries_continuous" in value:
        sym_cont[int(key), :] = np.asarray(value['symmetries_continuous'][0]['axis'], dtype=np.float32)
    elif "symmetries_discrete" in value:
        syms = value['symmetries_discrete']
        sym_disc[int(key), :, :] = np.asarray(syms[0], dtype=np.float32).reshape((4, 4))
    else:
        pass

    now = datetime.datetime.now()
    dateT = str(now)

    dict = {"info": {
        "description": dataset,
        "url": "no_assigned",
        "version": "1.0",
        "year": 2022,
        "contributor": "Stefan Thalhammer",
        "date_created": dateT
    },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    dictVal = copy.deepcopy(dict)

    annoID = 0
    gloCo = 1
    times = 0
    loops = 2

    syns = os.listdir(background)
    sub = os.listdir(root)

    for idx, set in enumerate(sub):
        set_root = os.path.join(root, set)

        # read camera poses
        cam_poses = os.path.join(root, set, 'groundtruth_handeye.txt')
        with open(cam_poses) as file:
            cameras = [line.rstrip().split() for line in file]
            for ldx, line in enumerate(cameras):
                cameras[ldx] = [float(i) for i in line]

        anno = os.path.join(root, set, 'poses.yaml')

        obj_ids = []
        obj_poses = []
        for item in yaml.load(open(anno)):
            obj_ids.append(item['id'])
            obj_poses.append(item['pose'])

        for sdx, pose in enumerate(cameras):

            if sdx != 67:
                continue
            cam_pose = cameras[sdx]
            T_cam = np.eye(4)
            T_cam[:3, 3] = cam_pose[1:4]
            T_cam[:3, :3] = tf3d.quaternions.quat2mat((cam_pose[7], cam_pose[4], cam_pose[5], cam_pose[6]))

            rand_bg = np.random.choice(syns)
            bg_img_path_j = os.path.join(background, rand_bg)
            bg_img = cv2.imread(bg_img_path_j)
            bg_x, bg_y, _ = bg_img.shape

            if bg_y > bg_x:
                bg_img = np.swapaxes(bg_img, 0, 1)

            bg_img = cv2.resize(bg_img, (resX, resY))

            template_samp = '0000'
            imgNum = set + template_samp[:-len(str(sdx))] + str(sdx)
            img_id = int(imgNum)
            imgNam = imgNum + '.png'
            iname = str(imgNam)

            fileName = os.path.join(target, 'images/train', imgNam[:-4] + '.png')
            myFile = Path(fileName)

            cnt = 0
            mask_ind = 0
            mask_img = np.zeros((resY, resX), dtype=np.uint8)
            visib_img = np.zeros((resY, resX, 3), dtype=np.uint8)

            boxes3D = []
            calib_K = []
            zeds = []
            renderings = []
            visibilities = []
            bboxes = []
            full_visib = []
            areas = []
            mask_idxs = []
            poses = []

            rotations = []
            translations = []

            for odx, objID in enumerate(obj_ids):
                pose = np.array(obj_poses[odx]).reshape(4, 4)

                obj_in_cam = np.linalg.inv(np.linalg.inv(pose) @ T_cam)
                print('pose: ', obj_in_cam)

                obj_poses[odx] = obj_in_cam.flatten()
                zeds.append(obj_in_cam[2, 3])

            zeds = np.asarray(zeds, dtype=np.float32)
            low2high = np.argsort(zeds)
            for l2hdx in low2high:

                objID = obj_ids[l2hdx]
                pose = np.array(obj_poses[l2hdx]).reshape(4, 4)

                t = pose[:3, 3].T
                R = pose[:3, :3]

                R_list = R.flatten().tolist()
                t_list = t.flatten().tolist()

                rot = tf3d.quaternions.mat2quat(R.reshape(3, 3))
                rot = np.asarray(rot, dtype=np.float32)
                tra = np.asarray(t, dtype=np.float32)
                pose = [tra[0], tra[1], tra[2], rot[0], rot[1], rot[2], rot[3]]
                pose = [np.asscalar(pose[0]), np.asscalar(pose[1]), np.asscalar(pose[2]),
                        np.asscalar(pose[3]), np.asscalar(pose[4]), np.asscalar(pose[5]), np.asscalar(pose[6])]
                #trans = np.asarray([pose[0], pose[1], pose[2]], dtype=np.float32)

                # light, render and append
                light_pose = [np.random.rand() * 3 - 1.0, np.random.rand() * 2 - 1.0, 0.0]
                light_color = [np.random.rand() * 0.1 + 0.9, np.random.rand() * 0.1 + 0.9, np.random.rand() * 0.1 + 0.9]
                light_ambient_weight = 0.2 + np.random.rand() * 0.5

                if objID == 1:  # fine with that
                    light_diffuse_weight = 0.6 + np.random.rand() * 0.3
                    light_spec_weight = 0.1 + np.random.rand() * 0.3
                    light_spec_shine = 0.9 + np.random.rand() * 0.2
                elif objID==5 or objID==6: # metall industrial objects
                    light_diffuse_weight = 0.25 + np.random.rand() * 0.2
                    light_spec_weight = 0.45 + np.random.rand() * 0.3
                    light_spec_shine = 0.5 + np.random.rand() * 0.75
                else: # sink and siphon
                    light_diffuse_weight = 0.4 + np.random.rand() * 0.3
                    light_spec_weight = 0.4 + np.random.rand() * 0.6
                    light_spec_shine = 0.5 + np.random.rand() * 0.25

                ren.set_light(light_pose, light_color, light_ambient_weight, light_diffuse_weight, light_spec_weight,
                              light_spec_shine)
                ren.render_object(objID, R_list, t_list, fx, fy, cx, cy)
                ren_img = ren.get_color_image(objID)

                partial_visib_img = np.where(visib_img > 0, 0, ren_img)
                partial_visib_mask = np.nan_to_num(partial_visib_img, copy=True, nan=0, posinf=0, neginf=0)
                partial_visib_mask = np.where(np.any(partial_visib_mask, axis=2) > 0, 1, 0)
                partial_mask_surf = np.sum(partial_visib_mask)

                print(partial_mask_surf)

                surf_visib = np.sum(ren_img)
                visib_fract = float(partial_mask_surf / surf_visib)
                print('visib_fract: ', visib_fract)
                if visib_fract > 1.0:
                    visib_fract = float(1.0)

                visibilities.append(visib_fract)
                visib_img = np.where(visib_img > 0, visib_img, ren_img)

                # compute bounding box and append
                non_zero = np.nonzero(partial_visib_mask)
                surf_ren = np.sum(non_zero[0])
                if non_zero[0].size != 0:
                    bb_xmin = np.nanmin(non_zero[1])
                    bb_xmax = np.nanmax(non_zero[1])
                    bb_ymin = np.nanmin(non_zero[0])
                    bb_ymax = np.nanmax(non_zero[0])
                    obj_bb = [int(bb_xmin), int(bb_ymin), int(bb_xmax - bb_xmin), int(bb_ymax - bb_ymin)]
                    # out of order with other lists
                    bboxes.append(obj_bb)
                    area = int(obj_bb[2] * obj_bb[3])
                    areas.append(area)
                else:
                    area = int(0)
                    obj_bb = [int(0), int(0), int(0), int(0)]
                    bboxes.append(obj_bb)
                    areas.append(area)

                bg_img = np.where(partial_visib_img > 0, partial_visib_img, bg_img)

                # mask calculation
                mask_id = mask_ind + 1
                mask_img = np.where(partial_visib_img.any(axis=2) > 0, mask_id, mask_img)
                mask_ind = mask_ind + 1

                annoID = annoID + 1
                tDbox = R.reshape(3, 3).dot(threeD_boxes[objID, :, :].T).T
                tDbox = tDbox + np.repeat(t[:, np.newaxis].T, 8, axis=0)
                box3D = toPix_array(tDbox, fx=fx, fy=fy, cx=cx, cy=cy)
                box3D = np.reshape(box3D, (16))
                box3D = box3D.tolist()

                # sym_cont_anno = []
                # sym_disc_anno = []
                # if obj_id == 3:
                #    sym_disc_anno = sym_disc[obj_id, :].tolist()

                # print(sym_disc_anno)

                if visib_fract > 0.5:

                    if area < min_box_area:
                        min_box_area = area
                        min_box = obj_bb

                    if area > max_box_area:
                        max_box_area = area
                        max_box = obj_bb

                tempTA = {
                    "id": annoID,
                    "image_id": img_id,
                    "category_id": int(objID),
                    "bbox": obj_bb,
                    "pose": pose,
                    "segmentation": box3D,
                    "mask_id": mask_id,
                    "area": area,
                    "iscrowd": 0,
                    "feature_visibility": visib_fract,
                }

                dict["annotations"].append(tempTA)
                cnt = cnt + 1

        tempTL = {
            "url": "https://bop.felk.cvut.cz/home/",
            "id": img_id,
            "name": iname,
        }
        dict["licenses"].append(tempTL)
        if myFile.exists():
            print('File exists, skip encoding, ', fileName)
        else:
            cv2.imwrite(fileName, bg_img)
            print("storing image in : ", fileName)
            mask_safe_path = fileName[:-4] + '_mask.png'
            cv2.imwrite(mask_safe_path, mask_img.astype(np.int8))
            tempTV = {
                "license": 2,
                "url": "https://bop.felk.cvut.cz/home/",
                "file_name": iname,
                "height": resY,
                "width": resX,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "date_captured": dateT,
                "id": img_id,
            }
            dict["images"].append(tempTV)
            # CIT
            '''
            if objID == 2 or objID == 5:
                x = (2 * (0.35 * z)) * np.random.rand() - (0.35 * z)  # 0.55 each side kinect
                y = (2 * (0.2 * z)) * np.random.rand() - (0.2 * z)  # 0.40 each side kinect
            else:
                x = (2 * (0.45 * z)) * np.random.rand() - (0.45 * z)  # 0.55 each side kinect
                y = (2 * (0.3 * z)) * np.random.rand() - (0.3 * z)  # 0.40 each side kinect
            '''
            if visu is True:
                boxes3D = [boxes3D[x] for x in low2high]
                obj_ids = [obj_ids[x] for x in low2high]
                #boxes3D = boxes3D[low2high]
                #obj_ids = obj_ids[low2high]
                img = bg_img
                for i, bb in enumerate(bboxes):
                    bb = np.array(bb)
                    cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                                  (255, 255, 255), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (int(bb[0]), int(bb[1]))
                    fontScale = 1
                    fontColor = (0, 0, 0)
                    fontthickness = 1
                    lineType = 2
                    gtText = str(obj_ids[i])
                    fontColor2 = (255, 255, 255)
                    fontthickness2 = 3
                    cv2.putText(img, gtText,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor2,
                                fontthickness2,
                                lineType)
                    pose = np.asarray(boxes3D[i], dtype=np.float32)
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
                    '''
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
                    '''
                # print(camR_vis[i], camT_vis[i])
                # draw_axis(img, camR_vis[i], camT_vis[i], K)
                cv2.imwrite(fileName, img)
                #print('STOP')
        #end_t = time.time()
        #times += end_t - start_t
        #avg_time = times / gloCo
        #rem_time = ((all_data - gloCo) * avg_time) / 60
        #print('time remaining: ', rem_time, ' min')
        #gloCo += 1

    for s in categories:
        objName = str(s)
        tempC = {
            "id": s,
            "name": objName,
            "supercategory": "object"
        }
        dict["categories"].append(tempC)

    valAnno = os.path.join(target, 'annotations/instances_train.json')

    with open(valAnno, 'w') as fpT:
        json.dump(dict, fpT)

    box_anno = os.path.join(target, 'annotations/boxes.txt')

    with open(box_anno, "w") as text_file:
        text_file.write("maximum box: %s \n" % str(max_box))
        text_file.write("minimum box: %s" % str(min_box))

    print('everythings done')
