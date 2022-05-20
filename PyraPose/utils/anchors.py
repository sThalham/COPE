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
import math

import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import transforms3d as tf3d
import cv2
from PIL import Image
import math
import matplotlib.pyplot as plt
import copy
from .visualization import Visualizer
import time
from .ego_to_allo import egocentric_to_allocentric


def anchor_targets_bbox(
    image_group,
    annotations_group,
    num_classes,
):

    assert(len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert(len(annotations_group) > 0), "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert('bboxes' in annotations), "Annotations should contain boxes."
        assert('labels' in annotations), "Annotations should contain labels."
        assert('poses' in annotations), "Annotations should contain labels."
        assert('segmentations' in annotations), "Annotations should contain poses"

    batch_size = len(image_group)
    pyramid_levels = [3, 4, 5]
    image_shapes = guess_shapes(image_group[0].shape[:2], pyramid_levels)
    location_shape = int(image_shapes[0][1] * image_shapes[0][0]) + int(image_shapes[1][1] * image_shapes[1][0]) + int(image_shapes[2][1] * image_shapes[2][0])
    location_offset = [0, int(image_shapes[0][1] * image_shapes[0][0]), int(image_shapes[0][1] * image_shapes[0][0]) + int(image_shapes[1][1] * image_shapes[1][0])]

    regression_batch = np.zeros((batch_size, location_shape, num_classes, 8, 16 + 1), dtype=keras.backend.floatx())
    #bbox_batch = np.zeros((batch_size, location_shape, num_classes, 4 + 4), dtype=keras.backend.floatx())
    labels_batch = np.zeros((batch_size, location_shape, num_classes + 1), dtype=keras.backend.floatx())
    locations_batch = np.zeros((batch_size, location_shape, num_classes, 3 + 1), dtype=keras.backend.floatx())
    rotations_batch = np.zeros((batch_size, location_shape, num_classes, 8, 6 + 1), dtype=keras.backend.floatx())
    reprojection_batch = np.zeros((batch_size, location_shape, num_classes), dtype=keras.backend.floatx())

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):

        #image_raw = image
        #image_raw[..., 0] += 103.939
        #image_raw[..., 1] += 116.779
        #image_raw[..., 2] += 123.68
        #is_there_sym = False
        #raw_images = []
        #raw_images.append(copy.deepcopy(image_raw))
        #raw_images.append(copy.deepcopy(image_raw))
        #raw_images.append(copy.deepcopy(image_raw))
        ##raw_images.append(copy.deepcopy(image_raw))
        #img_P3 = copy.deepcopy(image_raw)
        #img_P4 = copy.deepcopy(image_raw)
        #img_P5 = copy.deepcopy(image_raw)

        image_locations = locations_for_shape(image.shape)
        image_locations_rep = np.repeat(image_locations[:, np.newaxis, :], repeats=8, axis=1)
        # w/o mask
        mask = annotations['mask'][0]
        mask = cv2.medianBlur(mask, 7)

        # vanilla
        masks_level = []
        for jdx, resx in enumerate(image_shapes):
            mask_level = np.asarray(Image.fromarray(mask).resize((resx[1], resx[0]), Image.NEAREST)).flatten()
            masks_level.append(mask_level.flatten())
            back_objs = np.where(mask_level == 0)[0] + location_offset[jdx]
            labels_batch[index, back_objs, -1] = 0

        #calculated_boxes = np.empty((0, 16))

        for idx, pose in enumerate(annotations['poses']):

            cls = int(annotations['labels'][idx])
            mask_id = annotations['mask_ids'][idx]
            obj_diameter = annotations['diameters'][idx]

            # pyrmid_index from diameter
            ex = obj_diameter / pose[2]
            #reso_van = np.round(np.log(ex) / np.log(3.5))
            reso_van = np.round(np.log(ex) / np.log(3.0))
            #for norm of object diemnsions
            # reason:
            # long-shaped object fall into lower levels than diameter
            # while boxy objects
            if reso_van < -2:
                reso_van = -2
            reso_idx = int(2 + reso_van)
            locations_positive_obj = np.where(masks_level[reso_idx] == int(mask_id))[0] + location_offset[reso_idx]

            #locations_positive_obj = []
            #for reso_idx in range(len(masks_level)):
            #    locations_positive_obj.append(
            #        np.where(masks_level[reso_idx] == int(mask_id))[0] + location_offset[reso_idx])
            #locations_positive_obj = np.concatenate(locations_positive_obj, axis=0)

            #if reso_idx==0:
            #vizmask = np.zeros((4800, 3))
            #locations_positive_obj_3 = np.where(masks_level[0] == int(mask_id))[0]
            #vizmask[locations_positive_obj_3, :] = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            #vizmask = np.reshape(vizmask, (60, 80, 3))
            #vizmask = cv2.resize(vizmask, (640, 480), interpolation=cv2.INTER_NEAREST)
            #img_P3 = np.where(vizmask > 0, vizmask, img_P3)
            ##elif reso_idx==1:
            #vizmask = np.zeros((1200, 3))
            #locations_positive_obj_4 = np.where(masks_level[1] == int(mask_id))[0]
            #vizmask[locations_positive_obj_4, :] = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            #vizmask = np.reshape(vizmask, (30, 40, 3))
            #vizmask = cv2.resize(vizmask, (640, 480), interpolation=cv2.INTER_NEAREST)
            #img_P4 = np.where(vizmask > 0, vizmask, img_P4)
            ##elif reso_idx==2:
            #vizmask = np.zeros((300, 3))
            #locations_positive_obj_5 = np.where(masks_level[2] == int(mask_id))[0]
            #vizmask[locations_positive_obj_5, :] = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            #vizmask = np.reshape(vizmask, (15, 20, 3))
            #vizmask = cv2.resize(vizmask, (640, 480), interpolation=cv2.INTER_NEAREST)
            #img_P5 = np.where(vizmask > 0, vizmask, img_P5)


            #ego_pose = np.eye(4)
            #ego_pose[:3, :3] = tf3d.quaternions.quat2mat(pose[3:])
            #ego_pose[:3, 3] = pose[:3]
            #allo_pose = egocentric_to_allocentric(ego_pose)

            if locations_positive_obj.shape[0] > 1:

                labels_batch[index, locations_positive_obj, -1] = 1
                labels_batch[index, locations_positive_obj, cls] = 1

                '''
                trans = np.eye(4)
                trans[:3, :3] = tf3d.quaternions.quat2mat(pose[3:]).reshape((3, 3))
                trans[:3, 3] = pose[:3]
                rot = np.asarray(trans[:3, :3], dtype=np.float32)
                tra = trans[:3, 3]
                tDbox = rot[:3, :3].dot(annotations['segmentations'][idx].T).T
                tDbox = tDbox + np.repeat(tra[np.newaxis, 0:3], 8, axis=0)

                box3D = toPix_array(tDbox, fx=annotations['cam_params'][idx][0],
                                    fy=annotations['cam_params'][idx][1],
                                    cx=annotations['cam_params'][idx][2], cy=annotations['cam_params'][idx][3])
                box3D = np.reshape(box3D, (16))
                image_ns = raw_images[0]
                tDbox = box3D.astype(np.uint16)
                colEst = (255, 0, 0)
                colGT = (0, 204, 0)
                image_ns = cv2.line(image_ns, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()),
                                    colEst, 2)
                image_ns = cv2.line(image_ns, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()),
                                    colGT, 2)
                image_ns = cv2.line(image_ns, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()),
                                    colGT,
                                    2)
                image_ns = cv2.line(image_ns, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()),
                                    colGT,
                                    2)
                image_ns = cv2.line(image_ns, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()),
                                    colGT,
                                    2)
                image_ns = cv2.line(image_ns, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()),
                                    colGT,
                                    2)
                image_ns = cv2.line(image_ns, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()),
                                    colGT,
                                    2)
                image_ns = cv2.line(image_ns, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()),
                                    colGT,
                                    2)
                image_ns = cv2.line(image_ns, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                                    colGT,
                                    2)
                image_ns = cv2.line(image_ns, tuple(tDbox[10:12].ravel()),
                                    tuple(tDbox[12:14].ravel()),
                                    colGT,
                                    2)
                image_ns = cv2.line(image_ns, tuple(tDbox[12:14].ravel()),
                                    tuple(tDbox[14:16].ravel()),
                                    colGT,
                                    2)
                image_ns = cv2.line(image_ns, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                                    colGT,
                                    2)
                raw_images[0] = image_ns
                '''

                #sym_viz = False # parameter to visualize symmetries
                # handling rotational symmetries
                if np.sum(annotations['sym_con'][idx][0, :]) > 0:
                    #allo_pose = get_cont_sympose(allo_full, annotations['sym_con'][idx])
                    trans = np.eye(4)
                    trans[:3, :3] = tf3d.quaternions.quat2mat(pose[3:]).reshape((3, 3))
                    trans[:3, 3] = pose[:3]
                    pose_mat = get_cont_sympose(trans, annotations['sym_con'][idx])
                    pose[3:] = tf3d.quaternions.mat2quat(pose_mat[:3, :3])
                    '''

                    #sym_viz = True
                    #is_there_sym = True
                    #if sym_viz:
                    # viz from here on

                    rot = np.asarray(allo_pose[:3, :3], dtype=np.float32)
                    tra = allo_pose[:3, 3]
                    tDbox = rot[:3, :3].dot(annotations['segmentations'][idx].T).T
                    tDbox = tDbox + np.repeat(tra[np.newaxis, 0:3], 8, axis=0)

                    box3D = toPix_array(tDbox, fx=annotations['cam_params'][idx][0],
                                        fy=annotations['cam_params'][idx][1],
                                        cx=annotations['cam_params'][idx][2], cy=annotations['cam_params'][idx][3])
                    box3D = np.reshape(box3D, (16))
                    image_ns = raw_images[1]
                    tDbox = box3D.astype(np.uint16)
                    colGT = (255, 0, 0)
                    colEst = (0, 204, 0)
                    image_ns = cv2.line(image_ns, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()),
                                        colEst, 2)
                    image_ns = cv2.line(image_ns, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()),
                                        colGT, 2)
                    image_ns = cv2.line(image_ns, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()),
                                        colGT,
                                        2)
                    image_ns = cv2.line(image_ns, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()),
                                        colGT,
                                        2)
                    image_ns = cv2.line(image_ns, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()),
                                        colGT,
                                        2)
                    image_ns = cv2.line(image_ns, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()),
                                        colGT,
                                        2)
                    image_ns = cv2.line(image_ns, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()),
                                        colGT,
                                        2)
                    image_ns = cv2.line(image_ns, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()),
                                        colGT,
                                        2)
                    image_ns = cv2.line(image_ns, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                                        colGT,
                                        2)
                    image_ns = cv2.line(image_ns, tuple(tDbox[10:12].ravel()),
                                        tuple(tDbox[12:14].ravel()),
                                        colGT,
                                        2)
                    image_ns = cv2.line(image_ns, tuple(tDbox[12:14].ravel()),
                                        tuple(tDbox[14:16].ravel()),
                                        colGT,
                                        2)
                    image_ns = cv2.line(image_ns, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                                        colGT,
                                        2)
                    '''
                    #image_raw = image_ns
                        #raw_images[0] = image_ns

                rot = tf3d.quaternions.quat2mat(pose[3:])
                rot = np.asarray(rot, dtype=np.float32)
                tra = pose[:3]
                #rot = np.asarray(allo_pose[:3, :3], dtype=np.float32)
                #tra = allo_pose[:3, 3]
                full_T = np.eye((4))
                full_T[:3, :3] = rot
                full_T[:3, 3] = tra
                tDbox = rot[:3, :3].dot(annotations['segmentations'][idx].T).T
                tDbox = tDbox + np.repeat(tra[np.newaxis, 0:3], 8, axis=0)

                box3D = toPix_array(tDbox, fx=annotations['cam_params'][idx][0], fy=annotations['cam_params'][idx][1],
                                           cx=annotations['cam_params'][idx][2], cy=annotations['cam_params'][idx][3])
                box3D = np.reshape(box3D, (16))
                #calculated_boxes = np.concatenate([calculated_boxes, [box3D]], axis=0)

                # handling discrete symmetries
                hyps_boxes = np.repeat(box3D[np.newaxis, :], repeats=8, axis=0)
                #calculated_boxes = np.concatenate([calculated_boxes, hyps_boxes], axis=0)

                #hyps_pose = np.repeat(allo_pose[np.newaxis, :, :], repeats=8, axis=0)
                hyps_pose = np.repeat(full_T[np.newaxis, :, :], repeats=8, axis=0)
                symmetry_mask = np.zeros(8)
                symmetry_mask[0] = 1

                is_sym = False
                sym_disc = annotations['sym_dis'][idx]
                if np.sum(np.abs(sym_disc)) != 0:
                    for sdx in range(sym_disc.shape[0]):
                        if np.sum(np.abs(sym_disc[sdx, :])) != 0:
                            T_sym = np.matmul(full_T, np.array(sym_disc[sdx, :]).reshape((4, 4)))
                            #allo_sym = np.matmul(allo_pose, np.array(sym_disc[sdx, :]).reshape((4, 4)))
                            #hyps_pose[sdx, :, :] = allo_sym
                            hyps_pose[sdx, :, :] = T_sym
                            is_sym = True
                            rot_sym = T_sym[:3, :3]
                            tra = T_sym[:3, 3]
                            tDbox = rot_sym.dot(annotations['segmentations'][idx].T).T
                            tDbox = tDbox + np.repeat(tra[np.newaxis, 0:3], 8, axis=0)

                            box3D_sym = toPix_array(tDbox, fx=annotations['cam_params'][idx][0],
                                                fy=annotations['cam_params'][idx][1],
                                                cx=annotations['cam_params'][idx][2],
                                                cy=annotations['cam_params'][idx][3])
                            box3D_sym = np.reshape(box3D_sym, (16))
                            hyps_boxes[sdx, :] = box3D_sym
                            symmetry_mask[sdx+1] = 1
                            # viz from here on
                            '''
                            
                            sym_viz = True
                            is_there_sym = True
                            if sdx < 3:
                                image_now = raw_images[sdx+1]
                                tDbox = box3D_sym.astype(np.uint16)
                                colGT = (255, 0, 0)
                                colEst = (0, 204, 0)
                                image_now = cv2.line(image_now, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()),
                                                     colEst, 2)
                                image_now = cv2.line(image_now, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()),
                                                     colGT, 2)
                                image_now = cv2.line(image_now, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()),
                                                     colGT,
                                                     2)
                                image_now = cv2.line(image_now, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()),
                                                     colGT,
                                                     2)
                                image_now = cv2.line(image_now, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()),
                                                     colGT,
                                                     2)
                                image_now = cv2.line(image_now, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()),
                                                     colGT,
                                                     2)
                                image_now = cv2.line(image_now, tuple(tDbox[4:6].ravel()), tuplexpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy(tDbox[12:14].ravel()),
                                                     colGT,
                                                     2)
                                image_now = cv2.line(image_now, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()),
                                                     colGT,
                                                     2)
                                image_now = cv2.line(image_now, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                                                     colGT,
                                                     2)
                                image_now = cv2.line(image_now, tuple(tDbox[10:12].ravel()),
                                                     tuple(tDbox[12:14].ravel()),
                                                     colGT,
                                                     2)
                                i            print('anno: ', annotations)mage_now = cv2.line(image_now, tuple(tDbox[12:14].ravel()),
                                                     tuple(tDbox[14:16].ravel()),
                                                     colGT,
                                                     2)
                                image_now = cv2.line(image_now, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                                                     colGT,
                                                     2)
                                raw_images[sdx + 1] = image_now
                    #print('hyps pose: ', hyps_pose)
                    '''

                #if is_sym == True:
                    #print('cls: ', cls)
                    #print('sym: ', sym_disc.shape[0], symmetry_mask)

                #points = box3D_transform(box3D, image_locations[locations_positive_obj, :], obj_diameter)
                points = box3D_transform_symmetric(hyps_boxes, image_locations_rep[locations_positive_obj, :, :], obj_diameter)
                regression_batch[index, locations_positive_obj, cls, :, :16] = points
                #regression_batch[index, locations_positive_obj, cls, :, -1] = 1
                regression_batch[index, locations_positive_obj, cls, :, -1] = symmetry_mask

                #locations_batch[index, locations_positive_obj, cls, :, :2] = hyps_pose[:, :2, 3] * 0.002
                #locations_batch[index, locations_positive_obj, cls, :, 2] = ((hyps_pose[:, 2, 3] * 0.001) - 1.0) * 3.0
                #locations_batch[index, locations_positive_obj, cls, :, -1] = 1

                locations_batch[index, locations_positive_obj, cls, :2] = tra[:2] * 0.002
                locations_batch[index, locations_positive_obj, cls, 2] = ((tra[2] * 0.001) - 1.0) * 3.0
                locations_batch[index, locations_positive_obj, cls, -1] = 1

                rotations_batch[index, locations_positive_obj, cls, :, :6] = np.transpose(hyps_pose[:, :3, :2], axes=(0, 2, 1)).reshape(8, 6)
                #rotations_batch[index, locations_positive_obj, cls, :6] = full_T[:3, :2].T.reshape(6)
                #rotations_batch[index, locations_positive_obj, cls, :, -1] = 1
                rotations_batch[index, locations_positive_obj, cls, :, -1] = symmetry_mask

                #reprojection_batch[index, locations_positive_obj, cls, 16:] = 1
                reprojection_batch[index, locations_positive_obj, cls] = 1


                #rot = np.asarray(pose[], dtype=np.float32)
                #tra = pose[:3]
                #tDbox = rot[:3, :3].dot(annotations['segmentations'][idx].T).T
                #tDbox = tDbox + np.repeat(tra[np.newaxis, 0:3], 8, axis=0)
                # add noise to pose
                #box3D = toPix_array(tDbox, fx=annotations['cam_params'][idx][0], fy=annotations['cam_params'][idx][1],
                #                    cx=annotations['cam_params'][idx][2], cy=annotations['cam_params'][idx][3])
                #box3D = np.reshape(box3D, (16))

                '''
                tDbox = box3D.astype(np.uint16)
                colGT = (0, 205, 0)
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
                '''

        #rind = np.random.randint(0, 1000)
        #images_row1 = np.concatenate([raw_images[0], raw_images[1]], axis=1)
        #images_row2 = np.concatenate([raw_images[2], raw_images[3]], axis=1)
        #image_raw = np.concatenate([images_row1, images_row2], axis=0)
        #name = '/home/stefan/PyraPose_viz/anno_' + str(rind) + 'RGB.jpg'
        #cv2.imwrite(name, image_raw)


        #img_P3 = img_P3.astype(np.uint8)
        #img_P4 = img_P4.astype(np.uint8)
        #img_P5 = img_P5.astype(np.uint8)
        #rind = np.random.randint(0, 1000)
        #name = '/home/stefan/PyraPose_viz/anno_' + str(rind) + 'P3.jpg'
        #cv2.imwrite(name, img_P3)
        #name = '/home/stefan/PyraPose_viz/anno_' + str(rind) + 'P4.jpg'
        #cv2.imwrite(name, img_P4)
        #name = '/home/stefan/PyraPose_viz/anno_' + str(rind) + 'P5.jpg'
        #cv2.imwrite(name, img_P5)

        '''
        #single_hyp_box = np.amax(regression_batch[:, :, :, :, :-1], axis=3)
        #print('sum bboxes: ', np.sum(single_hyp_box))
        #print('true locations: ', np.sum(np.amax(np.amax(regression_batch[:, :, :, :, -1], axis=-1), axis=-1)))
        #print('conf: ', np.mean(confidences_batch[:, :, :, 16:23]), np.max(confidences_batch[:, :, :, 16:23]), np.min(confidences_batch[:, :, :, 16:23]))
        '''

    return regression_batch, labels_batch, locations_batch, rotations_batch, reprojection_batch#, reprojection_batch
    #return tf.convert_to_tensor(regression_batch), tf.convert_to_tensor(labels_batch), tf.convert_to_tensor(locations_batch), tf.convert_to_tensor(rotations_batch), tf.convert_to_tensor(reprojection_batch)#, tf.convert_to_tensor(confidences_batch)


def layer_shapes(image_shape, model):
    """Compute layer shapes given input image shape and the model.

    Args
        image_shape: The shape of the image.
        model: The model to use for computing how the image shape is transformed in the pyramid.

    Returns
        A dictionary mapping layer names to image shapes.
    """
    shape = {
        model.layers[0].name: (None,) + image_shape,
    }

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            inputs = [shape[lr.name] for lr in node.inbound_layers]
            if not inputs:
                continue
            shape[layer.name] = layer.compute_output_shape(inputs[0] if len(inputs) == 1 else inputs)

    return shape


def make_shapes_callback(model):
    """ Make a function for getting the shape of the pyramid levels.
    """
    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape["P{}".format(level)][1:3] for level in pyramid_levels]
        return image_shapes

    return get_shapes


def guess_shapes(image_shape, pyramid_levels):
    """Guess shapes based on pyramid levels.

    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def locations_for_shape(
    image_shape,
    pyramid_levels=None,
    shapes_callback=None,
    distributions=None,
):

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5]
        #pyramid_levels = [3, 3.5, 4, 4.5, 5]

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    if distributions is None:
        ratios = np.array([[1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4], [0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]])

    # compute anchors over all pyramid levels
    all_locations = np.zeros((0, 2))
    for idx, p in enumerate(image_shapes):
        ny, nx = p
        sy = image_shape[0] / ny
        sx = image_shape[1] / nx
        #print('nxy: ', nx, ny)
        #print('sxy: ', sx, sy)

        y = np.linspace(0, image_shape[0]-sy, num=int(ny)) + sy/2
        x = np.linspace(0, image_shape[1]-sx, num=int(nx)) + sx/2

        xv, yv = np.meshgrid(x, y)

        locations_level = np.concatenate([xv.flatten()[:, np.newaxis], yv.flatten()[:, np.newaxis]], axis=1)

        all_locations     = np.append(all_locations, locations_level, axis=0)

    #print('P3: ', all_locations[0:5, :2])
    #print('P4: ', all_locations[4800:4805, :2])
    #print('P5: ', all_locations[6000:6005, :2])

    return all_locations


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def box3D_transform(box, locations, obj_diameter, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.full(16, 0)  # np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    if std is None:
        std = np.full(16, 0.65)

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std) 
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    targets_dx0 = locations[:, 0] - box[0]
    targets_dy0 = locations[:, 1] - box[1]
    targets_dx1 = locations[:, 0] - box[2]
    targets_dy1 = locations[:, 1] - box[3]
    targets_dx2 = locations[:, 0] - box[4]
    targets_dy2 = locations[:, 1] - box[5]
    targets_dx3 = locations[:, 0] - box[6]
    targets_dy3 = locations[:, 1] - box[7]
    targets_dx4 = locations[:, 0] - box[8]
    targets_dy4 = locations[:, 1] - box[9]
    targets_dx5 = locations[:, 0] - box[10]
    targets_dy5 = locations[:, 1] - box[11]
    targets_dx6 = locations[:, 0] - box[12]
    targets_dy6 = locations[:, 1] - box[13]
    targets_dx7 = locations[:, 0] - box[14]
    targets_dy7 = locations[:, 1] - box[15]

    targets = np.stack((targets_dx0, targets_dy0, targets_dx1, targets_dy1, targets_dx2, targets_dy2, targets_dx3, targets_dy3, targets_dx4, targets_dy4, targets_dx5, targets_dy5, targets_dx6, targets_dy6, targets_dx7, targets_dy7), axis=1)
    targets = (targets - mean) / (std * obj_diameter)

    return targets


def box3D_transform_symmetric(box, locations, obj_diameter, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.full(16, 0)  # np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    if std is None:
        std = np.full(16, 0.65)

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    targets_dx0 = locations[:, :, 0] - box[:, 0]
    targets_dy0 = locations[:, :, 1] - box[:, 1]
    targets_dx1 = locations[:, :, 0] - box[:, 2]
    targets_dy1 = locations[:, :, 1] - box[:, 3]
    targets_dx2 = locations[:, :, 0] - box[:, 4]
    targets_dy2 = locations[:, :, 1] - box[:, 5]
    targets_dx3 = locations[:, :, 0] - box[:, 6]
    targets_dy3 = locations[:, :, 1] - box[:, 7]
    targets_dx4 = locations[:, :, 0] - box[:, 8]
    targets_dy4 = locations[:, :, 1] - box[:, 9]
    targets_dx5 = locations[:, :, 0] - box[:, 10]
    targets_dy5 = locations[:, :, 1] - box[:, 11]
    targets_dx6 = locations[:, :, 0] - box[:, 12]
    targets_dy6 = locations[:, :, 1] - box[:, 13]
    targets_dx7 = locations[:, :, 0] - box[:, 14]
    targets_dy7 = locations[:, :, 1] - box[:, 15]

    targets = np.stack((targets_dx0, targets_dy0, targets_dx1, targets_dy1, targets_dx2, targets_dy2, targets_dx3, targets_dy3, targets_dx4, targets_dy4, targets_dx5, targets_dy5, targets_dx6, targets_dy6, targets_dx7, targets_dy7), axis=2)
    targets = (targets - mean) / (std * obj_diameter)

    return targets


def boxes_transform(box, locations, obj_diameter, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.full(4, 0)  # np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    if std is None:
        std = np.full(4, 0.65)

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    targets_dx0 = locations[:, 0] - box[0]
    targets_dy0 = locations[:, 1] - box[1]
    targets_dx1 = locations[:, 0] - box[2]
    targets_dy1 = locations[:, 1] - box[3]

    targets = np.stack((targets_dx0, targets_dy0, targets_dx1, targets_dy1), axis=1)
    targets = (targets - mean) / (std * obj_diameter)
    print('boxes: ', np.mean(targets), np.var(targets))

    return targets


def toPix_array(translation, fx=None, fy=None, cx=None, cy=None):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy

    return np.stack((xpix, ypix), axis=1)


def get_cont_sympose(rot_pose, sym):

    #print('trans: ', rot_pose)
    cam_in_obj = np.dot(np.linalg.inv(rot_pose), (0, 0, 0, 1))
    if sym[0][2] == 1:
        alpha = math.atan2(cam_in_obj[1], cam_in_obj[0])
        rota = np.dot(rot_pose[:3, :3], tf3d.euler.euler2mat(0.0, 0.0, alpha, 'sxyz'))
    elif sym[0][1] == 1:
        alpha = math.atan2(cam_in_obj[0], cam_in_obj[2])
        rota = np.dot(rot_pose[:3, :3], tf3d.euler.euler2mat(0.0, alpha, 0.0, 'sxyz'))
    elif sym[0][0] == 1:
        alpha = math.atan2(cam_in_obj[2], cam_in_obj[1])
        rota = np.dot(rot_pose[:3, :3], tf3d.euler.euler2mat(alpha, 0.0, 0.0, 'sxyz'))
    #rot_pose[3:] = tf3d.quaternions.mat2quat(rota)
    rot_pose[:3, :3] = rota

    return rot_pose

