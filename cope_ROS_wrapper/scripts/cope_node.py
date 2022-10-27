#!/usr/bin/env python

import sys

import rospy
import message_filters
import tf2_ros

from actionlib import SimpleActionServer
from geometry_msgs.msg import Pose, PoseArray, Quaternion, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from tracebot_msgs.msg import LocateObjectAction, LocateObjectResult

import os
import sys
import math
import numpy as np
import copy
import transforms3d as tf3d
import json

import tensorflow as tf

import cv2
from cv_bridge import CvBridge, CvBridgeError

sys.path.append("/cope/cope/")
from cope import models

###################################
######## Utils funcitons ##########
###################################


def preprocess_image(x, mode='caffe'):
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def input_resize(image, target_size, intrinsics):
    # image: [y, x, c] expected row major
    # target_size: [y, x] expected row major
    # instrinsics: [fx, fy, cx, cy]

    intrinsics = np.asarray(intrinsics)
    y_size, x_size, c_size = image.shape

    if (y_size / x_size) < (target_size[0] / target_size[1]):
        resize_scale = target_size[0] / y_size
        crop = int((x_size - (target_size[1] / resize_scale)) * 0.5)
        image = image[:, crop:(x_size-crop), :]
        image = cv2.resize(image, (int(target_size[1]), int(target_size[0])))
        intrinsics = intrinsics * resize_scale
    else:
        resize_scale = target_size[1] / x_size
        crop = int((y_size - (target_size[0] / resize_scale)) * 0.5)
        image = image[crop:(y_size-crop), :, :]
        image = cv2.resize(image, (int(target_size[1]), int(target_size[0])))
        intrinsics = intrinsics * resize_scale

    return image, intrinsics


def toPix_array(translation, fx, fy, cx, cy):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1)


def run_estimation(image, model, threeD_boxes,
                   cam_fx, cam_fy, cam_cx, cam_cy):
    obj_names = []
    obj_poses = []
    obj_confs = []

    image, intrinsics = input_resize(image,
                         [480, 640],
                         [cam_fx, cam_fy, cam_cx, cam_cy])
    image_raw = copy.deepcopy(image)
    image = preprocess_image(image)
    #image_mask = copy.deepcopy(image)

    scores, labels, poses, mask, boxes = model.predict_on_batch((
            np.expand_dims(image, axis=0),
            np.expand_dims(np.array(intrinsics), axis=0)))
    # scores, labels, poses, _, _ = model.predict_on_batch(np.expand_dims(image, axis=0))

    scores = scores[labels != -1]
    poses = poses[labels != -1]
    labels = labels[labels != -1]

    for odx, inv_cls in enumerate(labels):

        true_cls = inv_cls + 1
        score = scores[odx]
        pose = poses[odx, :]

        R_est = np.array(pose[:9]).reshape((3, 3)).T
        t_est = np.array(pose[-3:]) * 0.001

        ori_points = np.ascontiguousarray(threeD_boxes[inv_cls, :, :], dtype=np.float32)
        eDbox = R_est.dot(ori_points.T).T
        eDbox = eDbox + np.repeat(t_est[np.newaxis, :], 8, axis=0)  # * 0.001
        est3D = toPix_array(eDbox, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3])
        eDbox = np.reshape(est3D, (16))
        pose = eDbox.astype(np.uint16)
        pose = np.where(pose < 3, 3, pose)
        colEst = (50, 205, 50)

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

        est_pose = np.zeros((7), dtype=np.float32)
        est_pose[:3] = t_est
        est_pose[3:] = tf3d.quaternions.mat2quat(R_est)
        obj_poses.append(est_pose)
        obj_names.append(inv_cls)
        obj_confs.append(score)

    return obj_names, obj_poses, obj_confs, image_raw


#################################
############### ROS #############
#################################
class PoseEstimation:
    def __init__(self, name):
        # COPE
        mesh_path = rospy.get_param('/locateobject/mesh_path', "/data/tracebot_objects_V1")
        model_path = rospy.get_param('/locateobject/model_path', "/data/cope_tracebot_120.h5")
        self._score_th = rospy.get_param("/locateobject/detection_threshold", 0.5)

        # Camera
        self.rgb, self.depth = None, None
        self.color_topic = rospy.get_param('/locateobject/color_topic',
                                           '/camera/color/image_raw')
        self.depth_topic = rospy.get_param('/locateobject/depth_topic',
                                           '/camera/aligned_depth_to_color/image_raw')
        self.camera_info_topic = rospy.get_param('/locateobject/camera_info_topic',
                                                 '/camera/color/camera_info')

        self.slop = 0.2  # max delay between rgb and depth image [seconds]
        sub_rgb, sub_depth = message_filters.Subscriber(self.color_topic, Image),\
                             message_filters.Subscriber(self.depth_topic, Image)
        sub_rgbd = message_filters.ApproximateTimeSynchronizer([sub_rgb, sub_depth], 10, self.slop)
        sub_rgbd.registerCallback(self._update_image)

        # Camera intrinsics
        rospy.loginfo(f"[{name}] Waiting for camera info...")
        self.camera_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo)
        self.intrinsics = np.array([v for v in self.camera_info.K]).reshape(3, 3)
        self.cam_fx, self.cam_fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        self.cam_cx, self.cam_cy = self.intrinsics[0, 2], self.intrinsics[1, 2]

        self.bridge = CvBridge()
        self.pose_pub = rospy.Publisher("/locateobject/poses", PoseArray, queue_size=10)
        self.viz_pub = rospy.Publisher("/locateobject/debug_visualization", Image, queue_size=10)

        self.object_models = [
            "BottleMedium",
            "BottleSmall",
            "Needle",
            "NeedleCap",
            "RedPlug",
            "Canister",
            "BottleLarge"
        ]

        with open(os.path.join(mesh_path, 'models_info.json')) as fp:
            mesh_info = json.load(fp)

        self.num_classes = len(mesh_info.keys())
        self.threeD_boxes = np.ndarray((self.num_classes, 8, 3), dtype=np.float32)
        self.obj_diameters = np.ndarray((self.num_classes), dtype=np.float32)

        for key, value in mesh_info.items():
            fac = 0.001
            x_minus = value['min_x'] * fac
            y_minus = value['min_y'] * fac
            z_minus = value['min_z'] * fac
            x_plus = value['size_x'] * fac + x_minus
            y_plus = value['size_y'] * fac + y_minus
            z_plus = value['size_z'] * fac + z_minus
            norm_pts = np.linalg.norm(np.array([value['size_x'], value['size_y'], value['size_z']]))
            three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                   [x_plus, y_plus, z_minus],
                                   [x_plus, y_minus, z_minus],
                                   [x_plus, y_minus, z_plus],
                                   [x_minus, y_plus, z_plus],
                                   [x_minus, y_plus, z_minus],
                                   [x_minus, y_minus, z_minus],
                                   [x_minus, y_minus, z_plus]])
            self.threeD_boxes[int(key)-1, :, :] = three_box_solo
            self.obj_diameters[int(key)-1] = norm_pts
            # self.num_classes += 1

        # self.model = load_model(model_path, self.obj_diameters, self.num_classes)
        self.model = models.load_model(model_path)
        self.model = models.convert_model(self.model,
                                          diameters=self.obj_diameters,
                                          classes=self.num_classes)
        rospy.logdebug(self.model.summary())

        # create server
        self._server = SimpleActionServer(name, LocateObjectAction, execute_cb=self.callback, auto_start=False)
        self._server.start()
        rospy.loginfo(f"[{name}] Action Server ready")

        if rospy.get_param('/locateobject/publish_tf', True):
            self._br = tf2_ros.TransformBroadcaster()
            self._publish_tf()

    def _update_image(self, rgb, depth):
        self.rgb, self.depth = rgb, depth

    def _publish_tf(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if not hasattr(self, '_last_result'):
                rate.sleep()
                continue

            object_types_count = {}
            for idx, otype in enumerate(self._last_result.object_types):
                ocount = object_types_count.get(otype, 1)
                object_types_count[otype] = ocount + 1
                opose = self._last_result.object_poses[idx]

                # Create the TransformStamped message
                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = self.camera_info.header.frame_id
                t.child_frame_id = f"{otype}_{ocount}"
                t.transform.translation.x = opose.position.x
                t.transform.translation.y = opose.position.y
                t.transform.translation.z = opose.position.z
                t.transform.rotation.x = opose.orientation.x
                t.transform.rotation.y = opose.orientation.y
                t.transform.rotation.z = opose.orientation.z
                t.transform.rotation.w = opose.orientation.w

                self._br.sendTransform(t)

            rate.sleep()

    def callback(self, goal):
        print("Received request")
        if self.rgb is None or self.depth is None:
            self._server.set_aborted(text=f"No synchronized camera image available for ({self.color_topic}, "
                                          f"{self.depth_topic}) with max delay {self.slop:0.3f}s.")
        elif goal.object_to_locate not in self.object_models + [""]:
            self._server.set_aborted(text=f"Unknown object_to_locate='{goal.object_to_locate}'. "
                                          f"Available objects are: {self.object_models}.")
        else:
            rgb_cv = self.bridge.imgmsg_to_cv2(self.rgb, "8UC3")
            rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)

            # Run inference
            det_objs, det_poses, det_confs, viz_img = run_estimation(
                rgb_cv, self.model, self.threeD_boxes,
                self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy)

            result = self.fill_msg(goal.object_to_locate, det_objs, det_poses, det_confs)
            self.viz_pose(viz_img)

            self._last_result = result
            self._server.set_succeeded(result)

    def fill_msg(self, object_to_locate, det_names, det_poses, det_confidences):
        msg = PoseArray()

        msg.header.frame_id = self.camera_info.header.frame_id
        msg.header.stamp = rospy.Time(0)

        result = LocateObjectResult()
        result.header = self.rgb.header
        result.color_image = self.rgb
        result.depth_image = self.depth
        result.camera_info = self.camera_info
        for det_name, det_pose, det_conf in zip(det_names, det_poses, det_confidences):
            obj_name = self.object_models[det_name]
            if object_to_locate != "" and obj_name != object_to_locate:
                continue
            item = Pose()
            item.position.x = det_pose[0]
            item.position.y = det_pose[1]
            item.position.z = det_pose[2]
            item.orientation.w = det_pose[3]
            item.orientation.x = det_pose[4]
            item.orientation.y = det_pose[5]
            item.orientation.z = det_pose[6]
            msg.poses.append(item)
            result.object_poses.append(item) # object_poses
            result.object_types.append(obj_name) # object_types
            result.confidences.append(det_conf) # confidences
        
        self.pose_pub.publish(msg)
        return result
    
    def viz_pose(self, image):
        msg = Image()
        msg.header.frame_id = self.camera_info.header.frame_id
        msg.header.stamp = self.camera_info.header.stamp
        data = self.bridge.cv2_to_imgmsg(image, "passthrough")
        self.viz_pub.publish(data)

if __name__ == '__main__':
    rospy.init_node('locate_object')
    print(rospy.get_name())
    server = PoseEstimation(rospy.get_name())
    rospy.spin()
