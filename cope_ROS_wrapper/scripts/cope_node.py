#!/usr/bin/env python

import sys

import rospy
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, Point, Point32, PolygonStamped

from scipy import ndimage, signal
import argparse
import os
import sys
import math
import numpy as np
import copy
import transforms3d as tf3d
import json
import copy

#import keras
import tensorflow as tf
import open3d
import ros_numpy

#print(sys.path)
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
from cv_bridge import CvBridge, CvBridgeError

sys.path.append("/cope/")
from cope import models

from object_detector_msgs.srv import get_poses, get_posesResponse
from object_detector_msgs.msg import PoseWithConfidence
from geometry_msgs.msg import PoseArray, Pose
###################################
##### Global Variable Space #######
######## aka. death zone ##########
###################################

LABEL_CLS_MAPPING = [
    "BottleMedium",
    "BottleSmall",
    "Needle",
    "NeedleCap",
    "RedPlug",
    "Canister",
    "BottleLarge"
]


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


def toPix_array(translation, fx, fy, cx, cy):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1)


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


        with open(os.path.join(mesh_path, 'models_info.json')) as fp:
            mesh_info = json.load(fp)

        self.num_classes = len(mesh_info.keys())
        self.threeD_boxes = np.ndarray((self.num_classes, 8, 3), dtype=np.float32)
        self.sphere_diameters = np.ndarray((self.num_classes), dtype=np.float32)

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
            self.sphere_diameters[int(key)-1] = norm_pts
            # self.num_classes += 1

        self.model = load_model(model_path, self.sphere_diameters, self.num_classes)

        self.pose_srv = rospy.Service(name, get_poses, self.callback)
        rospy.loginfo(f"[{name}] Server ready")

    def _update_image(self, rgb, depth):
        self.rgb, self.depth = rgb, depth

    def callback(self, req):
        print("Received request")
        rgb_cv = self.bridge.imgmsg_to_cv2(self.rgb, "8UC3")
        rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)

        # Run inference
        det_objs, det_poses, det_confs, viz_img = run_estimation(
            rgb_cv, self.model, self.threeD_boxes,
            self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy)
        msg = self.fill_msg(det_objs, det_poses, det_confs)
        self.viz_pose(viz_img)
        return msg

    def fill_msg(self, det_names, det_poses, det_confidences):
        msg = PoseArray()

        msg.header.frame_id = self.camera_info.header.frame_id
        msg.header.stamp = rospy.Time(0)

        for idx in range(len(det_names)):
            item = Pose()
            item.position.x = det_poses[idx][0] 
            item.position.y = det_poses[idx][1] 
            item.position.z = det_poses[idx][2] 
            item.orientation.w = det_poses[idx][3] 
            item.orientation.x = det_poses[idx][4] 
            item.orientation.y = det_poses[idx][5] 
            item.orientation.z = det_poses[idx][6]
            msg.poses.append(item)
        self.pose_pub.publish(msg)

        msg = get_posesResponse()
        for idx in range(len(det_names)):
            item = PoseWithConfidence()
            item.name = det_names[idx] 
            item.confidence = det_confidences[idx]
            item.pose = Pose()
            det_pose = det_poses[idx]
            item.pose.position.x = det_pose[0]
            item.pose.position.y = det_pose[1]
            item.pose.position.z = det_pose[2]
            item.pose.orientation.w = det_pose[3]
            item.pose.orientation.x = det_pose[4]
            item.pose.orientation.y = det_pose[5]
            item.pose.orientation.z = det_pose[6]
            msg.poses.append(item)

        return msg


    def viz_pose(self, image):
        msg = Image()
        msg.header.frame_id = self.camera_info.header.frame_id
        msg.header.stamp = rospy.Time(0)
        data = self.bridge.cv2_to_imgmsg(image, "passthrough")
        self.viz_pub.publish(data)

        
#################################
########## RetNetPose ###########
#################################
def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    return tf.Session(config=config)


def parse_args(args):

    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    parser.add_argument('model',              help='Path to RetinaNet model.')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')

    return parser.parse_args(args)


def load_model(model_path, sphere_diameters, num_classes):


    #if args.gpu:
    #    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #keras.backend.tensorflow_backend.set_session(get_session())

    anchor_params = None
    backbone = 'resnet50'

    print('Loading model, this may take a second...')
    print(model_path)
    model = models.load_model(model_path, backbone_name=backbone)
    #graph = tf.compat.v1.get_default_graph()
    # model = models.convert_model(model, anchor_params=anchor_params) # convert model
    model = models.convert_model(model, diameters=sphere_diameters, classes=num_classes) # TODO diameter and classes
    # print model summary
    print(model.summary())

    return model#, graph


def run_estimation(image, model, threeD_boxes,
                   cam_fx, cam_fy, cam_cx, cam_cy):
    obj_names = []
    obj_poses = []
    obj_confs = []

    image_raw = copy.deepcopy(image)
    image = preprocess_image(image)
    #image_mask = copy.deepcopy(image)

    scores, labels, poses, _, _ = model.predict_on_batch(np.expand_dims(image, axis=0))

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
        est3D = toPix_array(eDbox, cam_fx, cam_fy, cam_cx, cam_cy)
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
        obj_names.append(LABEL_CLS_MAPPING[inv_cls])
        obj_confs.append(score)

    return obj_names, obj_poses, obj_confs, image_raw


if __name__ == '__main__':
    rospy.init_node('locate_object')
    print(rospy.get_name())
    server = PoseEstimation(rospy.get_name())
    rospy.spin()
