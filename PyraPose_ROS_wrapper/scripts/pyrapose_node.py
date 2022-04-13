#!/usr/bin/env python

import sys

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseStamped, Point, Point32, PolygonStamped
from cv_bridge import CvBridge, CvBridgeError

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
from PIL import Image as Pilimage

sys.path.append("/stefan/PyraPoseAF")
from PyraPose import models

from object_detector_msgs.srv import get_poses, get_posesResponse
from object_detector_msgs.msg import PoseWithConfidence
from geometry_msgs.msg import PoseArray, Pose
###################################
##### Global Variable Space #######
######## aka. death zone ##########
###################################



# LineMOD
#fxkin = 572.41140
#fykin = 573.57043
#cxkin = 325.26110
#cykin = 242.04899

# YCB-video
#fxkin = 1066.778
#fykin = 1067.487
#cxkin = 312.9869
#cykin = 241.3109

# our Kinect
#fxkin = 575.81573
#fykin = 575.81753
#cxkin = 314.5
#cykin = 235.5

# HSRB
fxhsr = 538.391033
fyhsr = 538.085452
cxhsr_van = 315.30747
cyhsr_van = 233.048356
cxhsr = 320
cyhsr = 240

# magic intrinsics
#fxkin = 1066.778
#fykin = 1067.487
#cxkin = 320.0
#cykin = 240.0


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
    cloud_final[cloud_final[:,2]==0] = np.NaN

    return cloud_final


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
class PoseEstimationClass:
    #def __init__(self, model, mesh_path, threshold, topic, graph):
    def __init__(self, model, mesh_path, threshold, topic):
        #event that will block until the info is received
        #attribute for storing the rx'd message
        self._score_th = threshold
        #self.graph = graph

        self._msg = None
        self.seq = None
        self.time = None
        self.frame_id = None
        self.bridge = CvBridge()
        self.pose_pub = rospy.Publisher("/pyrapose/poses", PoseArray, queue_size=10)
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.depth_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/image_raw', Image, self.depth_callback)

        self.threeD_boxes = np.ndarray((1, 8, 3), dtype=np.float32)
        self.sphere_diameters = np.ndarray((1), dtype=np.float32)
        self.num_classes = 0
        mesh_info = os.path.join(mesh_path, 'models_info.json')
        for key, value in json.load(open(mesh_info)).items():
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
            self.num_classes += 1

        self._model = model = load_model(model, self.sphere_diameters, self.num_classes)

    def depth_callback(self, data):
        self.depth = data

    def callback(self, data):
        self.seq = data.header.seq
        self.time = data.header.stamp
        self.frame_id = data.header.frame_id
        self._msg = self.bridge.imgmsg_to_cv2(data, "8UC3")
        #self._msg = ros_numpy.numpify(data)
        self._dep =self.bridge.imgmsg_to_cv2(self.depth, "16UC1")

        
        sha_y, sha_x, _ = self._msg.shape
        pad_img = np.zeros((sha_y * 2, sha_x * 2, 3), dtype=np.uint8)
        pad_img[int(sha_y*0.5):-int(sha_y*0.5), int(sha_x*0.5):-int(sha_x*0.5), :] = self._msg
        rgbImg = pad_img[int((sha_y*0.5)+cyhsr_van-240):int((sha_y*0.5)+cyhsr_van+240), int((sha_x*0.5)+cxhsr_van-320):int((sha_x*0.5)+cxhsr_van+320), :]
        #self._msg = cv2.resize(self._msg, (640, 480))
        
        self._msg = cv2.cvtColor(self._msg, cv2.COLOR_BGR2RGB)
        cv2.imwrite('/stefan/test.png', self._msg)
        #self._dep = self._dep[int(y_min):int(y_max), int(x_min):ihnt(x_max)]
        #self._dep = cv2.resize(self._dep, (640, 480))


        det_objs, det_poses, det_confs = run_estimation(self._msg, self._model, self.threeD_boxes)#, self.seq)

        self.publish_pose(det_objs, det_poses, det_confs)
        rospy.sleep(2)
    

    def publish_pose(self, det_names, det_poses, det_confidences):
        msg = PoseArray()
        msg.header.frame_id = '/head_rgbd_sensor_rgb_frame'
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
        # msg = get_posesResponse()
        # for idx in range(len(det_names)):
        #     item = PoseWithConfidence()
        #     item.name = det_names[idx] 
        #     item.confidence = det_confidences[idx]
        #     item.pose = Pose()
        #     det_pose = det_poses[idx]
        #     item.pose.position.x = det_pose[0]
        #     item.pose.position.y = det_pose[1]
        #     item.pose.position.z = det_pose[2]
        #     item.pose.orientation.w = det_pose[3]
        #     item.pose.orientation.x = det_pose[4]
        #     item.pose.orientation.y = det_pose[5]
        #     item.pose.orientation.z = det_pose[6]
        #     msg.poses.append(item)

        # self.pose_pub.publish(msg)


class PoseEstimationServer:
    def __init__(self, model, mesh_path, threshold, topic, service_name):
        #event that will block until the info is received
        #attribute for storing the rx'd message
        self._score_th = threshold

        self._msg = None
        self.seq = None
        self.time = None
        self.frame_id = None
        self.bridge = CvBridge()
        self.topic = topic
        self.pose_pub = rospy.Publisher("/pyrapose/poses", PoseArray, queue_size=10)
        self.pose_srv = rospy.Service(service_name, get_poses, self.callback)
        self.image_sub = rospy.Subscriber(topic, Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/image_raw', Image, self.depth_callback)

        self.viz_pub = rospy.Publisher("/pyrapose/visualize", Image, queue_size=10)


        self.threeD_boxes = np.ndarray((1, 8, 3), dtype=np.float32)
        self.sphere_diameters = np.ndarray((1), dtype=np.float32)
        self.num_classes = 0
        mesh_info = os.path.join(mesh_path, 'models_info.json')
        for key, value in json.load(open(mesh_info)).items():
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
            self.num_classes += 1

        self._model = model = load_model(model, self.sphere_diameters, self.num_classes)
    
    def image_callback(self, data):
        self.image = data

    def depth_callback(self, data):
        self.depth = data

    def callback(self, req):
        print("Received request, waiting for image")
        rospy.wait_for_message(self.topic, Image)
        print("Received image")
        data = self.image
        self.seq = data.header.seq
        self.time = data.header.stamp
        self.frame_id = data.header.frame_id
        self._msg = self.bridge.imgmsg_to_cv2(data, "8UC3")
        #self._msg = ros_numpy.numpify(data)
        self._dep =self.bridge.imgmsg_to_cv2(self.depth, "16UC1")

        sha_y, sha_x, _ = self._msg.shape
        pad_img = np.zeros((sha_y * 2, sha_x * 2, 3), dtype=np.uint8)
        pad_img[int(sha_y*0.5):-int(sha_y*0.5), int(sha_x*0.5):-int(sha_x*0.5), :] = self._msg
        rgbImg = pad_img[int((sha_y*0.5)+cyhsr_van-240):int((sha_y*0.5)+cyhsr_van+240), int((sha_x*0.5)+cxhsr_van-320):int((sha_x*0.5)+cxhsr_van+320), :]
        #self._msg = cv2.resize(self._msg, (640, 480))
        
        self._msg = cv2.cvtColor(self._msg, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('/stefan/test.png', self._msg)
        #self._dep = self._dep[int(y_min):int(y_max), int(x_min):int(x_max)]
        #self._dep = cv2.resize(self._dep, (640, 480))


        det_objs, det_poses, det_confs, viz_img = run_estimation(self._msg, self._model, self.threeD_boxes)#, self.seq)
        msg = self.fill_pose(det_objs, det_poses, det_confs)
        self.viz_pose(viz_img)
        return msg

    def fill_pose(self, det_names, det_poses, det_confidences):
        msg = PoseArray()
        msg.header.frame_id = '/head_rgbd_sensor_rgb_frame'
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
        msg.header.frame_id = '/head_rgbd_sensor_rgb_frame'
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


def run_estimation(image, model, threeD_boxes):
    obj_names = []
    obj_poses = []
    obj_confs = []

    image_raw = copy.deepcopy(image)
    image = preprocess_image(image)
    #image_mask = copy.deepcopy(image)

    scores, labels, poses, mask = model.predict_on_batch(np.expand_dims(image, axis=0))

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
        est3D = toPix_array(eDbox, fxhsr, fyhsr, cxhsr, cyhsr)
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
        obj_names.append('transparent_canister')
        obj_confs.append(score)

    return obj_names, obj_poses, obj_confs, image_raw


if __name__ == '__main__':

    # ROS params
    mesh_path = ''
    msg_topic = '/camera/rgb/image_color'
    score_threshold = 0.5
    icp_threshold = 0.15
    service_name = 'get_poses'
    try:
        model_path = rospy.get_param('/PyraPose/model_path')
    except KeyError:
        print("please set path to model! example:/home/desired/path/to/resnet_xy.h5")
    try:
        mesh_path = rospy.get_param('/PyraPose/meshes_path')
    except KeyError:
        print("please set path to meshes! example:/home/desired/path/to/meshes/")

    if rospy.has_param('/PyraPose/detection_threshold'):
        score_threshold = rospy.get_param("/PyraPose/detection_threshold")
        print('Detection threshold set to: ', score_threshold)
    if rospy.has_param('/PyraPose/image_topic'):
        msg_topic = rospy.get_param("/PyraPose/image_topic")
        print("Subscribing to msg topic: ", msg_topic)
    if rospy.has_param('/PyraPose/icp_threshold'):
        icp_threshold = rospy.get_param("/PyraPose/icp_threshold")
        print("icp threshold set to: ", icp_threshold)
    if rospy.has_param('/PyraPose/service_call'):
        service_name = rospy.get_param("/PyraPose/service_call")
        print("service call set to: ", service_name)

    #model, graph = load_model(model_path)
    #model = load_model(model_path)
    try:
        if rospy.get_param('/PyraPose/node_type') == 'continuous':
            print("node type set to continuous")
            pose_estimation = PoseEstimationClass(model_path, mesh_path, score_threshold, msg_topic)#, graph)
        elif rospy.get_param('/PyraPose/node_type') == 'service':
            print("node type set to service")
            pose_estimation = PoseEstimationServer(model_path, mesh_path, score_threshold, msg_topic, service_name)
    except KeyError:
        print("node_type should either be continuous or service.")
    rospy.init_node('PyraPose', anonymous=True)

    rospy.spin()






