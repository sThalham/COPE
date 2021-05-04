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

import keras
import tensorflow as tf
import open3d
import ros_numpy

#print(sys.path)
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
from PIL import Image as Pilimage

sys.path.append("/RGBDPose")
from RGBDPose import models
from RGBDPose.utils.config import read_config_file, parse_anchor_parameters
from RGBDPose.utils.eval import evaluate
from RGBDPose.utils.keras_version import check_keras_version
from RGBDPose.utils import ply_loader

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
# fxkin = 538.391033
# fykin = 538.085452
# cxkin = 315.30747
# cykin = 233.048356

# magic intrinsics
fxkin = 1066.778
fykin = 1067.487
cxkin = 320.0
cykin = 240.0



def get_evaluation_kiru(pcd_temp_,pcd_scene_,inlier_thres,tf,final_th, model_dia):#queue
    tf_pcd =np.eye(4)
    pcd_temp_.transform(tf)

    mean_temp = np.mean(np.array(pcd_temp_.points)[:, 2])
    mean_scene = np.median(np.array(pcd_scene_.points)[:, 2])
    pcd_diff = mean_scene - mean_temp

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
    open3d.estimate_normals(pcd_scene_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=5.0, max_nn=10))

    reg_p2p = open3d.registration.registration_icp(pcd_temp_,pcd_scene_ , inlier_thres, np.eye(4),
                                                   open3d.registration.TransformationEstimationPointToPoint(),
                                                   open3d.registration.ICPConvergenceCriteria(max_iteration = 5)) #5?
    tf = np.matmul(reg_p2p.transformation,tf)
    tf_pcd = np.matmul(reg_p2p.transformation,tf_pcd)
    pcd_temp_.transform(reg_p2p.transformation)

    #open3d.estimate_normals(pcd_temp_, search_param=open3d.KDTreeSearchParamHybrid(
    #    radius=2.0, max_nn=30))
    points_unfiltered = np.asarray(pcd_temp_.points)
    last_pcd_temp = []
    for i, normal in enumerate(pcd_temp_.normals):
        if normal[2] < 0:
            last_pcd_temp.append(points_unfiltered[i, :])

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



#################################
############### ROS #############
#################################
class PoseEstimationClass:
    #def __init__(self, model, mesh_path, threshold, topic, graph):
    def __init__(self, model, mesh_path, threshold, topic):
        #event that will block until the info is received
        #attribute for storing the rx'd message
        self._model = model
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

        self.threeD_boxes = np.ndarray((22, 8, 3), dtype=np.float32)
        mesh_info = os.path.join(mesh_path, 'models_info.json')
        for key, value in json.load(open(mesh_info)).items():
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
            self.threeD_boxes[int(key), :, :] = three_box_solo
        ply_path = os.path.join(mesh_path, 'obj_000005.ply')
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_6 = open3d.PointCloud()
        self.model_6.points = open3d.Vector3dVector(model_vsd['pts'])
        self.pcd_model_6 = open3d.PointCloud()
        self.pcd_model_6.points = open3d.Vector3dVector(model_vsd['pts'])
        open3d.estimate_normals(self.pcd_model_6, search_param=open3d.KDTreeSearchParamHybrid(
        radius=2.0, max_nn=30))
        ply_path = mesh_path + '/obj_000008.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_9 = open3d.PointCloud()
        self.model_9.points = open3d.Vector3dVector(model_vsd['pts'])
        self.pcd_model_9 = open3d.PointCloud()
        self.pcd_model_9.points = open3d.Vector3dVector(model_vsd['pts'])
        open3d.estimate_normals(self.pcd_model_9, search_param=open3d.KDTreeSearchParamHybrid(
        radius=2.0, max_nn=30))
        ply_path = mesh_path + '/obj_000009.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_10 = open3d.PointCloud()
        self.model_10.points = open3d.Vector3dVector(model_vsd['pts'])
        self.pcd_model_10 = open3d.PointCloud()
        self.pcd_model_10.points = open3d.Vector3dVector(model_vsd['pts'])
        open3d.estimate_normals(self.pcd_model_10, search_param=open3d.KDTreeSearchParamHybrid(
        radius=2.0, max_nn=30))
        ply_path = mesh_path + '/obj_000010.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_11 = open3d.PointCloud()
        self.model_11.points = open3d.Vector3dVector(model_vsd['pts'])
        self.pcd_model_11 = open3d.PointCloud()
        self.pcd_model_11.points = open3d.Vector3dVector(model_vsd['pts'])
        open3d.estimate_normals(self.pcd_model_11, search_param=open3d.KDTreeSearchParamHybrid(
        radius=2.0, max_nn=30))
        ply_path = mesh_path + '/obj_000021.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_61 = open3d.PointCloud()
        self.model_61.points = open3d.Vector3dVector(model_vsd['pts'])
        self.pcd_model_61 = open3d.PointCloud()
        self.pcd_model_61.points = open3d.Vector3dVector(model_vsd['pts'])
        open3d.estimate_normals(self.pcd_model_61, search_param=open3d.KDTreeSearchParamHybrid(
        radius=2.0, max_nn=30))
    def depth_callback(self, data):
        self.depth = data

    def callback(self, data):
        self.seq = data.header.seq
        self.time = data.header.stamp
        self.frame_id = data.header.frame_id
        self._msg = self.bridge.imgmsg_to_cv2(data, "8UC3")
        #self._msg = ros_numpy.numpify(data)
        self._dep =self.bridge.imgmsg_to_cv2(self.depth, "16UC1")

        
        f_sca_x = 538.391033 / 1066.778
        f_sca_y = 538.085452 / 1067.487
        x_min = 315.30747 * f_sca_x
        x_max = 315.30747 + (640.0 - 315.30747) * f_sca_x
        y_min = 233.04356 * f_sca_y
        y_max = 233.04356 + (480.0 - 233.04356) * f_sca_y
        self._msg = self._msg[int(y_min):int(y_max), int(x_min):int(x_max), :]
        self._msg = cv2.resize(self._msg, (640, 480))
        
        self._msg = cv2.cvtColor(self._msg, cv2.COLOR_BGR2RGB)
        cv2.imwrite('/stefan/test.png', self._msg)
        self._dep = self._dep[int(y_min):int(y_max), int(x_min):int(x_max)]
        self._dep = cv2.resize(self._dep, (640, 480))


        det_objs, det_poses, det_confs = run_estimation(self._msg, self._dep, self._model, self._score_th, self.threeD_boxes, self.pcd_model_6, self.pcd_model_9, self.pcd_model_10, self.pcd_model_11, self.pcd_model_61)#, self.seq)

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
        self._model = model
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


        self.threeD_boxes = np.ndarray((22, 8, 3), dtype=np.float32)
        mesh_info = os.path.join(mesh_path, 'models_info.json')
        for key, value in json.load(open(mesh_info)).items():
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
            self.threeD_boxes[int(key), :, :] = three_box_solo
        ply_path = os.path.join(mesh_path, 'obj_000005.ply')
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_6 = open3d.PointCloud()
        self.model_6.points = open3d.Vector3dVector(model_vsd['pts'])
        self.pcd_model_6 = open3d.PointCloud()
        self.pcd_model_6.points = open3d.Vector3dVector(model_vsd['pts'])
        open3d.estimate_normals(self.pcd_model_6, search_param=open3d.KDTreeSearchParamHybrid(
        radius=2.0, max_nn=30))
        ply_path = mesh_path + '/obj_000008.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_9 = open3d.PointCloud()
        self.model_9.points = open3d.Vector3dVector(model_vsd['pts'])
        self.pcd_model_9 = open3d.PointCloud()
        self.pcd_model_9.points = open3d.Vector3dVector(model_vsd['pts'])
        open3d.estimate_normals(self.pcd_model_9, search_param=open3d.KDTreeSearchParamHybrid(
        radius=2.0, max_nn=30))
        ply_path = mesh_path + '/obj_000009.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_10 = open3d.PointCloud()
        self.model_10.points = open3d.Vector3dVector(model_vsd['pts'])
        self.pcd_model_10 = open3d.PointCloud()
        self.pcd_model_10.points = open3d.Vector3dVector(model_vsd['pts'])
        open3d.estimate_normals(self.pcd_model_10, search_param=open3d.KDTreeSearchParamHybrid(
        radius=2.0, max_nn=30))
        ply_path = mesh_path + '/obj_000010.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_11 = open3d.PointCloud()
        self.model_11.points = open3d.Vector3dVector(model_vsd['pts'])
        self.pcd_model_11 = open3d.PointCloud()
        self.pcd_model_11.points = open3d.Vector3dVector(model_vsd['pts'])
        open3d.estimate_normals(self.pcd_model_11, search_param=open3d.KDTreeSearchParamHybrid(
        radius=2.0, max_nn=30))
        ply_path = mesh_path + '/obj_000021.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_61 = open3d.PointCloud()
        self.model_61.points = open3d.Vector3dVector(model_vsd['pts'])
        self.pcd_model_61 = open3d.PointCloud()
        self.pcd_model_61.points = open3d.Vector3dVector(model_vsd['pts'])
        open3d.estimate_normals(self.pcd_model_61, search_param=open3d.KDTreeSearchParamHybrid(
        radius=2.0, max_nn=30))
    
    def image_callback(self, data):
        self.image = data

    def depth_callback(self, data):
        self.depth = data

    def callback(self, req):
        #print(data)
        rospy.wait_for_message(self.topic, Image)
        data = self.image
        self.seq = data.header.seq
        self.time = data.header.stamp
        self.frame_id = data.header.frame_id
        self._msg = self.bridge.imgmsg_to_cv2(data, "8UC3")
        #self._msg = ros_numpy.numpify(data)
        self._dep =self.bridge.imgmsg_to_cv2(self.depth, "16UC1")

        
        f_sca_x = 538.391033 / 1066.778
        f_sca_y = 538.085452 / 1067.487
        x_min = 315.30747 * f_sca_x
        x_max = 315.30747 + (640.0 - 315.30747) * f_sca_x
        y_min = 233.04356 * f_sca_y
        y_max = 233.04356 + (480.0 - 233.04356) * f_sca_y
        self._msg = self._msg[int(y_min):int(y_max), int(x_min):int(x_max), :]
        self._msg = cv2.resize(self._msg, (640, 480))
        
        self._msg = cv2.cvtColor(self._msg, cv2.COLOR_BGR2RGB)
        cv2.imwrite('/stefan/test.png', self._msg)
        self._dep = self._dep[int(y_min):int(y_max), int(x_min):int(x_max)]
        self._dep = cv2.resize(self._dep, (640, 480))

        det_objs, det_poses, det_confs = run_estimation(self._msg, self._dep, self._model, self._score_th, self.threeD_boxes, self.pcd_model_6, self.pcd_model_9, self.pcd_model_10, self.pcd_model_11, self.pcd_model_61)#, self.seq)

        msg = self.fill_pose(det_objs, det_poses, det_confs)
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


def load_model(model_path):

    check_keras_version()

    #if args.gpu:
    #    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #keras.backend.tensorflow_backend.set_session(get_session())

    anchor_params = None
    backbone = 'resnet50'

    print('Loading model, this may take a second...')
    print(model_path)
    model = models.load_model(model_path, backbone_name=backbone)
    #graph = tf.compat.v1.get_default_graph()
    model = models.convert_model(model, anchor_params=anchor_params) # convert model

    # print model summary
    print(model.summary())

    return model#, graph

mask_pub = rospy.Publisher('/pyrapose/masks', Image, queue_size=10)
#def run_estimation(image, model, score_threshold, graph, frame_id):
def run_estimation(image, image_dep, model, score_threshold, threeD_boxes, model_6, model_9, model_10, model_11, model_61):
    obj_names = []
    obj_poses = []
    obj_confs = []

    image_mask = copy.deepcopy(image)
    image = preprocess_image(image)
    #image_mask = copy.deepcopy(image)

    #cv2.imwrite('/home/sthalham/retnetpose_image.jpg', image)

    if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

    #with graph.as_default():
    boxes3D, scores, mask = model.predict_on_batch(np.expand_dims(image, axis=0))

    for inv_cls in range(scores.shape[2]):

        if inv_cls == 0:
            true_cls = 5
        elif inv_cls == 1:
            true_cls = 8
        elif inv_cls == 2:
            true_cls = 9
        elif inv_cls == 3:
            true_cls = 10
        elif inv_cls == 4:
            true_cls = 21
        #true_cat = inv_cls + 1
        #true_cls = true_cat

        cls_mask = scores[0, :, inv_cls]

        cls_indices = np.where(cls_mask > score_threshold)
        #cls_indices = np.argmax(cls_mask)
        #print(cls_mask[cls_indices])

        cls_img = image
        obj_mask = mask[0, :, inv_cls]
        if inv_cls == 0:
            obj_col = [1, 255, 255]
        elif inv_cls == 1:
            obj_col = [1, 1, 128]
        elif inv_cls == 2:
            obj_col = [255, 255, 1]
        elif inv_cls == 3:
            obj_col = [220, 245, 245]
        elif inv_cls == 4:
            obj_col = [128, 1, 1]
        cls_img = np.where(obj_mask > 0.5, 1, 0)
        cls_img = cls_img.reshape((60, 80)).astype(np.uint8)
        cls_img = np.asarray(Pilimage.fromarray(cls_img).resize((640, 480), Pilimage.NEAREST))
        depth_mask = copy.deepcopy(cls_img)
        cls_img = np.repeat(cls_img[:, :, np.newaxis], 3, 2)
        cls_img = cls_img.astype(np.uint8)
        cls_img[:, :, 0] *= obj_col[0]
        cls_img[:, :, 1] *= obj_col[1]
        cls_img[:, :, 2] *= obj_col[2]
        image_mask = np.where(cls_img > 0, cls_img, image_mask.astype(np.uint8))
        #cv2.imwrite('/stefan/mask.png', image_mask)

        #if len(cls_indices[0]) < 1:
        if len(cls_indices[0]) < 1:
            continue

        if true_cls == 5:
            name = '006_mustard_bottle'
            pcd_model = model_6
        elif true_cls == 8:
            name = '009_gelatin_box'
            pcd_model = model_9
        elif true_cls == 9:
            name = '010_potted_meat_can'
            pcd_model = model_10
        elif true_cls == 10:
            name = '011_banana'
            pcd_model = model_11
        elif true_cls == 21:
            name = '061_foam_brick'
            pcd_model = model_61
        else:
            continue 

        obj_names.append(name)
        #obj_confs.append(np.sum(cls_mask[cls_indices[0]]))
        obj_confs.append(np.sum(cls_mask[cls_indices]))


        k_hyp = len(cls_indices[0])
        #k_hyp = 1
        ori_points = np.ascontiguousarray(threeD_boxes[(true_cls), :, :], dtype=np.float32)  # .reshape((8, 1, 3))
        K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

        ##############################
        # pnp
        pose_votes = boxes3D[0, cls_indices, :]
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
        t_est = otvec[:, 0]
              
        
        if np.sum(depth_mask) > 3000 :

            print('--------------------- ICP refinement -------------------')

            print('cls: ', true_cls)
            pcd_img = np.where(depth_mask, image_dep, np.NaN)
            pcd_img = create_point_cloud(pcd_img, fxkin, fykin, cxkin, cykin, 1.0)
            pcd_img = pcd_img[~np.isnan(pcd_img).any(axis=1)]
            pcd_crop = open3d.PointCloud()
            pcd_crop.points = open3d.Vector3dVector(pcd_img)
            open3d.estimate_normals(pcd_crop, search_param=open3d.KDTreeSearchParamHybrid(radius=20.0, max_nn=30))

            guess = np.zeros((4, 4), dtype=np.float32)
            guess[:3, :3] = R_est
            guess[:3, 3] = t_est.T * 1000.0
            guess[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).T

            pcd_model = open3d.geometry.voxel_down_sample(pcd_model, voxel_size=5.0)
            pcd_crop = open3d.geometry.voxel_down_sample(pcd_crop, voxel_size=5.0)
            open3d.estimate_normals(pcd_crop, search_param=open3d.KDTreeSearchParamHybrid(radius=10.0, max_nn=10))
            open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(radius=10.0, max_nn=10))

            pcd_model.transform(guess)

            # remove model vertices facing away from camera
            points_unfiltered = np.asarray(pcd_model.points)
            last_pcd_temp = []
            for i, normal in enumerate(pcd_model.normals):
                if normal[2] < 0:
                    last_pcd_temp.append(points_unfiltered[i, :])

            pcd_model.points = open3d.Vector3dVector(np.asarray(last_pcd_temp))
            open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
                    radius=20.0, max_nn=10))

            # align model with median depth of scene
            mean_crop = np.median(np.array(pcd_crop.points), axis=0)
            mean_model = np.mean(np.array(pcd_model.points), axis=0)
            pcd_crop_filt = copy.deepcopy(pcd_crop)

            pcd_min = guess[2, 3] - 75
            pcd_max = guess[2, 3] + 75
            new_pcd_scene = []
            for i, point in enumerate(pcd_crop.points):
                if point[2] > pcd_min and point[2] < pcd_max:
                    new_pcd_scene.append(point)

            #mean_crop = np.mean(np.array(pcd_crop_filt.points), axis=0)

            print(mean_crop, mean_crop.shape)
            #print(guess[:3, 3], guess[:3, 3].shape)
            #print(mean_crop-guess[:3, 3])
            print('euclid: ', np.linalg.norm((mean_crop-guess[:3, 3]), ord=2))
            print('num_points: ', len(new_pcd_scene))
            if len(new_pcd_scene)< 50 or np.linalg.norm((mean_crop-guess[:3, 3]), ord=2) > 75:
                print('use pcd mean')
                if len(new_pcd_scene) > 50 and np.linalg.norm((mean_crop-guess[:3, 3]), ord=2) > 75:
                    print('recalc mean')
                    pcd_crop_filt.points = open3d.Vector3dVector(np.asarray(new_pcd_scene))
                    mean_crop = np.mean(np.array(pcd_crop_filt.points), axis=0)
                pcd_diff = mean_crop - mean_model
                pcd_model.translate(pcd_diff)
                open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
                    radius=10.0, max_nn=10))
                guess[:3, 3] = mean_crop
            else:
                print('use pose')
                pcd_crop.points = open3d.Vector3dVector(np.asarray(new_pcd_scene))

            open3d.estimate_normals(pcd_crop, search_param=open3d.KDTreeSearchParamHybrid(
                radius=20.0, max_nn=10))

            #reg_p2p = open3d.registration.registration_icp(pcd_model, pcd_crop, 5.0, np.eye(4),
            #                                                open3d.registration.TransformationEstimationPointToPlane(), open3d.registration.ICPConvergenceCriteria(max_iteration=100))
            reg_icp = cv2.ppf_match_3d_ICP(100, tolerence=0.075, numLevels=4)
            model_points = np.asarray(pcd_model.points, dtype=np.float32)
            model_normals = np.asarray(pcd_model.normals, dtype=np.float32)
            crop_points = np.asarray(pcd_crop.points, dtype=np.float32)
            crop_normals = np.asarray(pcd_crop.normals, dtype=np.float32)
            pcd_source = np.zeros((model_points.shape[0], 6), dtype=np.float32)
            pcd_target = np.zeros((crop_points.shape[0], 6), dtype=np.float32)
            pcd_source[:, :3] = model_points * 0.001
            pcd_source[:, 3:] = model_normals
            pcd_target[:, :3] = crop_points * 0.001
            pcd_target[:, 3:] = crop_normals

            retval, residual, pose = reg_icp.registerModelToScene(pcd_source, pcd_target)
            
            print('residual: ', residual)
            #pcd_model.transform(reg_p2p.transformation)
            guess[:3, 3] = guess[:3, 3] * 0.001
            guess = np.matmul(pose, guess)
            R_est = guess[:3, :3]
            t_est = guess[:3, 3] 

            #print('guess: ', guess)
        

        est_pose = np.zeros((7), dtype=np.float32)
        est_pose[:3] = t_est
        est_pose[3:] = tf3d.quaternions.mat2quat(R_est)
        obj_poses.append(est_pose)

    bridge = CvBridge()

    image_mask_msg = bridge.cv2_to_imgmsg(image_mask)

    mask_pub.publish(image_mask_msg)
    return obj_names, obj_poses, obj_confs


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
    model = load_model(model_path)
    try:
        if rospy.get_param('/PyraPose/node_type') == 'continuous':
            print("node type set to continuous")
            pose_estimation = PoseEstimationClass(model, mesh_path, score_threshold, msg_topic)#, graph)
        elif rospy.get_param('/PyraPose/node_type') == 'service':
            print("node type set to service")
            pose_estimation = PoseEstimationServer(model, mesh_path, score_threshold, msg_topic, service_name)
    except KeyError:
        print("node_type should either be continuous or service.")
    rospy.init_node('PyraPose', anonymous=True)

    rospy.spin()






