

import numpy as np
import json
import transforms3d as tf3d
import copy
import cv2
import open3d
from ..utils import ply_loader
from .pose_error import reproj, add, adi, re, te, vsd
from PIL import Image
import imgaug.augmenters as iaa

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


#fxkin = 1066.778
#fykin = 1067.487
#cxkin = 312.9869
#cykin = 241.3109

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


def toPix_array(translation):

    xpix = ((translation[:, 0] * fxkin) / translation[:, 2]) + cxkin
    ypix = ((translation[:, 1] * fykin) / translation[:, 2]) + cykin
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


def load_pcd(cat):
    # load meshes
    #mesh_path = "/home/sthalham/data/LINEMOD/models/"
    mesh_path = "/home/stefan/data/Meshes/ycb_video/models/"
    template = '000000'
    lencat = len(cat)
    cat = template[:-lencat] + cat
    ply_path = mesh_path + 'obj_' + cat + '.ply'
    model_vsd = ply_loader.load_ply(ply_path)
    pcd_model = open3d.PointCloud()
    pcd_model.points = open3d.Vector3dVector(model_vsd['pts'])
    open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
        radius=20.0, max_nn=30))
    # open3d.draw_geometries([pcd_model])
    model_vsd_mm = copy.deepcopy(model_vsd)
    model_vsd_mm['pts'] = model_vsd_mm['pts'] * 1000.0
    pcd_model = open3d.read_point_cloud(ply_path)

    return pcd_model, model_vsd, model_vsd_mm


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


def evaluate_ycbv(generator, model, threshold=0.05):
    threshold = 0.5

    mesh_info = '/home/stefan/data/Meshes/ycb_video/models/models_info.json'

    threeD_boxes = np.ndarray((22, 8, 3), dtype=np.float32)
    sym_cont = np.ndarray((22, 3), dtype=np.float32)
    sym_disc = np.ndarray((28, 4, 4), dtype=np.float32)
    model_dia = np.zeros((22), dtype=np.float32)

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
        threeD_boxes[int(key), :, :] = three_box_solo
        model_dia[int(key)] = value['diameter'] * fac

    # start collecting results
    results = []
    image_ids = []
    image_indices = []
    idx = 0

    tp = np.zeros((22), dtype=np.uint32)
    fp = np.zeros((22), dtype=np.uint32)
    fn = np.zeros((22), dtype=np.uint32)

    # interlude end

    tp_add = np.zeros((22), dtype=np.uint32)
    fp_add = np.zeros((22), dtype=np.uint32)
    fn_add = np.zeros((22), dtype=np.uint32)

    rotD = np.zeros((22), dtype=np.uint32)
    less5 = np.zeros((22), dtype=np.uint32)
    rep_e = np.zeros((22), dtype=np.uint32)
    rep_less5 = np.zeros((22), dtype=np.uint32)
    add_e = np.zeros((22), dtype=np.uint32)
    add_less_d = np.zeros((22), dtype=np.uint32)
    vsd_e = np.zeros((22), dtype=np.uint32)
    vsd_less_t = np.zeros((22), dtype=np.uint32)
    
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

    allPoses = np.zeros((22), dtype=np.uint32)
    trueDets = np.zeros((22), dtype=np.uint32)
    falseDets = np.zeros((22), dtype=np.uint32)
    truePoses = np.zeros((22), dtype=np.uint32)
    falsePoses = np.zeros((22), dtype=np.uint32)

    seq = iaa.Sequential([
        # blur
        # iaa.SomeOf((0, 2), [
        #    iaa.GaussianBlur((0.0, 0.5)),
        # ]),
        # brightness
        iaa.OneOf([
            iaa.Sequential([
                # foam
                iaa.Add((-9, -5)),
            ]),
        ]),
    ], random_order=True)
    image_raw = cv2.imread('/home/stefan/ICRA21_paper/grasping/videos_raw/test.png', 1)
    shiftx = np.random.randint(10)
    shifty = np.random.randint(10)
    ymin = shifty
    ymax= 479 - (10 - shifty)
    xmin = shiftx
    xmax = 639 - (10 - shifty)

    image_raw = seq.augment_image(image_raw)
    img_est = False
    boxes3D = None
    scores = None
    mask = None

    for index in progressbar.progressbar(range(generator.size()), prefix='LineMOD evaluation: '):

        # image_raw = generator.load_image(index)
        # image_raw = np.asarray(Image.open('/home/stefan/ICRA21_paper/videos_raw/test.png').convert('RGB'))

        #print(np.mean(np.reshape(image_iaa, (307200, 3)), axis=0))
        # image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        image = generator.preprocess_image(image_raw)
        image, scale = generator.resize_image(image)
        # print(pose_votes.shape)
        image_raw_dep = generator.load_image_dep(index)
        image_dep = np.where(image_raw_dep > 0, image_raw_dep, np.NaN)

        vid_path = '/home/stefan/ICRA21_paper/videos_mobile/jello_grasp/frames/'
        mobile_path = vid_path + 'frame' + str(index) + '.jpg'
        image_mob = cv2.imread(mobile_path, 1)

        if img_est == False:
            boxes3D, scores, mask = model.predict_on_batch(np.expand_dims(image, axis=0))  # , np.expand_dims(image_dep, axis=0)])
            img_est = True
        else:
            pass

        image = image_raw
        image_mask = copy.deepcopy(image_raw)
        image_pose = copy.deepcopy(image_raw)

        for inv_cls in range(scores.shape[2]):

            # cls = inv_cls + 1

            if inv_cls == 0:
                true_cat = 5
            elif inv_cls == 1:
                true_cat = 8
            elif inv_cls == 2:
                true_cat = 9
            elif inv_cls == 3:
                true_cat = 10
            elif inv_cls == 4:
                true_cat = 21
            cls = true_cat

            cls_mask = scores[0, :, inv_cls]
            cls_indices = np.where(cls_mask > threshold)

            obj_mask = mask[0, :, inv_cls]
            # print(np.nanmax(obj_mask))
            if inv_cls == 0:
                obj_col = [128, 1, 1]
            elif inv_cls == 1:
                obj_col = [255, 255, 1]
            elif inv_cls == 2:
                obj_col = [1, 1, 128]
            elif inv_cls == 3:
                obj_col = [220, 245, 245]
            elif inv_cls == 4:
                obj_col = [1, 255, 255]
            cls_img = np.where(obj_mask > 0.5, 1, 0)
            cls_img = cls_img.reshape((60, 80)).astype(np.uint8)
            cls_img = np.asarray(Image.fromarray(cls_img).resize((640, 480), Image.NEAREST))
            depth_mask = copy.deepcopy(cls_img)
            cls_img = np.repeat(cls_img[:, :, np.newaxis], 3, 2)
            cls_img = cls_img.astype(np.uint8)
            cls_img[:, :, 0] *= obj_col[0]
            cls_img[:, :, 1] *= obj_col[1]
            cls_img[:, :, 2] *= obj_col[2]
            image_mask = np.where(cls_img > 0, cls_img, image_mask)

            if cls == 5:
                model_vsd = mv5
                pcd_model = pc5
            elif cls == 8:
                model_vsd = mv8
                pcd_model = pc8
            elif cls == 9:
                model_vsd = mv9
                pcd_model = pc9
            elif cls == 10:
                model_vsd = mv10
                pcd_model = pc10
            elif cls == 21:
                model_vsd = mv21
                pcd_model = pc21

            k_hyp = len(cls_indices[0])
            ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
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
            R_raw, _ = cv2.Rodrigues(orvec)
            t_raw = otvec

            guess = np.zeros((4, 4), dtype=np.float32)
            guess[:3, :3] = R_raw
            guess[:3, 3] = t_raw.T * 1000.0
            guess[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).T

            pcd_now = copy.deepcopy(pcd_model)
            pcd_now.transform(guess)
            cloud_points = np.asarray(pcd_now.points)
            model_image = toPix_array(cloud_points)
            for idx in range(model_image.shape[0]):
                if int(model_image[idx, 1]) > 479 or int(model_image[idx, 0]) > 639 or int(model_image[idx, 1]) < 0 or int(model_image[idx, 0]) < 0:
                    continue
                image_pose[int(model_image[idx, 1]), int(model_image[idx, 0]), :] = obj_col

            ##############################
            '''
            print(np.sum(depth_mask))
            if np.sum(depth_mask) > 1000:
                print('--------------------- ICP refinement -------------------')
                cv2.imwrite('/home/stefan/RGBDPose_viz/pred_mask_' + str(index) + '_.jpg', image_mask)

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

                pcd_model = open3d.geometry.voxel_down_sample(pcd_model, voxel_size=10.0)
                pcd_crop = open3d.geometry.voxel_down_sample(pcd_crop, voxel_size=10.0)
                open3d.estimate_normals(pcd_crop, search_param=open3d.KDTreeSearchParamHybrid(radius=20.0, max_nn=10))
                open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(radius=20.0, max_nn=10))

                pcd_model.transform(guess)

                # print('model unfiltered: ', pcd_model)
                pcd_crop.paint_uniform_color(np.array([0.99, 0.0, 0.00]))
                pcd_model.paint_uniform_color(np.array([0.00, 0.99, 0.00]))

                # remove model vertices facing away from camera
                points_unfiltered = np.asarray(pcd_model.points)
                last_pcd_temp = []
                for i, normal in enumerate(pcd_model.normals):
                    if normal[2] < 0:
                        last_pcd_temp.append(points_unfiltered[i, :])

                pcd_model.points = open3d.Vector3dVector(np.asarray(last_pcd_temp))

                # open3d.draw_geometries([pcd_crop, pcd_model])

                # align translation
                mean_crop = np.mean(np.array(pcd_crop.points), axis=0)
                mean_model = np.median(np.array(pcd_model.points), axis=0)
                pcd_diff = mean_crop - mean_model

                print('pcd_diff: ', pcd_diff)

                # align model with median depth of scene
                # new_pcd_trans = []
                # for i, point in enumerate(pcd_model.points):
                #    poi = np.asarray(point)
                #    poi = poi + pcd_diff
                #    new_pcd_trans.append(poi)
                # pcd_model.points = open3d.Vector3dVector(np.asarray(new_pcd_trans))
                pcd_model.translate(pcd_diff)
                open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
                    radius=20.0, max_nn=10))
                open3d.draw_geometries([pcd_crop, pcd_model])
                guess[:3, 3] = guess[:3, 3] + pcd_diff

                reg_p2p = open3d.registration.registration_icp(pcd_model, pcd_crop, 1.0, np.eye(4),
                                                               open3d.registration.TransformationEstimationPointToPlane(),
                                                               open3d.registration.ICPConvergenceCriteria(
                                                                   max_iteration=100))

                print('icp: ', reg_p2p.transformation)
                pcd_model.transform(reg_p2p.transformation)
                guess = np.matmul(reg_p2p.transformation, guess)
                R_icp = guess[:3, :3]
                t_icp = guess[:3, 3] * 0.001
                t_icp = t_est[:, np.newaxis]

                print('guess: ', guess)
                open3d.draw_geometries([pcd_crop, pcd_model])
        

            eDbox = R_est.dot(ori_points.T).T
            # eDbox = eDbox + np.repeat(t_est[:, np.newaxis], 8, axis=1).T
            eDbox = eDbox + np.repeat(t_est, 8, axis=0)
            est3D = toPix_array(eDbox)
            eDbox = np.reshape(est3D, (16))
            pose = eDbox.astype(np.uint16)

            colEst = (255, 0, 0)

            image = cv2.line(image, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                             2)
            image = cv2.line(image, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                             2)
            image = cv2.line(image, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                             2)
            image = cv2.line(image, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                             2)

            '''
            if cls == 5:
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

                name = '/home/stefan/RGBDPose_viz/ycbvimg_' + str(index) + '.jpg'
                cv2.imwrite(name, image)
                print('break')

        image_mask = image_mask[ymin:ymax, xmin:xmax, :]
        image_pose = image_pose[ymin:ymax, xmin:xmax, :]

        img_vid = np.ones((900, 1600, 3), dtype=np.uint8) * 255 # * 192
        #spam
        image_mob_res = image_mob[50:950, 150:1150]
        #mustard
        image_mob_res = image_mob[50:950, 100:1100]
        #banana
        image_mob_res = image_mob[100:1000, 250:1250]
        image_mob_res = cv2.resize(image_mob_res, (995, 870))
        img_vid[15:885, 15:1010, :] = image_mob_res
        img_vid[465:885, 1025:1585, :] = cv2.resize(image_mask, (560, 420))
        img_vid[15:435, 1025:1585, :] = cv2.resize(image_pose, (560, 420))

        cv2.rectangle(img_vid, (int(15), int(15)), (int(1010), int(885)),
                      (0, 0, 0), 2)
        cv2.rectangle(img_vid, (int(1025), int(465)), (int(1585), int(885)),
                      (0, 0, 0), 2)
        cv2.rectangle(img_vid, (int(1025), int(15)), (int(1585), int(435)),
                      (0, 0, 0), 2)

        #name = '/home/stefan/RGBDPose_viz/img_' + str(index) + '.jpg'
        #cv2.imwrite(name, image)
        #name = '/home/stefan/ICRA21_paper/images_jello/pic_' + str(index) + '.jpg'
        #cv2.imwrite(name, img_vid)













        '''
        image_raw = generator.load_image(index)
        image = generator.preprocess_image(image_raw)
        image, scale = generator.resize_image(image)
            # print(pose_votes.shape)
        image_raw_dep = generator.load_image_dep(index)
        image_dep = np.where(image_raw_dep > 0, image_raw_dep, np.NaN)
        #image_raw_dep = np.multiply(image_raw_dep, 255.0 / 2000.0)
        #image_raw_dep = np.repeat(image_raw_dep[:, :, np.newaxis], 3, 2)
        # image_raw_dep = get_normal(image_raw_dep, fxkin, fykin, cxkin, cykin)
        #image_dep = generator.preprocess_image(image_raw_dep)
        #image_dep, scale = generator.resize_image(image_dep)

        anno = generator.load_annotations(index)

        if len(anno['labels']) < 1:
            continue

        desired_ann = False
        checkLab = anno['labels']  # +1 to real_class
        for idx, lab in enumerate(checkLab):
            allPoses[int(lab) + 1] += 1
            checkLab[idx] += 1
            if (lab+1) in [5, 8, 9, 10, 21]:
                desired_ann = True

        #if desired_ann == False:
        #    continue
        #print(checkLab)

        # run network
        images = []
        images.append(image)
        images.append(image_dep)
        boxes3D, scores, mask = model.predict_on_batch(np.expand_dims(image, axis=0))#, np.expand_dims(image_dep, axis=0)])

        image = image_raw
        image_mask = copy.deepcopy(image_raw)

        for idx, lab in enumerate(checkLab):
            t_tra = anno['poses'][idx][:3]
            t_rot = anno['poses'][idx][3:]
            t_rot = tf3d.quaternions.quat2mat(t_rot)
            R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
            t_gt = np.array(t_tra, dtype=np.float32)
            t_gt = t_gt * 0.001

            ori_points = np.ascontiguousarray(threeD_boxes[int(lab), :, :], dtype=np.float32)
            colGT = (0, 204, 0)
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

        for inv_cls in range(scores.shape[2]):

            #cls = inv_cls + 1

            if inv_cls == 0:
                true_cat = 5
            elif inv_cls == 1:
                true_cat = 8
            elif inv_cls == 2:
                true_cat = 9
            elif inv_cls == 3:
                true_cat = 10
            elif inv_cls == 4:
                true_cat = 21
            cls = true_cat

            cls_mask = scores[0, :, inv_cls]
            cls_indices = np.where(cls_mask > threshold)

            if true_cat not in checkLab:
                falsePoses[int(cls)] += 1
                continue

            if len(cls_indices[0]) < 1:
                # print('not enough inlier')
                continue
            trueDets[int(true_cat)] += 1
            print(np.nanmax(cls_mask), len(cls_indices[0]))

            #print('detection: ', true_cat)

            obj_mask = mask[0, :, inv_cls]
            #print(np.nanmax(obj_mask))
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
            cls_img = np.asarray(Image.fromarray(cls_img).resize((640, 480), Image.NEAREST))
            depth_mask = copy.deepcopy(cls_img)
            cls_img = np.repeat(cls_img[:, :, np.newaxis], 3, 2)
            cls_img = cls_img.astype(np.uint8)
            cls_img[:, :, 0] *= obj_col[0]
            cls_img[:, :, 1] *= obj_col[1]
            cls_img[:, :, 2] *= obj_col[2]
            image_mask = np.where(cls_img > 0, cls_img, image_mask)


            anno_ind = np.argwhere(anno['labels'] == true_cat)

            t_tra = anno['poses'][anno_ind[0][0]][:3]
            t_rot = anno['poses'][anno_ind[0][0]][3:]
            # print(t_rot)

            BOP_obj_id = np.asarray([true_cat], dtype=np.uint32)

            # print(cls)

            if cls == 1:
                model_vsd = mv1
                pcd_model = pc1
            elif cls == 2:
                model_vsd = mv2
                pcd_model = pc2
            elif cls == 3:
                model_vsd = mv3
                pcd_model = pc3
            elif cls == 4:
                model_vsd = mv4
                pcd_model = pc4
            elif cls == 5:
                model_vsd = mv5
                pcd_model = pc5
            elif cls == 6:
                model_vsd = mv6
                pcd_model = pc6
            elif cls == 7:
                model_vsd = mv7
                pcd_model = pc7
            elif cls == 8:
                model_vsd = mv8
                pcd_model = pc8
            elif cls == 9:
                model_vsd = mv9
                pcd_model = pc9
            elif cls == 10:
                model_vsd = mv10
                pcd_model = pc10
            elif cls == 11:
                model_vsd = mv11
                pcd_model = pc11
            elif cls == 12:
                model_vsd = mv12
                pcd_model = pc12
            elif cls == 13:
                model_vsd = mv13
                pcd_model = pc13
            elif cls == 14:
                model_vsd = mv14
                pcd_model = pc14
            elif cls == 15:
                model_vsd = mv15
                pcd_model = pc15
            elif cls == 16:
                model_vsd = mv16
                pcd_model = pc16
            elif cls == 17:
                model_vsd = mv17
                pcd_model = pc17
            elif cls == 18:
                model_vsd = mv18
                pcd_model = pc18
            elif cls == 19:
                model_vsd = mv19
                pcd_model = pc19
            elif cls == 20:
                model_vsd = mv20
                pcd_model = pc20
            elif cls == 21:
                model_vsd = mv21
                pcd_model = pc21

            k_hyp = len(cls_indices[0])
            ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
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
            t_est = otvec

            ##############################
            print(np.sum(depth_mask))
            if np.sum(depth_mask) > 1000:
                print('--------------------- ICP refinement -------------------')
                cv2.imwrite('/home/stefan/RGBDPose_viz/pred_mask_' + str(index) + '_.jpg', image_mask)

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

                pcd_model = open3d.geometry.voxel_down_sample(pcd_model, voxel_size=10.0)
                pcd_crop = open3d.geometry.voxel_down_sample(pcd_crop, voxel_size=10.0)
                open3d.estimate_normals(pcd_crop, search_param=open3d.KDTreeSearchParamHybrid(radius=20.0, max_nn=10))
                open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(radius=20.0, max_nn=10))

                pcd_model.transform(guess)

                # print('model unfiltered: ', pcd_model)
                pcd_crop.paint_uniform_color(np.array([0.99, 0.0, 0.00]))
                pcd_model.paint_uniform_color(np.array([0.00, 0.99, 0.00]))


                # remove model vertices facing away from camera
                points_unfiltered = np.asarray(pcd_model.points)
                last_pcd_temp = []
                for i, normal in enumerate(pcd_model.normals):
                    if normal[2] < 0:
                        last_pcd_temp.append(points_unfiltered[i, :])

                pcd_model.points = open3d.Vector3dVector(np.asarray(last_pcd_temp))

                #open3d.draw_geometries([pcd_crop, pcd_model])

                # align translation
                mean_crop = np.mean(np.array(pcd_crop.points), axis=0)
                mean_model = np.median(np.array(pcd_model.points), axis=0)
                pcd_diff = mean_crop - mean_model

                print('pcd_diff: ', pcd_diff)

                # align model with median depth of scene
                #new_pcd_trans = []
                #for i, point in enumerate(pcd_model.points):
                #    poi = np.asarray(point)
                #    poi = poi + pcd_diff
                #    new_pcd_trans.append(poi)
                #pcd_model.points = open3d.Vector3dVector(np.asarray(new_pcd_trans))
                pcd_model.translate(pcd_diff)
                open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
                    radius=20.0, max_nn=10))
                open3d.draw_geometries([pcd_crop, pcd_model])
                guess[:3, 3] = guess[:3, 3] + pcd_diff

                reg_p2p = open3d.registration.registration_icp(pcd_model, pcd_crop, 1.0, np.eye(4),
                                                               open3d.registration.TransformationEstimationPointToPlane(), open3d.registration.ICPConvergenceCriteria(max_iteration=100))

                print('icp: ', reg_p2p.transformation)
                pcd_model.transform(reg_p2p.transformation)
                guess = np.matmul(reg_p2p.transformation, guess)
                R_est = guess[:3, :3]
                t_est = guess[:3, 3] * 0.001
                t_est = t_est[:, np.newaxis]

                print('guess: ', guess)
                open3d.draw_geometries([pcd_crop, pcd_model])



                #reg_icp = cv2.ppf_match_3d_ICP(100, tolerence=0.005, numLevels=4)
                #model_points = np.asarray(pcd_model.points, dtype=np.float32)
                #model_normals = np.asarray(pcd_model.normals, dtype=np.float32)
                #crop_points = np.asarray(pcd_crop.points, dtype=np.float32)
                #crop_normals = np.asarray(pcd_crop.points, dtype=np.float32)
                #pcd_source = np.zeros((model_points.shape[0], 6), dtype=np.float32)
                #pcd_target = np.zeros((crop_points.shape[0], 6), dtype=np.float32)
                #pcd_source[:, :3] = model_points * 0.001
                #pcd_source[:, 3:] = model_normals
                #pcd_target[:, :3] = crop_points * 0.001
                #pcd_target[:, 3:] = crop_normals
                #retval, residual, pose = reg_icp.registerModelToScene(pcd_source, pcd_target)

            BOP_R = R_est.flatten().tolist()
            BOP_t = t_est.flatten().tolist()

            # result = [int(BOP_scene_id), int(BOP_im_id), int(BOP_obj_id), float(BOP_score), BOP_R[0], BOP_R[1], BOP_R[2], BOP_R[3], BOP_R[4], BOP_R[5], BOP_R[6], BOP_R[7], BOP_R[8], BOP_t[0], BOP_t[1], BOP_t[2]]
            # result = [int(BOP_scene_id), int(BOP_im_id), int(BOP_obj_id), float(BOP_score), BOP_R, BOP_t]
            # results_image.append(result)

            # t_rot = tf3d.euler.euler2mat(t_rot[0], t_rot[1], t_rot[2])
            t_rot = tf3d.quaternions.quat2mat(t_rot)
            R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
            t_gt = np.array(t_tra, dtype=np.float32)

            t_gt = t_gt * 0.001
            t_est = t_est.T
            # print('pose: ', pose)

            model_vsd["pts"] = model_vsd["pts"] * 0.001

            if cls == 1 or cls == 11:
                err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
            else:
                err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

            if err_add < model_dia[true_cat] * 0.1:
                truePoses[int(true_cat)] += 1

            print(' ')
            print('error: ', err_add, 'threshold', model_dia[cls] * 0.1)

            eDbox = R_est.dot(ori_points.T).T
            #eDbox = eDbox + np.repeat(t_est[:, np.newaxis], 8, axis=1).T
            eDbox = eDbox + np.repeat(t_est, 8, axis=0)
            est3D = toPix_array(eDbox)
            eDbox = np.reshape(est3D, (16))
            pose = eDbox.astype(np.uint16)

            colEst = (255, 0, 0)

            image = cv2.line(image, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
            image = cv2.line(image, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                             2)
            image = cv2.line(image, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                             2)
            image = cv2.line(image, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                             2)
            image = cv2.line(image, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                             2)

            
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
        

        #name = '/home/stefan/RGBDPose_viz/img_' + str(index) + '.jpg'
        #cv2.imwrite(name, image)
        name = '/home/stefan/RGBDPose_viz/mask.jpg'
        cv2.imwrite(name, image_mask)

        print('break')

    recall = np.zeros((22), dtype=np.float32)
    precision = np.zeros((22), dtype=np.float32)
    detections = np.zeros((22), dtype=np.float32)
    for i in range(1, (allPoses.shape[0])):
        recall[i] = truePoses[i] / allPoses[i]
        precision[i] = truePoses[i] / (truePoses[i] + falsePoses[i])
        detections[i] = trueDets[i] / allPoses[i]

        if np.isnan(recall[i]):
            recall[i] = 0.0
        if np.isnan(precision[i]):
            precision[i] = 0.0

        print('CLS: ', i)
        print('true detections: ', detections[i])
        print('recall: ', recall[i])
        print('precision: ', precision[i])

    recall_all = np.sum(recall[1:]) / 22.0
    precision_all = np.sum(precision[1:]) / 22.0
    detections_all = np.sum(detections[1:]) / 22.0
    print('ALL: ')
    print('true detections: ', detections_all)
    print('recall: ', recall_all)
    print('precision: ', precision_all)
    '''
