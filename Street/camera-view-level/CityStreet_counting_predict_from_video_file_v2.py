from __future__ import print_function
import os
import sys
import re
import glob
import h5py
import numpy as np

np.set_printoptions(precision=6)
fixed_seed = 999
np.random.seed(fixed_seed)  # Set seed for reproducibility
import tensorflow as tf

tf.set_random_seed(fixed_seed)
import keras

print("Using keras {}".format(keras.__version__))
assert keras.__version__.startswith('2.')
from keras.optimizers import SGD
# from datagen import datagen

from net_def import build_model_FCN_model_api as build_FCNN



import cv2
# assert cv2.__version__.startswith('3.4')

# import im_patch
# import matplotlib.pyplot as plt

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# set enough GPU memory as needed(default, all GPU memory is used)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def check_folder(dir_path):
    files = os.listdir(dir_path)
    xls_count = 0
    avi_count = 0
    mp4_count = 0
    # print("all {} files: {}".format(len(files), files))
    for file in files:
        if file.endswith('.xls'):
            xls_count += 1
            # print("found one label file: {}".format(file))
        elif file.endswith('.avi'):
            avi_count += 1
            # print("found one data file: {}".format(file))
        elif file.endswith('.mp4'):
            mp4_count += 1

        elif file.endswith(('.h5', '.hdf5')):
            print("found one label file (h5): {}".format(file))
        else:
            print("Unknown file type: {}".format(file))
            sys.exit(1)
    assert avi_count == xls_count
    if xls_count > 1 or avi_count > 1:
        print("more than one data file: {}".format(files))
        return False
    else:
        return True


def build_model_load_weights(image_dim, model_dir, model_name):
    opt = SGD(lr=0.001)
    model = build_FCNN(
        batch_size=1,
        patch_size=image_dim,
        optimizer=opt,
        output_ROI_mask=False,
    )
    # weight_files = os.listdir(model_dir)
    # weight_files.sort()
    # weight_files = weight_files[::-1]
    #
    # pattern = re.compile(model_name)               # 14-5.8780
    # #07-1067.8649.h5 29-5.8353 07-1067.8649 04-1067.5609-better.h5
    #
    # for file in weight_files:
    #     has_match = pattern.search(file)
    #     if has_match:
    #         break
    # best_model = has_match.group(0)
    # print(">>>> best_model: {}".format(best_model))
    # best_model = weight_files[-1]
    model.load_weights(filepath=os.path.join(model_dir, model_name), by_name=True)
    print(model_name)
    # model.save_weights('load_weights/feature_weights/06-2984.1581-better_encoder_decoder.h5')

    return model



def GAME_recursive(density, gt, currentLevel, targetLevel):
    density_slice = range(4)
    gt_slice = range(4)
    res = range(4)

    if currentLevel == targetLevel:
        a = sum(density.flatten())
        b = sum(gt.flatten())
        game = abs(sum(density.flatten()) - sum(gt.flatten()))
    else:

        # Zero - padding to make the image even if needed
        dim = density.shape
        if np.mod(dim[0], 2) != 0:
            density = np.pad(density, [[1, 0], [0, 0]], 'constant')
            gt = np.pad(gt, [[1, 0], [0, 0]], 'constant')

        if np.mod(dim[1], 2) != 0:
            density = np.pad(density, [[0, 0], [0, 1]], 'constant')
            gt = np.pad(gt, [[1, 0], [0, 0]], 'constant')

        dim2 = gt.shape

        # Creating the four slices
        density_slice[0] = density[:dim[0]/2, :dim[1]/2]
        density_slice[1] = density[:dim[0]/2, dim[1]/2:]
        density_slice[2] = density[dim[0]/2:, :dim[1]/2]
        density_slice[3] = density[dim[0]/2:, dim[1]/2:]

        gt_slice[0] = gt[:dim2[0]/2, :dim2[1]/2]
        gt_slice[1] = gt[:dim2[0]/2, dim2[1]/2:]
        gt_slice[2] = gt[dim2[0]/2:, :dim2[1]/2]
        gt_slice[3] = gt[dim2[0]/2:, dim2[1]/2:]

        currentLevel = currentLevel + 1

        for a in range(4):
            res[a] = GAME_recursive(density_slice[a], gt_slice[a], currentLevel, targetLevel)

        game = sum(res)

    return game




def main(exp_name):
    # save_dir = os.path.join("/opt/visal/di/models/models_keras2/model_cell", exp_name)
    save_dir = os.path.join("models/", exp_name)
    print("save_dir: {}".format(save_dir))
    scaler_stability_factor = 1000

    stage = 'test_pred'
    print(stage)

    model_name = 'xxx'
    counting_results_name = 'counting_results/xxx/' + model_name
    h5_savename = counting_results_name + '/counting_num_' + stage

    if os.path.isdir(counting_results_name):
        os.rmdir(counting_results_name)
    os.mkdir(counting_results_name)

    model = build_model_load_weights(image_dim=(380, 676, 1),
                                     model_dir='models/xxx',
                                     model_name=model_name)  # projection/
    print(model_name)
    #################################################################

    # train
    train_path0 = '../../dataset/Street/'


    train_view1_1 = train_path0 + 'dmaps/train/Street_view1_dmap_10.h5'
    train_view2_1 = train_path0 + 'dmaps/train/Street_view2_dmap_10.h5'
    train_view3_1 = train_path0 + 'dmaps/train/Street_view3_dmap_10.h5'

    train_view2_1_u = train_path0 + 'unsynced_images_h5/timeshift_[0+3-3]s/train/Street_train_view2_gray_images_unsynced.h5'
    train_view3_1_u = train_path0 + 'unsynced_images_h5/timeshift_[0+3-3]s/train/Street_train_view3_gray_images_unsynced.h5'


    train_GP_1 = train_path0 + 'GP_dmaps/train/Street_groundplane_train_dmaps_10.h5'

    h5file_train_view1 = [train_view1_1]
    h5file_train_view2 = [train_view2_1]
    h5file_train_view3 = [train_view3_1]
    h5file_train_view2_u = [train_view2_1_u]
    h5file_train_view3_u = [train_view3_1_u]
    h5file_train_GP = [train_GP_1]


    test_view1_1 = train_path0 + 'dmaps/test/Street_view1_dmap_10.h5'
    test_view2_1 = train_path0 + 'dmaps/test/Street_view2_dmap_10.h5'
    test_view3_1 = train_path0 + 'dmaps/test/Street_view3_dmap_10.h5'

    test_view2_1_u = train_path0 + 'unsynced_images_h5/timeshift_[0+3-3]s/test/Street_test_view2_gray_images_unsynced.h5'
    test_view3_1_u = train_path0 + 'unsynced_images_h5/timeshift_[0+3-3]s/test/Street_test_view3_gray_images_unsynced.h5'

    test_GP_1 = train_path0 + 'GP_dmaps/test/Street_groundplane_test_dmaps_10.h5'

    h5file_test_view1 = [test_view1_1]
    h5file_test_view2 = [test_view2_1]
    h5file_test_view3 = [test_view3_1]
    h5file_test_view2_u = [test_view2_1_u]
    h5file_test_view3_u = [test_view3_1_u]
    h5file_test_GP = [test_GP_1]

    if stage == 'train':
        h5file_view1 = h5file_train_view1
        h5file_view2 = h5file_train_view2
        h5file_view3 = h5file_train_view3
        h5file_GP = h5file_train_GP
    else:
        h5file_view1 = h5file_test_view1
        h5file_view2 = h5file_test_view2
        h5file_view3 = h5file_test_view3
        h5file_GP = h5file_test_GP

    # load the train or test data
    with h5py.File(h5file_view1[0], 'r') as f:
        images_i = f['images'].value
        density_maps_i = f['density_maps'].value
        dmp_h = density_maps_i.shape[1]
        dmp_w = density_maps_i.shape[2]
        img_h = images_i.shape[1]
        img_w = images_i.shape[2]

    with h5py.File(h5file_GP[0], 'r') as f:
        density_maps_i = f['density_maps'].value
        gdmp_h = density_maps_i.shape[1]
        gdmp_w = density_maps_i.shape[2]



    count_view1_roi_GP = []
    count_view2_roi_GP = []
    count_view3_roi_GP = []
    count_gplane = []
    pred_dmap_gplane = []

    GAME_all = []

    for j in range(1):

        # view 1
        density_maps1 = np.zeros([1, dmp_h, dmp_w, 1])
        images1 = np.zeros([1, img_h, img_w, 1])

        h5file_view1_i = h5file_view1[j]
        with h5py.File(h5file_view1_i, 'r') as f:
            images_i = f['images'].value
            density_maps_i = f['density_maps'].value
        density_maps1 = np.concatenate([density_maps1, density_maps_i], 0)
        images1 = np.concatenate([images1, images_i], 0)

        density_maps1 = density_maps1[1:, :, :, :]
        images1 = images1[1:, :, :, :]
        h1_test = images1
        h1_dmaps_test = density_maps1

        # view 2
        density_maps2 = np.zeros([1, dmp_h, dmp_w, 1])
        images2 = np.zeros([1, img_h, img_w, 1])

        h5file_view2_i = h5file_view2[j]
        with h5py.File(h5file_view2_i, 'r') as f:
            images_i = f['images'].value
            density_maps_i = f['density_maps'].value
        density_maps2 = np.concatenate([density_maps2, density_maps_i], 0)
        images2 = np.concatenate([images2, images_i], 0)

        density_maps2 = density_maps2[1:, :, :, :]
        images2 = images2[1:, :, :, :]
        h2_test = images2
        h2_dmaps_test = density_maps2

        # view 3
        density_maps3 = np.zeros([1, dmp_h, dmp_w, 1])
        images3 = np.zeros([1, img_h, img_w, 1])
        h5file_view3_i = h5file_view3[j]
        with h5py.File(h5file_view3_i, 'r') as f:
            images_i = f['images'].value
            density_maps_i = f['density_maps'].value
        density_maps3 = np.concatenate([density_maps3, density_maps_i], 0)
        images3 = np.concatenate([images3, images_i], 0)
        density_maps3 = density_maps3[1:, :, :, :]
        images3 = images3[1:, :, :, :]
        h3_test = images3
        h3_dmaps_test = density_maps3

        ##########  unsynced input:  ###################
        h5file_view2_u = h5file_test_view2_u[j]
        with h5py.File(h5file_view2_u, 'r') as f:
            images_i_u = f['images'].value
        images2_u = images_i_u

        h5file_view3_u = h5file_test_view3_u[j]
        with h5py.File(h5file_view3_u, 'r') as f:
            images_i_u = f['images'].value
        images3_u = images_i_u

        # GP
        density_maps4 = np.zeros([1, gdmp_h, gdmp_w, 1])
        # images4 = np.asarray([])
        h5file_GP_i = h5file_GP[j]
        with h5py.File(h5file_GP_i, 'r') as f:
            # images_i = f['images'].value
            density_maps_i = f['density_maps'].value
        density_maps4 = np.concatenate([density_maps4, density_maps_i], 0)
        # images3 = np.concatenate([images3, images_i], 0)
        density_maps4 = density_maps4[1:, :, :, :]
        h4_dmaps_test = density_maps4




        ########## depth ratio maps input ###########
        # view 1
        scale_number = 3
        scale_range = range(scale_number)
        scale_size = 4
        # view 1
        view1_image_depth = np.load('coords_correspondence_Street/view_depth_image/'
                                    'v1_1_depth_image_avgHeight.npz')
        view1_image_depth = view1_image_depth.f.arr_0
        h = view1_image_depth.shape[0]
        w = view1_image_depth.shape[1]
        h_scale = h / scale_size
        w_scale = w / scale_size
        view1_image_depth_resized = cv2.resize(view1_image_depth, (w_scale, h_scale))

        # set the center's scale of the image view1 as median of the all scales
        scale_center = np.median(scale_range)
        depth_center = view1_image_depth_resized[h_scale / 2, w_scale / 2]
        view1_image_depth_resized_log2 = np.log2(view1_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view1_image_depth_resized_log2
        view1_image_depth_resized_log2 = np.expand_dims(view1_image_depth_resized_log2, axis=2)
        view1_image_depth_resized_log2 = np.expand_dims(view1_image_depth_resized_log2, axis=0)

        # view 2
        view2_image_depth = np.load('coords_correspondence_Street/view_depth_image/'
                                    'v1_2_depth_image_avgHeight.npz')
        view2_image_depth = view2_image_depth.f.arr_0
        view2_image_depth_resized = cv2.resize(view2_image_depth, (w_scale, h_scale))
        view2_image_depth_resized_log2 = np.log2(view2_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view2_image_depth_resized_log2
        view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=2)
        view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=0)

        # view 3
        view3_image_depth = np.load('coords_correspondence_Street/view_depth_image/'
                                    'v1_3_depth_image_avgHeight.npz')
        view3_image_depth = view3_image_depth.f.arr_0
        view3_image_depth_resized = cv2.resize(view3_image_depth, (w_scale, h_scale))
        view3_image_depth_resized_log2 = np.log2(view3_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view3_image_depth_resized_log2
        view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=2)
        view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=0)

        # GP mask:
        view1_gp_mask = np.load('coords_correspondence_Street/mask/view1_GP_mask.npz')
        view1_gp_mask = view1_gp_mask.f.arr_0
        view2_gp_mask = np.load('coords_correspondence_Street/mask/view2_GP_mask.npz')
        view2_gp_mask = view2_gp_mask.f.arr_0
        view3_gp_mask = np.load('coords_correspondence_Street/mask/view3_GP_mask.npz')
        view3_gp_mask = view3_gp_mask.f.arr_0

        view_gp_mask = view1_gp_mask + view2_gp_mask + view3_gp_mask
        view_gp_mask = np.clip(view_gp_mask, 0, 1)
        # plt.imshow(view_gp_mask)
        view1_gp_mask2 = cv2.resize(view1_gp_mask, (640, 768))
        view2_gp_mask2 = cv2.resize(view2_gp_mask, (640, 768))
        view3_gp_mask2 = cv2.resize(view3_gp_mask, (640, 768))
        view_gp_mask = cv2.resize(view_gp_mask, (640 / 4, 768 / 4))

        count1 = []
        count2 = []
        count3 = []

        list_pred = list()
        pred_dmaps_list = []
        image_dim = None
        f_count = 0

        #plt.figure()

        for i in range(h3_test.shape[0]):
            # i = 121

            frame1_s0 = h1_test[i:i + 1]
            frame1 = frame1_s0[0, :, :, 0]

            frame1_s1_0 = cv2.resize(frame1, (frame1.shape[1] / 2, frame1.shape[0] / 2))
            frame1_s1 = np.expand_dims(frame1_s1_0, axis=0)
            frame1_s1 = np.expand_dims(frame1_s1, axis=3)

            frame1_s2_0 = cv2.resize(frame1_s1_0, (frame1_s1_0.shape[1] / 2, frame1_s1_0.shape[0] / 2))
            frame1_s2 = np.expand_dims(frame1_s2_0, axis=0)
            frame1_s2 = np.expand_dims(frame1_s2, axis=3)

            # frame2_s0 = h2_test[i:i + 1]
            # frame2 = frame2_s0[0, :, :, 0]
            #
            # frame2_s1_0 = cv2.resize(frame2, (frame2.shape[1] / 2, frame2.shape[0] / 2))
            # frame2_s1 = np.expand_dims(frame2_s1_0, axis=0)
            # frame2_s1 = np.expand_dims(frame2_s1, axis=3)
            #
            # frame2_s2_0 = cv2.resize(frame2_s1_0, (frame2_s1_0.shape[1] / 2, frame2_s1_0.shape[0] / 2))
            # frame2_s2 = np.expand_dims(frame2_s2_0, axis=0)
            # frame2_s2 = np.expand_dims(frame2_s2, axis=3)
            #
            # frame3_s0 = h3_test[i:i + 1]
            # frame3 = frame3_s0[0, :, :, 0]
            #
            # frame3_s1_0 = cv2.resize(frame3, (frame3.shape[1] / 2, frame3.shape[0] / 2))
            # frame3_s1 = np.expand_dims(frame3_s1_0, axis=0)
            # frame3_s1 = np.expand_dims(frame3_s1, axis=3)
            #
            # frame3_s2_0 = cv2.resize(frame3_s1_0, (frame3_s1_0.shape[1] / 2, frame3_s1_0.shape[0] / 2))
            # frame3_s2 = np.expand_dims(frame3_s2_0, axis=0)
            # frame3_s2 = np.expand_dims(frame3_s2, axis=3)


            ########################## unsynced input ###########################
            img2_u_s0 = images2_u[i, :, :, 0]
            img2_u_s1 = cv2.resize(img2_u_s0, (img2_u_s0.shape[1] / 2, img2_u_s0.shape[0] / 2))
            img2_u_s2 = cv2.resize(img2_u_s1, (img2_u_s1.shape[1] / 2, img2_u_s1.shape[0] / 2))

            img2_u_s0 = np.expand_dims(img2_u_s0, axis=0)
            img2_u_s0 = np.expand_dims(img2_u_s0, axis=3)
            img2_u_s1 = np.expand_dims(img2_u_s1, axis=0)
            img2_u_s1 = np.expand_dims(img2_u_s1, axis=3)
            img2_u_s2 = np.expand_dims(img2_u_s2, axis=0)
            img2_u_s2 = np.expand_dims(img2_u_s2, axis=3)


            img3_u_s0 = images3_u[i, :, :, 0]
            img3_u_s1 = cv2.resize(img3_u_s0, (img3_u_s0.shape[1] / 2, img3_u_s0.shape[0] / 2))
            img3_u_s2 = cv2.resize(img3_u_s1, (img3_u_s1.shape[1] / 2, img3_u_s1.shape[0] / 2))

            img3_u_s0 = np.expand_dims(img3_u_s0, axis=0)
            img3_u_s0 = np.expand_dims(img3_u_s0, axis=3)
            img3_u_s1 = np.expand_dims(img3_u_s1, axis=0)
            img3_u_s1 = np.expand_dims(img3_u_s1, axis=3)
            img3_u_s2 = np.expand_dims(img3_u_s2, axis=0)
            img3_u_s2 = np.expand_dims(img3_u_s2, axis=3)


            dmap1 = h1_dmaps_test[i:i + 1]
            dmap2 = h2_dmaps_test[i:i + 1]
            dmap3 = h3_dmaps_test[i:i + 1]
            dmap4 = h4_dmaps_test[i:i + 1]

            count1_gt_i = np.sum(np.sum(dmap1[0, :, :, 0]))
            count2_gt_i = np.sum(np.sum(dmap2[0, :, :, 0]))
            count3_gt_i = np.sum(np.sum(dmap3[0, :, :, 0]))
            count4_gt_i = np.sum(np.sum(dmap4[0, :, :, 0]))


            pred_dmap = model.predict_on_batch([frame1_s0, frame1_s1, frame1_s2,

                                                # frame2_s0, frame2_s1, frame2_s2,
                                                # frame3_s0, frame3_s1, frame3_s2,

                                                img2_u_s0, img2_u_s1, img2_u_s2,
                                                img3_u_s0, img3_u_s1, img3_u_s2,

                                                view1_image_depth_resized_log2,
                                                view2_image_depth_resized_log2,
                                                view3_image_depth_resized_log2])

            # count1_pred_i = np.sum(pred_dmap[0].flatten())/1000
            # count2_pred_i = np.sum(pred_dmap[1].flatten())/1000
            # count3_pred_i = np.sum(pred_dmap[2].flatten())/1000

            pred_dmap_0 = pred_dmap[-3] / 1000
            # pred_dmap_0 = pred_dmap_0*view_gp_mask
            count4_pred_i = np.sum(pred_dmap_0.flatten())
            pred_dmap_gplane.append(pred_dmap_0)

            # GAME metric
            currentLevel = 0
            GAME_l = np.zeros(4)
            for l in range(4):
                targetLevel = l
                GAME_l[l] = GAME_recursive(pred_dmap_0[0, :, :, 0],
                               dmap4[0, :, :, 0], currentLevel, targetLevel)
            GAME_all.append(GAME_l)


            # count1.append([count1_gt_i, count1_pred_i])
            # count2.append([count2_gt_i, count2_pred_i])
            # count3.append([count3_gt_i, count3_pred_i])
            count_gplane.append([count1_gt_i, count2_gt_i, count3_gt_i, count4_gt_i, count4_pred_i])

            # roi GP pred
            count_view1_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0] * view1_gp_mask))
            count_view2_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0] * view2_gp_mask))
            count_view3_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0] * view3_gp_mask))
            # roi GP gt
            count_view1_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view1_gp_mask2))
            count_view2_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view2_gp_mask2))
            count_view3_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view3_gp_mask2))
            count_view1_roi_GP.append([count_view1_roi_GP_gt_i, count_view1_roi_GP_i])
            count_view2_roi_GP.append([count_view2_roi_GP_gt_i, count_view2_roi_GP_i])
            count_view3_roi_GP.append([count_view3_roi_GP_gt_i, count_view3_roi_GP_i])

    # GP
    mae_GP = np.asarray(count_gplane)[:, 4] - np.asarray(count_gplane)[:, 3]
    mae_GP = np.mean(np.abs(mae_GP))
    print(mae_GP)

    # GP_nae
    nae_GP = np.asarray(count_gplane)[:, 4] - np.asarray(count_gplane)[:, 3]
    nae_GP = np.mean(np.abs(nae_GP)/np.asarray(count_gplane)[:, 3])
    print(nae_GP)


    # GAME
    GAME_all = np.asarray(GAME_all)
    GAME = np.mean(GAME_all, axis=0)
    print('GAME:')
    print(GAME)

    # GP roi / GP
    dif_view1_GP = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view1_GP = np.mean(np.abs(dif_view1_GP))
    print(mae_view1_GP)
    dif_view2_GP = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view2_GP = np.mean(np.abs(dif_view2_GP))
    print(mae_view2_GP)
    dif_view3_GP = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view3_GP = np.mean(np.abs(dif_view3_GP))
    print(mae_view3_GP)

    # GP roi / GP_nae
    dif_view1_GP = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    nae_view1_GP = np.mean(np.abs(dif_view1_GP)/np.asarray(count_gplane)[:, 3])
    print(nae_view1_GP)
    dif_view2_GP = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    nae_view2_GP = np.mean(np.abs(dif_view2_GP)/np.asarray(count_gplane)[:, 3])
    print(nae_view2_GP)
    dif_view3_GP = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    nae_view3_GP = np.mean(np.abs(dif_view3_GP)/np.asarray(count_gplane)[:, 3])
    print(nae_view3_GP)



    # GP roi / GP roi
    mae_view1_GProi = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_view1_roi_GP)[:, 0]
    mae_view1_GProi = np.mean(np.abs(mae_view1_GProi))
    print(mae_view1_GProi)
    mae_view2_GProi = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_view2_roi_GP)[:, 0]
    mae_view2_GProi = np.mean(np.abs(mae_view2_GProi))
    print(mae_view2_GProi)
    mae_view3_GProi = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_view3_roi_GP)[:, 0]
    mae_view3_GProi = np.mean(np.abs(mae_view3_GProi))
    print(mae_view3_GProi)

    # GP roi / GP roi_nae
    nae_view1_GProi = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_view1_roi_GP)[:, 0]
    nae_view1_GProi = np.mean(np.abs(nae_view1_GProi)/np.asarray(count_view1_roi_GP)[:, 0])
    print(nae_view1_GProi)
    nae_view2_GProi = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_view2_roi_GP)[:, 0]
    nae_view2_GProi = np.mean(np.abs(nae_view2_GProi)/np.asarray(count_view2_roi_GP)[:, 0])
    print(nae_view2_GProi)
    nae_view3_GProi = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_view3_roi_GP)[:, 0]
    nae_view3_GProi = np.mean(np.abs(nae_view3_GProi)/np.asarray(count_view3_roi_GP)[:, 0])
    print(nae_view3_GProi)



    # GP roi/view
    dif_view1 = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 0]
    mae_view1 = np.mean(np.abs(dif_view1))
    print(mae_view1)
    dif_view2 = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 1]
    mae_view2 = np.mean(np.abs(dif_view2))
    print(mae_view2)
    dif_view3 = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 2]
    mae_view3 = np.mean(np.abs(dif_view3))
    print(mae_view3)

    # GP roi/view_nae
    dif_view1 = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 0]
    nae_view1 = np.mean(np.abs(dif_view1)/np.asarray(count_gplane)[:, 0])
    print(nae_view1)
    dif_view2 = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 1]
    nae_view2 = np.mean(np.abs(dif_view2)/np.asarray(count_gplane)[:, 1])
    print(nae_view2)
    dif_view3 = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 2]
    nae_view3 = np.mean(np.abs(dif_view3)/np.asarray(count_gplane)[:, 2])
    print(nae_view3)



    with h5py.File(h5_savename, 'w') as f:
        f.create_dataset("count1_GProi", data=count_view1_roi_GP)
        f.create_dataset("count2_GProi", data=count_view2_roi_GP)
        f.create_dataset("count3_GProi", data=count_view3_roi_GP)
        f.create_dataset("count_gplane", data=count_gplane)

        f.create_dataset("mae_GP", data=mae_GP)
        f.create_dataset("nae_GP", data=nae_GP)

        f.create_dataset("GAME", data=GAME)
        f.create_dataset("pred_dmap_gplane", data=pred_dmap_gplane)

        f.create_dataset("mae_view1_GP", data=mae_view1_GP)
        f.create_dataset("mae_view2_GP", data=mae_view2_GP)
        f.create_dataset("mae_view3_GP", data=mae_view3_GP)

        f.create_dataset("mae_view1", data=mae_view1)
        f.create_dataset("mae_view2", data=mae_view2)
        f.create_dataset("mae_view3", data=mae_view3)

        f.create_dataset("mae_view1_GProi", data=mae_view1_GProi)
        f.create_dataset("mae_view2_GProi", data=mae_view2_GProi)
        f.create_dataset("mae_view3_GProi", data=mae_view3_GProi)

        f.create_dataset("nae_view1_GP", data=nae_view1_GP)
        f.create_dataset("nae_view2_GP", data=nae_view2_GP)
        f.create_dataset("nae_view3_GP", data=nae_view3_GP)

        f.create_dataset("nae_view1", data=nae_view1)
        f.create_dataset("nae_view2", data=nae_view2)
        f.create_dataset("nae_view3", data=nae_view3)

        f.create_dataset("nae_view1_GProi", data=nae_view1_GProi)
        f.create_dataset("nae_view2_GProi", data=nae_view2_GProi)
        f.create_dataset("nae_view3_GProi", data=nae_view3_GProi)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file",
        type=str,
        default='cell',
        action="store")
    parser.add_argument(
        "-e", "--exp_name",
        type=str,
        default='FCNN-sgd-whole-raw',
        action="store")
    args = parser.parse_args()
    main(exp_name=args.exp_name)
