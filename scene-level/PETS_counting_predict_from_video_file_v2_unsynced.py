from __future__ import print_function
import sys
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


from net_def_moreSupervised_consistLoss import build_model_FCN_model_api as build_FCNN
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

    model.load_weights(filepath=os.path.join(model_dir, model_name), by_name=True)
    print(model_name)

    return model





def main(exp_name):

    save_dir = os.path.join("models/", exp_name)
    print("save_dir: {}".format(save_dir))
    scaler_stability_factor = 1000

    stage = 'test_pred'
    print(stage)

    model_name = '03-83.8578-better.h5'
    counting_results_name = 'counting_results/PETS_randomUnsyncedGT_consistLoss/' + model_name
    h5_savename = counting_results_name + '/counting_num_' + stage

    if os.path.isdir(counting_results_name):
        os.rmdir(counting_results_name)
    os.mkdir(counting_results_name)

    model = build_model_load_weights(image_dim=(288, 384, 1),
                                     model_dir='models/PETS_randomUnsyncedGT_consistLoss/',
                                     model_name=model_name)  # projection/

    #################################################################

    # train
    train_path0 = '/home/data/CityU_backup/qnap/dataset/Multi-view/PETS_2009/dmaps/'
    train_path1 = '/home/data/CityU_backup/qnap/dataset/Multi-view/PETS_2009/unsynced_dmaps/randomUnsynced/'

    train_view1_1 = train_path0 + 'S1L3/14_17/train_test/PETS_S1L3_1_view1_train_test_10.h5'
    train_view1_2 = train_path0 + 'S1L3/14_33/train_test/PETS_S1L3_2_view1_train_test_10.h5'
    train_view1_3 = train_path0 + 'S2L2/14_55/train_test/10_10/PETS_S2L2_1_view1_train_test_10.h5'
    train_view1_4 = train_path0 + 'S2L3/14_41/train_test/10_10/PETS_S2L3_1_view1_train_test_10.h5'

    train_view2_1 = train_path0 + 'S1L3/14_17/train_test/PETS_S1L3_1_view2_train_test_10.h5'
    train_view2_2 = train_path0 + 'S1L3/14_33/train_test/PETS_S1L3_2_view2_train_test_10.h5'
    train_view2_3 = train_path0 + 'S2L2/14_55/train_test/10_10/PETS_S2L2_1_view2_train_test_10.h5'
    train_view2_4 = train_path0 + 'S2L3/14_41/train_test/10_10/PETS_S2L3_1_view2_train_test_10.h5'

    train_view3_1 = train_path0 + 'S1L3/14_17/train_test/PETS_S1L3_1_view3_train_test_10.h5'
    train_view3_2 = train_path0 + 'S1L3/14_33/train_test/PETS_S1L3_2_view3_train_test_10.h5'
    train_view3_3 = train_path0 + 'S2L2/14_55/train_test/10_10/PETS_S2L2_1_view3_train_test_10.h5'
    train_view3_4 = train_path0 + 'S2L3/14_41/train_test/10_10/PETS_S2L3_1_view3_train_test_10.h5'

    train_GP_1 = train_path1 + 'GP_maps/PETS_S1L3_1_groundplane_dmaps_10.h5'
    train_GP_2 = train_path1 + 'GP_maps/PETS_S1L3_2_groundplane_dmaps_10.h5'
    train_GP_3 = train_path1 + 'GP_maps/PETS_S2L2_1_groundplane_dmaps_10.h5'
    train_GP_4 = train_path1 + 'GP_maps/PETS_S2L3_1_groundplane_dmaps_10.h5'

    h5file_train_view1 = [train_view1_1, train_view1_2, train_view1_3, train_view1_4]
    h5file_train_view2 = [train_view2_1, train_view2_2, train_view2_3, train_view2_4]
    h5file_train_view3 = [train_view3_1, train_view3_2, train_view3_3, train_view3_4]
    h5file_train_GP = [train_GP_1, train_GP_2, train_GP_3, train_GP_4]

    test_path0 = '/home/data/CityU_backup/qnap/dataset/Multi-view/PETS_2009/dmaps/'
    test_view1_1 = test_path0 + 'S1L1/13_57/train_test/PETS_S1L1_1_view1_train_test_10.h5'
    test_view1_2 = test_path0 + 'S1L1/13_59/train_test/PETS_S1L1_2_view1_train_test_10.h5'
    test_view1_3 = test_path0 + 'S1L2/14_06/train_test/PETS_S1L2_1_view1_train_test_10.h5'
    test_view1_4 = test_path0 + 'S1L2/14_31/train_test/PETS_S1L2_2_view1_train_test_10.h5'

    test_view2_1 = test_path0 + 'S1L1/13_57/train_test/PETS_S1L1_1_view2_train_test_10.h5'
    test_view2_2 = test_path0 + 'S1L1/13_59/train_test/PETS_S1L1_2_view2_train_test_10.h5'
    test_view2_3 = test_path0 + 'S1L2/14_06/train_test/PETS_S1L2_1_view2_train_test_10.h5'
    test_view2_4 = test_path0 + 'S1L2/14_31/train_test/PETS_S1L2_2_view2_train_test_10.h5'

    test_view3_1 = test_path0 + 'S1L1/13_57/train_test/PETS_S1L1_1_view3_train_test_10.h5'
    test_view3_2 = test_path0 + 'S1L1/13_59/train_test/PETS_S1L1_2_view3_train_test_10.h5'
    test_view3_3 = test_path0 + 'S1L2/14_06/train_test/PETS_S1L2_1_view3_train_test_10.h5'
    test_view3_4 = test_path0 + 'S1L2/14_31/train_test/PETS_S1L2_2_view3_train_test_10.h5'

    test_GP_1 = train_path1 + 'GP_maps/PETS_S1L1_1_groundplane_dmaps_10.h5'
    test_GP_2 = train_path1 + 'GP_maps/PETS_S1L1_2_groundplane_dmaps_10.h5'
    test_GP_3 = train_path1 + 'GP_maps/PETS_S1L2_1_groundplane_dmaps_10.h5'
    test_GP_4 = train_path1 + 'GP_maps/PETS_S1L2_2_groundplane_dmaps_10.h5'

    h5file_test_GP = [test_GP_1, test_GP_2, test_GP_3, test_GP_4]

    h5file_test_view1 = [test_view1_1, test_view1_2, test_view1_3, test_view1_4]
    h5file_test_view2 = [test_view2_1, test_view2_2, test_view2_3, test_view2_4]
    h5file_test_view3 = [test_view3_1, test_view3_2, test_view3_3, test_view3_4]

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

    # predi

    count_view1_roi_GP = []
    count_view2_roi_GP = []
    count_view3_roi_GP = []
    count_gplane = []
    pred_dmap_gplane = []

    # GP mask:
    view1_gp_mask = np.load('coords_correspondence/mask/view1_gp_mask.npz')
    view1_gp_mask = view1_gp_mask.f.arr_0
    view2_gp_mask = np.load('coords_correspondence/mask/view2_gp_mask.npz')
    view2_gp_mask = view2_gp_mask.f.arr_0
    view3_gp_mask = np.load('coords_correspondence/mask/view3_gp_mask.npz')
    view3_gp_mask = view3_gp_mask.f.arr_0

    view_gp_mask = view1_gp_mask + view2_gp_mask + view3_gp_mask
    view_gp_mask = np.clip(view_gp_mask, 0, 1)
    # plt.imshow(view_gp_mask)
    view1_gp_mask2 = cv2.resize(view1_gp_mask, (int(610 / 4), int(710 / 4)))
    view2_gp_mask2 = cv2.resize(view2_gp_mask, (int(610 / 4), int(710 / 4)))
    view3_gp_mask2 = cv2.resize(view3_gp_mask, (int(610 / 4), int(710 / 4)))
    view_gp_mask = cv2.resize(view_gp_mask, (int(610 / 4), int(710 / 4)))

    ################################## use the synced model to test on the unsycned input  ##########
    f_offset = 35
    # the frame offset is 3 frames (0, +f_offset, -f_offset)
    # [f_offset: -f_offset, 2f_offset: , : -2f_offset] or [f_offset: -f_offset, :-2f_offset , 2f_offset:]
    # GT: [f_offset: -f_offset]

    for j in range(4):

        # view 1
        density_maps1 = np.zeros([1, dmp_h, dmp_w, 1])
        images1 = np.zeros([1, img_h, img_w, 1])

        h5file_view1_i = h5file_view1[j]
        with h5py.File(h5file_view1_i, 'r') as f:
            images_i = f['images'].value[np.abs(f_offset):-np.abs(f_offset), :, :, :]
            density_maps_i = f['density_maps'].value[np.abs(f_offset):-np.abs(f_offset), :, :, :]
        density_maps1 = np.concatenate([density_maps1, density_maps_i], 0)
        images1 = np.concatenate([images1, images_i], 0)

        density_maps1 = density_maps1[1:, :, :, :]
        images1 = images1[1:, :, :, :]

        h1_test = images1
        h1_dmaps_test = density_maps1

        num = h1_dmaps_test.shape[0]

        # view 2
        fixed_seed = 998
        np.random.seed(fixed_seed)  # Set seed for reproducibility
        import tensorflow as tf
        tf.set_random_seed(fixed_seed)

        density_maps2 = np.zeros([1, dmp_h, dmp_w, 1])
        images2 = np.zeros([1, img_h, img_w, 1])

        h5file_view2_i = h5file_view2[j]
        with h5py.File(h5file_view2_i, 'r') as f:
            images_i = f['images'].value  # [:-2*f_offset, :, :, :] # [2*f_offset:, :, :, :]
            density_maps_i = f['density_maps'].value  # [:-2*f_offset, :, :, :] # [2*f_offset:, :, :, :]
        density_maps2 = np.concatenate([density_maps2, density_maps_i], 0)
        images2 = np.concatenate([images2, images_i], 0)

        density_maps2 = density_maps2[1:, :, :, :]
        images2 = images2[1:, :, :, :]

        # choose frames randomly:
        ## set ramdom index
        random_index_view2 = np.random.randint(-f_offset, f_offset, num)
        random_index_view2 = random_index_view2 + np.asarray(range(f_offset, num + f_offset))

        density_maps2 = density_maps2[random_index_view2, :, :, :]
        images2 = images2[random_index_view2, :, :, :]

        h2_test = images2
        h2_dmaps_test = density_maps2

        # view 3
        fixed_seed = 997
        np.random.seed(fixed_seed)  # Set seed for reproducibility
        import tensorflow as tf
        tf.set_random_seed(fixed_seed)

        density_maps3 = np.zeros([1, dmp_h, dmp_w, 1])
        images3 = np.zeros([1, img_h, img_w, 1])
        h5file_view3_i = h5file_view3[j]
        with h5py.File(h5file_view3_i, 'r') as f:
            images_i = f['images'].value  # [2*f_offset:, :, :, :] # [:-2*f_offset, :, :, :]
            density_maps_i = f['density_maps'].value  # [2*f_offset:, :, :, :] #[:-2*f_offset, :, :, :]
        density_maps3 = np.concatenate([density_maps3, density_maps_i], 0)
        images3 = np.concatenate([images3, images_i], 0)

        density_maps3 = density_maps3[1:, :, :, :]
        images3 = images3[1:, :, :, :]

        # choose frames randomly:
        ## set ramdom index
        random_index_view3 = np.random.randint(-f_offset, f_offset, num)
        random_index_view3 = random_index_view3 + np.asarray(range(f_offset, num + f_offset))

        density_maps3 = density_maps3[random_index_view3, :, :, :]
        images3 = images3[random_index_view3, :, :, :]

        h3_test = images3
        h3_dmaps_test = density_maps3

        # GP
        density_maps4 = np.zeros([1, gdmp_h, gdmp_w, 1])
        # images4 = np.asarray([])
        h5file_GP_i = h5file_GP[j]
        with h5py.File(h5file_GP_i, 'r') as f:
            # images_i = f['images'].value
            density_maps_i = f['density_maps'].value#[np.abs(f_offset):-np.abs(f_offset), :, :, :]
        density_maps4 = np.concatenate([density_maps4, density_maps_i], 0)

        density_maps4 = density_maps4[1:, :, :, :]

        h4_dmaps_test = density_maps4

        # print(h4_dmaps_test.shape)

        # depth ratio maps input
        # view 1
        scale_number = 3
        scale_range = range(scale_number)
        scale_size = 2 * 4
        # view 1
        view1_image_depth = np.load('coords_correspondence/view_depth_image/'
                                    'v1_1_depth_image_halfHeight.npz')
        view1_image_depth = view1_image_depth.f.arr_0
        h = view1_image_depth.shape[0]
        w = view1_image_depth.shape[1]
        h_scale = int(h / scale_size)
        w_scale = int(w / scale_size)
        view1_image_depth_resized = cv2.resize(view1_image_depth, (w_scale, h_scale))

        # set the center's scale of the image view1 as median of the all scales
        scale_center = np.median(scale_range)
        depth_center = view1_image_depth_resized[int(h_scale / 2), int(w_scale / 2)]
        view1_image_depth_resized_log2 = np.log2(view1_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view1_image_depth_resized_log2
        view1_image_depth_resized_log2 = np.expand_dims(view1_image_depth_resized_log2, axis=2)
        view1_image_depth_resized_log2 = np.expand_dims(view1_image_depth_resized_log2, axis=0)

        # view 2
        view2_image_depth = np.load('coords_correspondence/view_depth_image/'
                                    'v1_2_depth_image_halfHeight.npz')
        view2_image_depth = view2_image_depth.f.arr_0
        view2_image_depth_resized = cv2.resize(view2_image_depth, (w_scale, h_scale))
        view2_image_depth_resized_log2 = np.log2(view2_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view2_image_depth_resized_log2
        view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=2)
        view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=0)

        # view 3
        view3_image_depth = np.load('coords_correspondence/view_depth_image/'
                                    'v1_3_depth_image_halfHeight.npz')
        view3_image_depth = view3_image_depth.f.arr_0
        view3_image_depth_resized = cv2.resize(view3_image_depth, (w_scale, h_scale))
        view3_image_depth_resized_log2 = np.log2(view3_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view3_image_depth_resized_log2
        view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=2)
        view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=0)



        count1 = []
        count2 = []
        count3 = []

        list_pred = list()
        pred_dmaps_list = []
        image_dim = None
        f_count = 0

        # print(h3_test.shape[0])

        for i in range(h3_test.shape[0]):

            frame1_s0 = h1_test[i:i + 1]
            frame1 = frame1_s0[0, :, :, 0]

            frame1_s1_0 = cv2.resize(frame1, (int(frame1.shape[1] / 2), int(frame1.shape[0] / 2)))
            frame1_s1 = np.expand_dims(frame1_s1_0, axis=0)
            frame1_s1 = np.expand_dims(frame1_s1, axis=3)

            frame1_s2_0 = cv2.resize(frame1_s1_0, (int(frame1_s1_0.shape[1] / 2), int(frame1_s1_0.shape[0] / 2)))
            frame1_s2 = np.expand_dims(frame1_s2_0, axis=0)
            frame1_s2 = np.expand_dims(frame1_s2, axis=3)

            frame2_s0 = h2_test[i:i + 1]
            frame2 = frame2_s0[0, :, :, 0]

            frame2_s1_0 = cv2.resize(frame2, (int(frame2.shape[1] / 2), int(frame2.shape[0] / 2)))
            frame2_s1 = np.expand_dims(frame2_s1_0, axis=0)
            frame2_s1 = np.expand_dims(frame2_s1, axis=3)

            frame2_s2_0 = cv2.resize(frame2_s1_0, (int(frame2_s1_0.shape[1] / 2), int(frame2_s1_0.shape[0] / 2)))
            frame2_s2 = np.expand_dims(frame2_s2_0, axis=0)
            frame2_s2 = np.expand_dims(frame2_s2, axis=3)

            frame3_s0 = h3_test[i:i + 1]
            frame3 = frame3_s0[0, :, :, 0]

            frame3_s1_0 = cv2.resize(frame3, (int(frame3.shape[1] / 2), int(frame3.shape[0] / 2)))
            frame3_s1 = np.expand_dims(frame3_s1_0, axis=0)
            frame3_s1 = np.expand_dims(frame3_s1, axis=3)

            frame3_s2_0 = cv2.resize(frame3_s1_0, (int(frame3_s1_0.shape[1] / 2), int(frame3_s1_0.shape[0] / 2)))
            frame3_s2 = np.expand_dims(frame3_s2_0, axis=0)
            frame3_s2 = np.expand_dims(frame3_s2, axis=3)

            dmap1 = h1_dmaps_test[i:i + 1]
            dmap2 = h2_dmaps_test[i:i + 1]
            dmap3 = h3_dmaps_test[i:i + 1]
            dmap4 = h4_dmaps_test[i:i + 1]

            count1_gt_i = np.sum(np.sum(dmap1[0, :, :, 0])) / 1000
            count2_gt_i = np.sum(np.sum(dmap2[0, :, :, 0])) / 1000
            count3_gt_i = np.sum(np.sum(dmap3[0, :, :, 0])) / 1000
            count4_gt_i = np.sum(np.sum(dmap4[0, :, :, 0]))

            pred_dmap = model.predict_on_batch([frame1_s0, frame1_s1, frame1_s2,

                                                # frame2_s0, frame2_s1, frame2_s2,
                                                # frame3_s0, frame3_s1, frame3_s2,

                                                frame2_s0, frame2_s1, frame2_s2,
                                                frame3_s0, frame3_s1, frame3_s2,

                                                view1_image_depth_resized_log2,
                                                view2_image_depth_resized_log2,
                                                view3_image_depth_resized_log2])

            # count1_pred_i = np.sum(pred_dmap[0].flatten())/1000
            # count2_pred_i = np.sum(pred_dmap[1].flatten())/1000
            # count3_pred_i = np.sum(pred_dmap[2].flatten())/1000

            pred_dmap_0 = pred_dmap[-3]
            # pred_dmap_0 = pred_dmap_0*view_gp_mask
            count4_pred_i = np.sum(pred_dmap_0.flatten()) / 1000

            pred_dmap_gplane.append(pred_dmap_0)

            # count1.append([count1_gt_i, count1_pred_i])
            # count2.append([count2_gt_i, count2_pred_i])
            # count3.append([count3_gt_i, count3_pred_i])
            count_gplane.append([count1_gt_i, count2_gt_i, count3_gt_i, count4_gt_i, count4_pred_i])

            # roi GP pred
            count_view1_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0] * view1_gp_mask2)) / 1000
            count_view2_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0] * view2_gp_mask2)) / 1000
            count_view3_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0] * view3_gp_mask2)) / 1000
            # roi GP gt
            count_view1_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view1_gp_mask))
            count_view2_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view2_gp_mask))
            count_view3_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view3_gp_mask))
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

    # GP roi/GP
    dif_view1_GP = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view1_GP = np.mean(np.abs(dif_view1_GP))
    print(mae_view1_GP)
    dif_view2_GP = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view2_GP = np.mean(np.abs(dif_view2_GP))
    print(mae_view2_GP)
    dif_view3_GP = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view3_GP = np.mean(np.abs(dif_view3_GP))
    print(mae_view3_GP)

    # GP roi
    mae_view1_GProi = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_view1_roi_GP)[:, 0]
    mae_view1_GProi = np.mean(np.abs(mae_view1_GProi))
    print(mae_view1_GProi)
    mae_view2_GProi = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_view2_roi_GP)[:, 0]
    mae_view2_GProi = np.mean(np.abs(mae_view2_GProi))
    print(mae_view2_GProi)
    mae_view3_GProi = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_view3_roi_GP)[:, 0]
    mae_view3_GProi = np.mean(np.abs(mae_view3_GProi))
    print(mae_view3_GProi)

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

    with h5py.File(h5_savename, 'w') as f:
        f.create_dataset("count1_GProi", data=count_view1_roi_GP)
        f.create_dataset("count2_GProi", data=count_view2_roi_GP)
        f.create_dataset("count3_GProi", data=count_view3_roi_GP)
        f.create_dataset("count_gplane", data=count_gplane)

        f.create_dataset("mae_GP", data=mae_GP)
        f.create_dataset("nae_GP", data=nae_GP)

        f.create_dataset("mae_view1_GProi", data=mae_view1_GProi)
        f.create_dataset("mae_view2_GProi", data=mae_view2_GProi)
        f.create_dataset("mae_view3_GProi", data=mae_view3_GProi)

        f.create_dataset("pre_dmap_gplane", data=pred_dmap_gplane)

        f.create_dataset("mae_view1", data=mae_view1)
        f.create_dataset("mae_view2", data=mae_view2)
        f.create_dataset("mae_view3", data=mae_view3)

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