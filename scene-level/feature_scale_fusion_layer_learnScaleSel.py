from keras.layers.core import Layer
import tensorflow as tf
import numpy as np
import cv2

# import matplotlib.pyplot as plt

from keras.engine import InputSpec


class feature_scale_fusion_layer(Layer):

    def __init__(self,
                 scale_number=3,
                 **kwargs):
        #self.scale = scale
        self.scale_number = scale_number
        #self.view = view
        super(feature_scale_fusion_layer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(feature_scale_fusion_layer, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        scale_number = self.scale_number
        feature = input_shape[0]
        return (int(feature[0]),
                int(feature[1]),
                int(feature[2]),
                int(feature[3]/scale_number))

    def call(self, x):
        scale_number = self.scale_number

        feature = x[0]
        mask = x[1]

        n_channels_single = int(feature.shape[-1].value / scale_number)
        batch_size = feature.shape[0].value
        height = feature.shape[1].value
        width = feature.shape[2].value
        num_channels = feature.shape[3].value
        feature_mask = tf.zeros([batch_size, height, width, n_channels_single])

        for i in range(scale_number):
            feature_i = feature[:, :, :, i * n_channels_single:(i + 1) * n_channels_single]
            mask_i = mask[:, :, :, i:i + 1]
            mask_i_n = tf.tile(mask_i, [1, 1, 1, n_channels_single])
            feature_mask_i = tf.multiply(feature_i, mask_i_n)
            feature_mask = feature_mask + feature_mask_i
        return feature_mask




###################################################################################
#         # load depth ratio map (masked)
#         # view 1
#         view1_image_depth = np.load('coords_correspondence/view_depth_image/'
#                                       'v1_1_depth_image_halfHeight.npz')
#         view1_image_depth = view1_image_depth.f.arr_0
#         # plt.figure()
#         # plt.imshow(view1_image_depth)
#         # plt.show()
#
#         h = view1_image_depth.shape[0]
#         w = view1_image_depth.shape[1]
#         scale_size = 2*4
#         h_scale = h/scale_size
#         w_scale = w/scale_size
#         view1_image_depth_resized = cv2.resize(view1_image_depth,
#                                                (w_scale, h_scale))
#
#         # set the center's scale of the image view1 as median of the all scales
#         scale_center = np.median(scale_range)
#         depth_center = view1_image_depth_resized[h_scale/2, w_scale/2]
#         view1_image_depth_ratio_resized_log2 = np.log2(view1_image_depth_resized/depth_center)
#
#         # view 2
#         view2_image_depth = np.load('coords_correspondence/view_depth_image/'
#                                       'v1_2_depth_image_halfHeight.npz')
#         view2_image_depth = view2_image_depth.f.arr_0
#         # plt.figure()
#         # plt.imshow(view2_image_depth)
#         # plt.show()
#
#         view2_image_depth_resized = cv2.resize(view2_image_depth,
#                                                (w_scale, h_scale))
#         view2_image_depth_ratio_resized_log2 = np.log2(view2_image_depth_resized/depth_center)
#
#         # view 3
#         view3_image_depth = np.load('coords_correspondence/view_depth_image/'
#                                       'v1_3_depth_image_halfHeight.npz')
#         view3_image_depth = view3_image_depth.f.arr_0
#         # plt.figure()
#         # plt.imshow(view3_image_depth)
#         # plt.show()
#
#         view3_image_depth_resized = cv2.resize(view3_image_depth,
#                                                (w_scale, h_scale))
#         view3_image_depth_ratio_resized_log2 = np.log2(view3_image_depth_resized/depth_center)
#
#         # build the scale-selection map:
#         scale_map_v1 = scale_center - np.round(view1_image_depth_ratio_resized_log2)
#         scale_map_v1 = np.clip(scale_map_v1, min(scale_range), max(scale_range))
#         scale_map_v2 = scale_center - np.round(view2_image_depth_ratio_resized_log2)
#         scale_map_v2 = np.clip(scale_map_v2, min(scale_range), max(scale_range))
#         scale_map_v3 = scale_center - np.round(view3_image_depth_ratio_resized_log2)
#         scale_map_v3 = np.clip(scale_map_v3, min(scale_range), max(scale_range))
#
#         scale_selection_map_v1 = np.zeros([h_scale, w_scale, scale_number])
#         scale_selection_map_v2 = np.zeros([h_scale, w_scale, scale_number])
#         scale_selection_map_v3 = np.zeros([h_scale, w_scale, scale_number])
#
#         for i in range(h_scale):
#             for j in range(w_scale):
#                 scale_sel1 = int(scale_map_v1[i, j])
#                 scale_selection_map_v1[i, j, scale_sel1] = 1
#
#                 scale_sel2 = int(scale_map_v2[i, j])
#                 scale_selection_map_v2[i, j, scale_sel2] = 1
#
#                 scale_sel3 = int(scale_map_v3[i, j])
#                 scale_selection_map_v3[i, j, scale_sel3] = 1
#
#         if view == 1:
#             scale_selection_map = scale_selection_map_v1
#         if view == 2:
#             scale_selection_map = scale_selection_map_v2
#         if view == 3:
#             scale_selection_map = scale_selection_map_v3
#
#         # fig0 = plt.figure()
#         # plt.subplot(1, 3, 1)
#         # plt.title('scale 0')
#         # plt.imshow(scale_selection_map_v1[:,:,0])
#         #
#         # plt.subplot(1, 3, 2)
#         # plt.title('scale 1')
#         # plt.imshow(scale_selection_map_v1[:,:,1])
#         #
#         # plt.subplot(1, 3, 3)
#         # plt.title('scale 2')
#         # plt.imshow(scale_selection_map_v1[:,:,2])
#         # # plt.show()
#         #
#         #
#         # fig1 = plt.figure()
#         # plt.subplot(1, 3, 1)
#         # plt.title('scale 0')
#         # plt.imshow(scale_selection_map_v2[:,:,0])
#         #
#         # plt.subplot(1, 3, 2)
#         # plt.title('scale 1')
#         # plt.imshow(scale_selection_map_v2[:,:,1])
#         #
#         # plt.subplot(1, 3, 3)
#         # plt.title('scale 2')
#         # plt.imshow(scale_selection_map_v2[:,:,2])
#         # # plt.show()
#         #
#         #
#         # fig2 = plt.figure()
#         # plt.subplot(1, 3, 1)
#         # plt.title('scale 0')
#         # plt.imshow(scale_selection_map_v3[:, :, 0])
#         #
#         # plt.subplot(1, 3, 2)
#         # plt.title('scale 1')
#         # plt.imshow(scale_selection_map_v3[:, :, 1])
#         #
#         # plt.subplot(1, 3, 3)
#         # plt.title('scale 2')
#         # plt.imshow(scale_selection_map_v3[:, :, 2])
#         # plt.show()
#
# ###################################################################################
#         # input for convolution
#
#         v_conv_add = tf.zeros([batch_size, height, width, num_channels_single])
#
#         for s in scale_range:
#             x2_s = x[:, :, :, (s + 0) * num_channels_single:(s + 1) * num_channels_single]
#
#             v2_s = scale_selection_map[:, :, s]
#             v2_s = tf.cast(v2_s, 'float32')
#             v2_s = tf.expand_dims(v2_s, axis=0)
#             v2_s = tf.expand_dims(v2_s, axis=3)
#             v2_s = tf.tile(v2_s, [batch_size, 1, 1, num_channels_single])
#
#             kx2_s_mul = tf.multiply(x2_s, v2_s)
#             v_conv_add = v_conv_add + kx2_s_mul
#
#         output = v_conv_add
#         return output