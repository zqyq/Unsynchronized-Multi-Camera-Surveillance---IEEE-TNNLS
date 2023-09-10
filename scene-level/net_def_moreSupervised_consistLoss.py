from __future__ import print_function

import keras
assert keras.__version__.startswith('2.')
from keras.models import Sequential
from keras.engine import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation,Conv2DTranspose, UpSampling2D, Reshape
from keras.layers.merge import Multiply, Add, Concatenate, Dot
from keras.regularizers import l2
from keras.initializers import Constant
from MyKerasLayers import AcrossChannelLRN
import numpy as np
from keras.layers import Lambda

import keras.backend as K

import cv2
import tensorflow as tf

# import sys
# sys.path.append('utils')
# from utils.data_utils import img_to_array, array_to_img
# from transformer import spatial_transformer_network
from spatial_transformer import SpatialTransformer
from feature_scale_fusion_layer_learnScaleSel import feature_scale_fusion_layer
from feature_scale_fusion_layer_rbm import feature_scale_fusion_layer_rbm
from UpSampling_layer import UpSampling_layer


from optical_flow_warping import optical_flow_warping
from SyncConsistLoss_layer import SyncConsistLoss_layer

# feature extraction for crowd counting
# view 1:
def feature_extraction_view1_s0(base_weight_decay,
                             input_shape,
                             trainable_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_s0'
    )(x)
    x1_1 = Activation('relu',
                      name='conv_block_1_act_s0'
                      )(x1_1)
    x1_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_norm_s0'
    )(x1_1)

    # conv block 2
    x1_2 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_s0'
    )(x1_1)
    x1_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_pool_s0'
    )(x1_2)
    x1_2 = Activation('relu',
                      name='conv_block_2_act_s0'
                      )(x1_2)
    x1_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_norm_s0'
    )(x1_2)

    # conv block 3
    x1_3 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_s0'
    )(x1_2)
    x1_3 = Activation('relu',
                      name='conv_block_3_act_s0'
                      )(x1_3)
    x1_3 = AcrossChannelLRN(
        local_size=5,


        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_norm_s0'
    )(x1_3)

    # conv block 4
    x1_4 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_s0'
    )(x1_3)
    x1_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_pool_s0'
    )(x1_4)
    x1_4 = Activation('relu', name='conv_block_4_act_s0'
                      )(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_norm_s0'
    )(x1_4)
    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view1_s0')

    # model = init_weights_vgg(model)
    return model
def feature_extraction_view1_s1(base_weight_decay,
                             input_shape,
                             trainable_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_s1'
    )(x)
    x1_1 = Activation('relu',
                      name='conv_block_1_act_s1'
                      )(x1_1)
    x1_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_norm_s1'
    )(x1_1)

    # conv block 2
    x1_2 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_s1'
    )(x1_1)
    x1_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_pool_s1'
    )(x1_2)
    x1_2 = Activation('relu',
                      name='conv_block_2_act_s1'
                      )(x1_2)
    x1_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_norm_s1'
    )(x1_2)

    # conv block 3
    x1_3 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_s1'
    )(x1_2)
    x1_3 = Activation('relu',
                      name='conv_block_3_act_s1'
                      )(x1_3)
    x1_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_norm_s1'
    )(x1_3)

    # conv block 4
    x1_4 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_s1'
    )(x1_3)
    x1_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_pool_s1'
    )(x1_4)
    x1_4 = Activation('relu', name='conv_block_4_act_s1'
                      )(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_norm_s1'
    )(x1_4)
    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view1_s1')

    # model = init_weights_vgg(model)
    return model
def feature_extraction_view1_s2(base_weight_decay,
                             input_shape,
                             trainable_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_s2'
    )(x)
    x1_1 = Activation('relu',
                      name='conv_block_1_act_s2'
                      )(x1_1)
    x1_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_norm_s2'
    )(x1_1)

    # conv block 2
    x1_2 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_s2'
    )(x1_1)
    x1_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_pool_s2'
    )(x1_2)
    x1_2 = Activation('relu',
                      name='conv_block_2_act_s2'
                      )(x1_2)
    x1_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_norm_s2'
    )(x1_2)

    # conv block 3
    x1_3 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_s2'
    )(x1_2)
    x1_3 = Activation('relu',
                      name='conv_block_3_act_s2'
                      )(x1_3)
    x1_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_norm_s2'
    )(x1_3)

    # conv block 4
    x1_4 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_s2'
    )(x1_3)
    x1_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_pool_s2'
    )(x1_4)
    x1_4 = Activation('relu', name='conv_block_4_act_s2'
                      )(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_norm_s2'
    )(x1_4)
    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view1_s2')
    # model = init_weights_vgg(model)
    return model
# view 2:
def feature_extraction_view2_s0(base_weight_decay,
                             input_shape,
                             trainable_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_2_s0'
    )(x)
    x1_1 = Activation('relu',
                      name='conv_block_1_2_act_s0'
                      )(x1_1)
    x1_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_2_norm_s0'
    )(x1_1)

    # conv block 2
    x1_2 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_2_s0'
    )(x1_1)
    x1_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_2_pool_s0'
    )(x1_2)
    x1_2 = Activation('relu',
                      name='conv_block_2_2_act_s0'
                      )(x1_2)
    x1_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_2_norm_s0'
    )(x1_2)

    # conv block 3
    x1_3 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_2_s0'
    )(x1_2)
    x1_3 = Activation('relu',
                      name='conv_block_3_2_act_s0'
                      )(x1_3)
    x1_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_2_norm_s0'
    )(x1_3)

    # conv block 4
    x1_4 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_2_s0'
    )(x1_3)
    x1_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_2_pool_s0'
    )(x1_4)
    x1_4 = Activation('relu', name='conv_block_4_2_act_s0'
                      )(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_2_norm_s0'
    )(x1_4)
    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view2_s0')

    # model = init_weights_vgg(model)
    return model
def feature_extraction_view2_s1(base_weight_decay,
                             input_shape,
                             trainable_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_2_s1'
    )(x)
    x1_1 = Activation('relu',
                      name='conv_block_1_2_act_s1'
                      )(x1_1)
    x1_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_2_norm_s1'
    )(x1_1)

    # conv block 2
    x1_2 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_2_s1'
    )(x1_1)
    x1_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_2_pool_s1'
    )(x1_2)
    x1_2 = Activation('relu',
                      name='conv_block_2_2_act_s1'
                      )(x1_2)
    x1_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_2_norm_s1'
    )(x1_2)

    # conv block 3
    x1_3 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_2_s1'
    )(x1_2)
    x1_3 = Activation('relu',
                      name='conv_block_3_2_act_s1'
                      )(x1_3)
    x1_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_2_norm_s1'
    )(x1_3)

    # conv block 4
    x1_4 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_2_s1'
    )(x1_3)
    x1_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_2_pool_s1'
    )(x1_4)
    x1_4 = Activation('relu', name='conv_block_4_2_act_s1'
                      )(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_2_norm_s1'
    )(x1_4)
    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view2_s1')
    # model = init_weights_vgg(model)
    return model
def feature_extraction_view2_s2(base_weight_decay,
                             input_shape,
                             trainable_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_2_s2'
    )(x)
    x1_1 = Activation('relu',
                      name='conv_block_1_2_act_s2'
                      )(x1_1)
    x1_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_2_norm_s2'
    )(x1_1)

    # conv block 2
    x1_2 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_2_s2'
    )(x1_1)
    x1_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_2_pool_s2'
    )(x1_2)
    x1_2 = Activation('relu',
                      name='conv_block_2_2_act_s2'
                      )(x1_2)
    x1_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_2_norm_s2'
    )(x1_2)

    # conv block 3
    x1_3 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_2_s2'
    )(x1_2)
    x1_3 = Activation('relu',
                      name='conv_block_3_2_act_s2'
                      )(x1_3)
    x1_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_2_norm_s2'
    )(x1_3)

    # conv block 4
    x1_4 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_2_s2'
    )(x1_3)
    x1_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_2_pool_s2'
    )(x1_4)
    x1_4 = Activation('relu', name='conv_block_4_2_act_s2'
                      )(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_2_norm_s2'
    )(x1_4)
    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view2_s2')

    # model = init_weights_vgg(model)
    return model
# view 3:
def feature_extraction_view3_s0(base_weight_decay,
                             input_shape,
                             trainable_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_3_s0'
    )(x)
    x1_1 = Activation('relu',
                      name='conv_block_1_3_act_s0'
                      )(x1_1)
    x1_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_3_norm_s0'
    )(x1_1)

    # conv block 2
    x1_2 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_3_s0'
    )(x1_1)
    x1_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_3_pool_s0'
    )(x1_2)
    x1_2 = Activation('relu',
                      name='conv_block_2_3_act_s0'
                      )(x1_2)
    x1_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_3_norm_s0'
    )(x1_2)

    # conv block 3
    x1_3 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_3_s0'
    )(x1_2)
    x1_3 = Activation('relu',
                      name='conv_block_3_3_act_s0'
                      )(x1_3)
    x1_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_3_norm_s0'
    )(x1_3)

    # conv block 4
    x1_4 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_3_s0'
    )(x1_3)
    x1_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_3_pool_s0'
    )(x1_4)
    x1_4 = Activation('relu', name='conv_block_4_3_act_s0'
                      )(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_3_norm_s0'
    )(x1_4)
    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view3_s0')

    # model = init_weights_vgg(model)
    return model
def feature_extraction_view3_s1(base_weight_decay,
                             input_shape,
                             trainable_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_3_s1'
    )(x)
    x1_1 = Activation('relu',
                      name='conv_block_1_3_act_s1'
                      )(x1_1)
    x1_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_3_norm_s1'
    )(x1_1)

    # conv block 2
    x1_2 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_3_s1'
    )(x1_1)
    x1_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_3_pool_s1'
    )(x1_2)
    x1_2 = Activation('relu',
                      name='conv_block_2_3_act_s1'
                      )(x1_2)
    x1_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_3_norm_s1'
    )(x1_2)

    # conv block 3
    x1_3 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_3_s1'
    )(x1_2)
    x1_3 = Activation('relu',
                      name='conv_block_3_3_act_s1'
                      )(x1_3)
    x1_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_3_norm_s1'
    )(x1_3)

    # conv block 4
    x1_4 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_3_s1'
    )(x1_3)
    x1_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_3_pool_s1'
    )(x1_4)
    x1_4 = Activation('relu', name='conv_block_4_3_act_s1'
                      )(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_3_norm_s1'
    )(x1_4)
    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view3_s1')

    # model = init_weights_vgg(model)
    return model
def feature_extraction_view3_s2(base_weight_decay,
                             input_shape,
                             trainable_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_3_s2'
    )(x)
    x1_1 = Activation('relu',
                      name='conv_block_1_3_act_s2'
                      )(x1_1)
    x1_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_3_norm_s2'
    )(x1_1)

    # conv block 2
    x1_2 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_3_s2'
    )(x1_1)
    x1_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_3_pool_s2'
    )(x1_2)
    x1_2 = Activation('relu', name='conv_block_2_3_act_s2'
                      )(x1_2)
    x1_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_3_norm_s2'
    )(x1_2)

    # conv block 3
    x1_3 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_3_s2'
    )(x1_2)
    x1_3 = Activation('relu',
                      name='conv_block_3_3_act_s2'
                      )(x1_3)
    x1_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_3_norm_s2'
    )(x1_3)

    # conv block 4
    x1_4 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_3_s2'
    )(x1_3)
    x1_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_3_pool_s2'
    )(x1_4)
    x1_4 = Activation('relu', name='conv_block_4_3_act_s2'
                      )(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_3_norm_s2'
    )(x1_4)
    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view3_s2')

    # model = init_weights_vgg(model)
    return model


def feature_extraction_view1_OP(base_weight_decay,
                             input_shape,
                             trainable_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_1'
    )(x)
    x1_1 = Activation('relu',
                      # name='conv_block_1_act'
                      )(x1_1)
    x1_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_1_norm'
    )(x1_1)

    # conv block 2
    x1_2 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_2'
    )(x1_1)
    x1_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        # name='conv_block_2_pool'
    )(x1_2)
    x1_2 = Activation('relu',
                      # name='conv_block_2_act'
                      )(x1_2)
    x1_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_2_norm'
    )(x1_2)

    # conv block 3
    x1_3 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_3'
    )(x1_2)
    x1_3 = Activation('relu',
                      # name='conv_block_3_act'
                      )(x1_3)
    x1_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_3_norm'
    )(x1_3)

    # conv block 4
    x1_4 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_4'
    )(x1_3)
    x1_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        # name='conv_block_4_pool'
    )(x1_4)
    x1_4 = Activation('relu',  # name='conv_block_4_act'
                      )(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_4_norm'
    )(x1_4)

    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view1_OP')
    # model = init_weights_vgg(model)

    return model

def feature_extraction_view2_OP(base_weight_decay,
                             input_shape,
                             trainable_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_1'
    )(x)
    x1_1 = Activation('relu',
                      # name='conv_block_1_act'
                      )(x1_1)
    x1_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_1_norm'
    )(x1_1)

    # conv block 2
    x1_2 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_2'
    )(x1_1)
    x1_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        # name='conv_block_2_pool'
    )(x1_2)
    x1_2 = Activation('relu',
                      # name='conv_block_2_act'
                      )(x1_2)
    x1_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_2_norm'
    )(x1_2)

    # conv block 3
    x1_3 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_3'
    )(x1_2)
    x1_3 = Activation('relu',
                      # name='conv_block_3_act'
                      )(x1_3)
    x1_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_3_norm'
    )(x1_3)

    # conv block 4
    x1_4 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_4'
    )(x1_3)
    x1_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        # name='conv_block_4_pool'
    )(x1_4)
    x1_4 = Activation('relu',  # name='conv_block_4_act'
                      )(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_4_norm'
    )(x1_4)

    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view2_OP')
    return model

def feature_extraction_view3_OP(base_weight_decay,
                             input_shape,
                             trainable_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_1'
    )(x)
    x1_1 = Activation('relu',
                      # name='conv_block_1_act'
                      )(x1_1)
    x1_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_1_norm'
    )(x1_1)

    # conv block 2
    x1_2 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_2'
    )(x1_1)
    x1_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        # name='conv_block_2_pool'
    )(x1_2)
    x1_2 = Activation('relu',
                      # name='conv_block_2_act'
                      )(x1_2)
    x1_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_2_norm'
    )(x1_2)

    # conv block 3
    x1_3 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_3'
    )(x1_2)
    x1_3 = Activation('relu',
                      # name='conv_block_3_act'
                      )(x1_3)
    x1_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_3_norm'
    )(x1_3)

    # conv block 4
    x1_4 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        # name='conv_block_4'
    )(x1_3)
    x1_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        # name='conv_block_4_pool'
    )(x1_4)
    x1_4 = Activation('relu',  # name='conv_block_4_act'
                      )(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        # name='conv_block_4_norm'
    )(x1_4)

    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view3_OP')
    return model


# single-view dmap output
def view1_decoder(base_weight_decay, x, trainable_flag):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None
        #name='conv_block_5'
    )(x)
    x1_5 = Activation('relu'
                      #, name='conv_block_5_act'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        #name='conv_block_6'
    )(x1_5)
    x1_6 = Activation('relu'
                      #, name='conv_block_6_act'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        #name='conv_block_7'
    )(x1_6)
    x1_7 = Activation('relu'
                      #, name='conv_block_7_act'
                      )(x1_7)
    return x1_7
def view2_decoder(base_weight_decay, x, trainable_flag):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None
        #name='conv_block_5'
    )(x)
    x1_5 = Activation('relu'
                      #, name='conv_block_5_act'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        #name='conv_block_6'
    )(x1_5)
    x1_6 = Activation('relu'
                      #, name='conv_block_6_act'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        #name='conv_block_7'
    )(x1_6)
    x1_7 = Activation('relu'
                      #, name='conv_block_7_act'
                      )(x1_7)
    return x1_7
def view3_decoder(base_weight_decay, x, trainable_flag):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None
        #name='conv_block_5'
    )(x)
    x1_5 = Activation('relu'
                      #, name='conv_block_5_act'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        #name='conv_block_6'
    )(x1_5)
    x1_6 = Activation('relu'
                      #, name='conv_block_6_act'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        #name='conv_block_7'
    )(x1_6)
    x1_7 = Activation('relu'
                      #, name='conv_block_7_act'
                      )(x1_7)
    return x1_7

# fusion conv
def fusion_conv_v1(base_weight_decay, x):
    x1_02 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer= l2(base_weight_decay),
        use_bias=True,
        activation=None,
        #name='conv_block_1'
    )(x)
    x1_02 = Activation('relu')(x1_02)
    return  x1_02
def fusion_conv_v2(base_weight_decay, x):
    x1_02 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer= l2(base_weight_decay),
        use_bias=True,
        activation=None,
        #name='conv_block_1'
    )(x)
    x1_02 = Activation('relu')(x1_02)
    return  x1_02
def fusion_conv_v3(base_weight_decay, x):
    x1_03 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer= l2(base_weight_decay),
        use_bias=True,
        activation=None,
        #name='conv_block_1'
    )(x)
    x1_03 = Activation('relu')(x1_03)
    return  x1_03





def optical_flow_estimation_w12(base_weight_decay, input_shape,
                             trainable_flag):


    x = Input(batch_shape=input_shape)

    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=128,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='w12_conv_block_5'
    )(x)
    x1_5 = Activation('relu'
                      , name='w12_conv_block_5_act'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=128,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='w12_conv_block_6'
    )(x1_5)
    x1_6 = Activation('relu'
                      , name='w12_conv_block_6_act'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='w12_conv_block_7'
    )(x1_6)
    x1_7 = Activation('relu'
                      , name='w12_conv_block_7_act'
                      )(x1_7)

    # conv block 8
    x1_8 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='w12_conv_block_8'
    )(x1_7)
    x1_8 = Activation('relu'
                      , name='w12_conv_block_8_act'
                      )(x1_8)

    # conv block 9
    x1_9 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='w12_conv_block_9'
    )(x1_8)
    x1_9 = Activation('relu'
                      , name='w12_conv_block_9_act'
                      )(x1_9)

    # conv block 7
    x1_10 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=2,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='w12_conv_block_10'
    )(x1_9)
    # x1_10 = Activation('relu'
    #                   # , name='conv_block_7_act'
    #                   )(x1_10)
    model = Model(inputs=[x], outputs=[x1_10], name = 'optical_flow_estimation_w12')

    return model

def optical_flow_estimation_w13(base_weight_decay, input_shape,
                             trainable_flag):
    x = Input(batch_shape=input_shape)

    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=128,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='w13_conv_block_5'
    )(x)
    x1_5 = Activation('relu'
                      , name='w13_conv_block_5_act'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=128,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='w13_conv_block_6'
    )(x1_5)
    x1_6 = Activation('relu'
                      , name='w13_conv_block_6_act'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='w13_conv_block_7'
    )(x1_6)
    x1_7 = Activation('relu'
                      , name='w13_conv_block_7_act'
                      )(x1_7)

    # conv block 8
    x1_8 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='w13_conv_block_8'
    )(x1_7)
    x1_8 = Activation('relu'
                      , name='w13_conv_block_8_act'
                      )(x1_8)

    # conv block 9
    x1_9 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='w13_conv_block_9'
    )(x1_8)
    x1_9 = Activation('relu'
                      , name='w13_conv_block_9_act'
                      )(x1_9)

    # conv block 7
    x1_10 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=2,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='w13_conv_block_10'
    )(x1_9)
    # x1_10 = Activation('relu'
    #                   # , name='conv_block_7_act'
    #                   )(x1_10)
    model = Model(inputs=[x], outputs=[x1_10], name = 'optical_flow_estimation_w13')

    return model



def scale_selection_mask(base_weight_decay, input_depth_maps):
    view1_scale = Conv2D(
        data_format='channels_last',
        trainable=False,
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer=Constant(value=-1),
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        bias_initializer='ones',
        activation=None
        #name='scale_fusion1'
    )(input_depth_maps)
    #view1_scale_mask = Activation('softmax')(view1_scale)
    return  view1_scale



def multi_view_fusion_decoder(base_weight_decay,
                             input_shape,
                             trainable_flag):
    x_input = Input(batch_shape=input_shape)

    fusion_v123 = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=96,
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='scale_fusion'
    )(x_input)
    fusion_v123 = Activation('relu', name='scale_fusion_act')(fusion_v123)

    # conv block 9
    x = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_fusion1'
    )(fusion_v123)
    x = Activation('relu', name='conv_block_fusion1_act')(x)

    x = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_fusion2'
    )(x)
    x = Activation('relu', name='conv_block_fusion2_act')(x)

    x = Conv2D(
        data_format='channels_last',
        trainable=trainable_flag,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_fusion3'
    )(x)
    x_output = Activation('relu', name='conv_block_fusion3_act')(x)

    model = Model(inputs=[x_input], outputs=[x_output], name = 'multi_view_fusion_decoder')
    # model = init_weights_vgg(model)
    return model







######################## main structure #######################################


def build_model_FCN_model_api(batch_size,
                              optimizer,
                              patch_size = (128, 128),
                              base_weight_decay = 0.0005,
                              output_ROI_mask = True):
    print('Using build_model_FCN_model_api')

    # define the shared model: 
    net_name = 'Multi-view_FCN'

    scale_number = 3
    ##################### input  ###############################################
    input_shape0 = (batch_size, patch_size[0],   patch_size[1],   1)
    input_shape1 = (batch_size, int(patch_size[0]/2), int(patch_size[1]/2), 1)
    input_shape2 = (batch_size, int(patch_size[0]/4), int(patch_size[1]/4), 1)

    input_shape3 = (1, int(patch_size[0]/4), int(patch_size[1]/4), 1)

    input_shape4 = (batch_size,  int(710/4),  int(610/4), 64)

    input_shape5 = (batch_size,  int(710/4),  int(610/4), 96)

    input_patches1_s0 = Input(batch_shape = input_shape0, name='patches1_s0')
    input_patches1_s1 = Input(batch_shape = input_shape1, name='patches1_s1')
    input_patches1_s2 = Input(batch_shape = input_shape2, name='patches1_s2')

    # input_patches2_s0 = Input(batch_shape = input_shape0, name='patches2_s0')
    # input_patches2_s1 = Input(batch_shape = input_shape1, name='patches2_s1')
    # input_patches2_s2 = Input(batch_shape = input_shape2, name='patches2_s2')
    #
    # input_patches3_s0 = Input(batch_shape = input_shape0, name='patches3_s0')
    # input_patches3_s1 = Input(batch_shape = input_shape1, name='patches3_s1')
    # input_patches3_s2 = Input(batch_shape = input_shape2, name='patches3_s2')

    input_depth_maps_v1 = Input(batch_shape = input_shape3, name='depth_ratio_v1')
    input_depth_maps_v2 = Input(batch_shape = input_shape3, name='depth_ratio_v2')
    input_depth_maps_v3 = Input(batch_shape = input_shape3, name='depth_ratio_v3')

    input_patches2_u_s0 = Input(batch_shape = input_shape0, name='patches2_u_s0')
    input_patches2_u_s1 = Input(batch_shape = input_shape1, name='patches2_u_s1')
    input_patches2_u_s2 = Input(batch_shape = input_shape2, name='patches2_u_s2')

    input_patches3_u_s0 = Input(batch_shape = input_shape0, name='patches3_u_s0')
    input_patches3_u_s1 = Input(batch_shape = input_shape1, name='patches3_u_s1')
    input_patches3_u_s2 = Input(batch_shape = input_shape2, name='patches3_u_s2')



    if output_ROI_mask:
        # the output density patch/map is down-sampled by a factor of 4
        output_masks = Input(batch_shape=(batch_size, patch_size[0], patch_size[1], 1),
                             name='output_masks')

    trainable_flag = True
    encoder_view1_s0 = feature_extraction_view1_s0(base_weight_decay,
                                             input_shape1,
                                             trainable_flag)
    encoder_view1_s1 = feature_extraction_view1_s1(base_weight_decay,
                                             input_shape2,
                                             trainable_flag)
    encoder_view1_s2 = feature_extraction_view1_s2(base_weight_decay,
                                             input_shape3,
                                             trainable_flag)

    encoder_view2_s0 = feature_extraction_view2_s0(base_weight_decay,
                                             input_shape1,
                                             trainable_flag)
    encoder_view2_s1 = feature_extraction_view2_s1(base_weight_decay,
                                             input_shape2,
                                             trainable_flag)
    encoder_view2_s2 = feature_extraction_view2_s2(base_weight_decay,
                                             input_shape3,
                                             trainable_flag)

    encoder_view3_s0 = feature_extraction_view3_s0(base_weight_decay,
                                             input_shape1,
                                             trainable_flag)
    encoder_view3_s1 = feature_extraction_view3_s1(base_weight_decay,
                                             input_shape2,
                                             trainable_flag)
    encoder_view3_s2 = feature_extraction_view3_s2(base_weight_decay,
                                             input_shape3,
                                             trainable_flag)


    encoder_view1_OP = feature_extraction_view1_OP(base_weight_decay,
                                             input_shape1,
                                             trainable_flag)
    encoder_view2_OP = feature_extraction_view2_OP(base_weight_decay,
                                             input_shape2,
                                             trainable_flag)
    encoder_view3_OP = feature_extraction_view3_OP(base_weight_decay,
                                             input_shape3,
                                             trainable_flag)

    estimate_OP_w12 = optical_flow_estimation_w12(base_weight_decay,
                                                  input_shape4,
                                                  trainable_flag)
    estimate_OP_w13 = optical_flow_estimation_w13(base_weight_decay,
                                                  input_shape4,
                                                  trainable_flag)

    multi_view_decoder = multi_view_fusion_decoder(base_weight_decay,
                                                   input_shape5,
                                                   trainable_flag)


    ####################### view 1 #############################################
    # image pyramids:
    x1_s0_output = encoder_view1_s0(input_patches1_s0)
    x1_s1_output = encoder_view1_s1(input_patches1_s1)
    x1_s2_output = encoder_view1_s2(input_patches1_s2)

    x2_u_s0_output = encoder_view2_s0(input_patches2_u_s0)
    x2_u_s1_output = encoder_view2_s1(input_patches2_u_s1)
    x2_u_s2_output = encoder_view2_s2(input_patches2_u_s2)

    x3_u_s0_output = encoder_view3_s0(input_patches3_u_s0)
    x3_u_s1_output = encoder_view3_s1(input_patches3_u_s1)
    x3_u_s2_output = encoder_view3_s2(input_patches3_u_s2)

    # decoder:
    x1_7_s0 = view1_decoder(base_weight_decay, x1_s0_output, trainable_flag)
    x1_7_s1 = view1_decoder(base_weight_decay, x1_s1_output, trainable_flag)
    x1_7_s2 = view1_decoder(base_weight_decay, x1_s2_output, trainable_flag)

    x2_7_s0 = view2_decoder(base_weight_decay, x2_u_s0_output, trainable_flag)
    x2_7_s1 = view2_decoder(base_weight_decay, x2_u_s1_output, trainable_flag)
    x2_7_s2 = view2_decoder(base_weight_decay, x2_u_s2_output, trainable_flag)

    x3_7_s0 = view3_decoder(base_weight_decay, x3_u_s0_output, trainable_flag)
    x3_7_s1 = view3_decoder(base_weight_decay, x3_u_s1_output, trainable_flag)
    x3_7_s2 = view3_decoder(base_weight_decay, x3_u_s2_output, trainable_flag)


    #################################### fusion #############################################
    ################# get the scale-selection mask #####################
    # view depth of image
    batch_size = x1_s0_output.shape[0].value
    height = x1_s0_output.shape[1].value
    width = x1_s0_output.shape[2].value
    num_channels = x1_s0_output.shape[3].value
    output_shape = [1, height, width, 1]

    view1_scale = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer=Constant(value=-1),
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        bias_initializer='ones',
        activation=None,
        name='scale_fusion1'
    )(input_depth_maps_v1)

    view2_scale = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer=Constant(value=-1),
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        bias_initializer='ones',
        activation=None,
        name='scale_fusion2'
    )(input_depth_maps_v2)

    view3_scale = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        kernel_initializer=Constant(value=-1),
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        bias_initializer='ones',
        activation=None,
        name='scale_fusion3'
    )(input_depth_maps_v3)

    view1_scale_mask = feature_scale_fusion_layer_rbm(scale_number=scale_number)(view1_scale)
    view2_scale_mask = feature_scale_fusion_layer_rbm(scale_number=scale_number)(view2_scale)
    view3_scale_mask = feature_scale_fusion_layer_rbm(scale_number=scale_number)(view3_scale)


    #################### fusion with mask ##################
    # view 1
    ## conv
    x1_s0_output_fusion = fusion_conv_v1(base_weight_decay, x1_s0_output)
    x1_s1_output_fusion = fusion_conv_v1(base_weight_decay, x1_s1_output)
    x1_s2_output_fusion = fusion_conv_v1(base_weight_decay, x1_s2_output)

    ## up sampling
    x1_s1_output_fusion = UpSampling_layer(size=[height, width])([x1_s1_output_fusion])
    x1_s2_output_fusion = UpSampling_layer(size=[height, width])([x1_s2_output_fusion])

    concatenated_map_v1 = Concatenate(name='cat_map_v1')(
        [x1_s0_output_fusion, x1_s1_output_fusion, x1_s2_output_fusion])
    fusion_v1 = feature_scale_fusion_layer(scale_number=scale_number)([concatenated_map_v1, view1_scale_mask])

    ## proj
    fusion_v1_proj = SpatialTransformer(1, [int(710/4),  int(610/4)])(fusion_v1)

    ############################ unsyced input for crowd counting: ###############

    # unsyced input: view 2
    ## conv
    x2_u_s0_output_fusion = fusion_conv_v2(base_weight_decay, x2_u_s0_output)
    x2_u_s1_output_fusion = fusion_conv_v2(base_weight_decay, x2_u_s1_output)
    x2_u_s2_output_fusion = fusion_conv_v2(base_weight_decay, x2_u_s2_output)

    ## up sampling
    x2_u_s1_output_fusion = UpSampling_layer(size=[height, width])([x2_u_s1_output_fusion])
    x2_u_s2_output_fusion = UpSampling_layer(size=[height, width])([x2_u_s2_output_fusion])

    concatenated_map_v2_u = Concatenate(name='cat_map_v2_u')(
        [x2_u_s0_output_fusion, x2_u_s1_output_fusion, x2_u_s2_output_fusion])

    fusion_v2_u = feature_scale_fusion_layer(scale_number=scale_number)([concatenated_map_v2_u,
                                                                         view2_scale_mask])
    ## proj
    fusion_v2_u_proj = SpatialTransformer(2, [int(710/4),  int(610/4)])(fusion_v2_u)

    # unsyced input: view 3
    ## conv
    x3_u_s0_output_fusion = fusion_conv_v3(base_weight_decay, x3_u_s0_output)
    x3_u_s1_output_fusion = fusion_conv_v3(base_weight_decay, x3_u_s1_output)
    x3_u_s2_output_fusion = fusion_conv_v3(base_weight_decay, x3_u_s2_output)

    ## up sampling
    x3_u_s1_output_fusion = UpSampling_layer(size=[height, width])([x3_u_s1_output_fusion])
    x3_u_s2_output_fusion = UpSampling_layer(size=[height, width])([x3_u_s2_output_fusion])

    concatenated_map_v3_u = Concatenate(name='cat_map_v3_u')(
        [x3_u_s0_output_fusion, x3_u_s1_output_fusion, x3_u_s2_output_fusion])

    fusion_v3_u = feature_scale_fusion_layer(scale_number=scale_number)([concatenated_map_v3_u,
                                                                         view3_scale_mask])

    ## proj
    fusion_v3_u_proj = SpatialTransformer(3, [int(710/4),  int(610/4)])(fusion_v3_u)




    ############################## optical flow estimation and warping  #####################
    # view 1:
    x1_4_OP = encoder_view1_OP(input_patches1_s0)
    x1_4_proj_OP = SpatialTransformer(1, [int(710/4),  int(610/4)])(x1_4_OP)

    # view 2:
    # optical flow
    # x2_4_OP = encoder_view2_OP(input_patches2_s0)
    x2_4_u_OP = encoder_view2_OP(input_patches2_u_s0)

    # x2_4_proj_OP = SpatialTransformer(2, [710 / 4, 610 / 4])(x2_4_OP)
    x2_4_u_proj_OP = SpatialTransformer(2, [int(710/4),  int(610/4)])(x2_4_u_OP)

    # x12_OP_cat_0 = Concatenate(name='x12_OP_cat_0')([x1_4_proj_OP, x2_4_proj_OP])
    # w12_0 = estimate_OP_w12(x12_OP_cat_0)
    x12_OP_cat = Concatenate(name='x12_OP_cat')([x1_4_proj_OP, x2_4_u_proj_OP])
    w12 = estimate_OP_w12(x12_OP_cat)

    # x2_4_proj_est_0 = optical_flow_warping(view=2, name='w12_0')([w12_0, fusion_v2_proj])
    fusion_v2_proj_est = optical_flow_warping(view=2, name='w12')([w12, fusion_v2_u_proj])

    # view 3:
    # optical flow
    # x3_4_OP = encoder_view3_OP(input_patches3_s0)
    x3_4_u_OP = encoder_view3_OP(input_patches3_u_s0)

    # x3_4_proj_OP = SpatialTransformer(3, [710 / 4, 610 / 4])(x3_4_OP)
    x3_4_u_proj_OP = SpatialTransformer(3, [int(710/4),  int(610/4)])(x3_4_u_OP)

    # x13_OP_cat_0 = Concatenate(name='x13_OP_cat_0')([x1_4_proj_OP, x3_4_proj_OP])
    # w13_0 = estimate_OP_w13(x13_OP_cat_0)
    x13_OP_cat = Concatenate(name='x13_OP_cat')([x1_4_proj_OP, x3_4_u_proj_OP])
    w13 = estimate_OP_w13(x13_OP_cat)

    # x3_4_proj_est_0 = optical_flow_warping(view=3, name='w13_0')([w13_0, fusion_v3_proj])
    fusion_v3_proj_est = optical_flow_warping(view=3, name='w13')([w13, fusion_v3_u_proj])



    ################# concatenate ################
    concatenated_map = Concatenate(name='cat_map_fusion')([fusion_v1_proj,
                                                           fusion_v2_proj_est,
                                                           fusion_v3_proj_est])
    x_output = multi_view_decoder(concatenated_map)

    l12 = SyncConsistLoss_layer()([fusion_v1_proj, fusion_v2_u_proj])
    l13 = SyncConsistLoss_layer()([fusion_v1_proj, fusion_v3_u_proj])

    if output_ROI_mask:
        rgr_output = 'den_map_roi'
        output = Multiply(name=rgr_output)([x_output, output_masks])
        print('Layer name of regression output: {}'.format(rgr_output))
        model = Model(inputs = [input_patches1_s0, input_patches1_s1, input_patches1_s2,
                                # input_patches2_s0, input_patches2_s1, input_patches2_s2,
                                # input_patches3_s0, input_patches3_s1, input_patches3_s2,
                                input_patches2_u_s0, input_patches2_u_s1, input_patches2_u_s2,
                                input_patches3_u_s0, input_patches3_u_s1, input_patches3_u_s2,
                                input_depth_maps_v1, input_depth_maps_v2, input_depth_maps_v3,
                                output_masks],
                      outputs = [x1_7_s0, x1_7_s1, x1_7_s2,
                                 x2_7_s0, x2_7_s1, x2_7_s2,
                                 x3_7_s0, x3_7_s1, x3_7_s2,
                                 x_output,
                                 l12, l13],  # , x_output
                      name=net_name)
    else:
        model = Model(inputs = [input_patches1_s0, input_patches1_s1, input_patches1_s2,
                                # input_patches2_s0, input_patches2_s1, input_patches2_s2,
                                # input_patches3_s0, input_patches3_s1, input_patches3_s2,
                                input_patches2_u_s0, input_patches2_u_s1, input_patches2_u_s2,
                                input_patches3_u_s0, input_patches3_u_s1, input_patches3_u_s2,
                                input_depth_maps_v1, input_depth_maps_v2, input_depth_maps_v3],
                      outputs = [x1_7_s0, x1_7_s1, x1_7_s2,
                                 x2_7_s0, x2_7_s1, x2_7_s2,
                                 x3_7_s0, x3_7_s1, x3_7_s2,
                                 x_output,
                                 l12, l13], #, x_output
                      name = net_name+'overall')

    def zero_loss(y_true, y_pred):
        return K.mean(y_pred)

    model.compile(optimizer=optimizer,
                  loss=['mse','mse','mse',
                        'mse','mse','mse',
                        'mse','mse','mse',
                        'mse',
                        zero_loss, zero_loss],
                  loss_weights=[0.01, 0.001, 0.00001,
                                0.01, 0.001, 0.00001,
                                0.01, 0.001, 0.00001,
                                1,
                                1, 1]) # [0, 0, 0]
                  # train the single-view counting first, then the sync and multi-view counting.

    return model