from __future__ import print_function

import keras
assert keras.__version__.startswith('2.')
from keras.models import Sequential
from keras.engine import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation,Conv2DTranspose, UpSampling2D, Reshape
from keras.layers.merge import Multiply, Add, Concatenate, Subtract
from keras.regularizers import l2
from keras.initializers import Constant
from MyKerasLayers import AcrossChannelLRN
import numpy as np
from keras.layers import Lambda



from spatial_transformer import SpatialTransformer
from feature_scale_fusion_layer_learnScaleSel import feature_scale_fusion_layer
from feature_scale_fusion_layer_rbm import feature_scale_fusion_layer_rbm
from UpSampling_layer import UpSampling_layer

from optical_flow_warping import optical_flow_warping

from Correlation_Layer import Correlation_Layer
from ReduceSum_layer import ReduceSum_layer

# feature extraction for optical flow estimation:
# view 1
def feature_extraction_view1_s0(base_weight_decay, input_shape, train_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer= l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_s0'
    )(x)
    x1_1 = Activation('relu',name='conv_block_1_act_s0')(x1_1)
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
        trainable=train_flag,
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
    x1_2 = Activation('relu',name='conv_block_2_act_s0')(x1_2)
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
        trainable=train_flag,
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
    x1_3 = Activation('relu', name='conv_block_3_act_s0')(x1_3)
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
        trainable=train_flag,
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
    x1_4 = Activation('relu', name='conv_block_4_act_s0')(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_norm_s0'
    )(x1_4)
    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view1_s0')
    return model

def feature_extraction_view1_s1(base_weight_decay, input_shape, train_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer= l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_s1'
    )(x)
    x1_1 = Activation('relu',name='conv_block_1_act_s1')(x1_1)
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
        trainable=train_flag,
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
    x1_2 = Activation('relu',name='conv_block_2_act_s1')(x1_2)
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
        trainable=train_flag,
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
    x1_3 = Activation('relu', name='conv_block_3_act_s1')(x1_3)
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
        trainable=train_flag,
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
    x1_4 = Activation('relu', name='conv_block_4_act_s1')(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_norm_s1'
    )(x1_4)
    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view1_s1')

    return model

def feature_extraction_view1_s2(base_weight_decay, input_shape, train_flag):
    x = Input(batch_shape=input_shape)

    x1_1 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer= l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_s2'
    )(x)
    x1_1 = Activation('relu',name='conv_block_1_act_s2')(x1_1)
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
        trainable=train_flag,
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
    x1_2 = Activation('relu',name='conv_block_2_act_s2')(x1_2)
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
        trainable=train_flag,
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
    x1_3 = Activation('relu', name='conv_block_3_act_s2')(x1_3)
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
        trainable=train_flag,
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
    x1_4 = Activation('relu', name='conv_block_4_act_s2')(x1_4)
    x1_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_norm_s2'
    )(x1_4)
    model = Model(inputs=[x], outputs=[x1_4], name = 'feature_extraction_view1_s2')

    return model

# view 2
def feature_extraction_view2_s0(base_weight_decay, input_shape, train_flag):
    x = Input(batch_shape=input_shape)

    x2_1 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer= l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_2_s0'
    )(x)
    x2_1 = Activation('relu', name='conv_block_1_act_2_s0')(x2_1)
    x2_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_norm_2_s0'
    )(x2_1)

    # conv block 2
    x2_2 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_2_s0'
    )(x2_1)
    x2_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_pool_2_s0'
    )(x2_2)
    x2_2 = Activation('relu', name='conv_block_2_act_2_s0')(x2_2)
    x2_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_norm_2_s0'
        )(x2_2)

    # conv block 3
    x2_3 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_2_s0'
    )(x2_2)
    x2_3 = Activation('relu', name='conv_block_3_act_2_s0')(x2_3)
    x2_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_norm_2_s0'
    )(x2_3)

    # conv block 4
    x2_4 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_2_s0'
    )(x2_3)
    x2_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_pool_2_s0'
    )(x2_4)
    x2_4 = Activation('relu', name='conv_block_4_act_2_s0')(x2_4)
    x2_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_norm_2_s0'
    )(x2_4)
    model = Model(inputs=[x], outputs=[x2_4], name = 'feature_extraction_view2_s0')

    return model

def feature_extraction_view2_s1(base_weight_decay, input_shape, train_flag):
    x = Input(batch_shape=input_shape)

    x2_1 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer= l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_2_s1'
    )(x)
    x2_1 = Activation('relu', name='conv_block_1_act_2_s1')(x2_1)
    x2_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_norm_2_s1'
    )(x2_1)

    # conv block 2
    x2_2 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_2_s1'
    )(x2_1)
    x2_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_pool_2_s1'
    )(x2_2)
    x2_2 = Activation('relu', name='conv_block_2_act_2_s1')(x2_2)
    x2_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_norm_2_s1'
        )(x2_2)

    # conv block 3
    x2_3 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_2_s1'
    )(x2_2)
    x2_3 = Activation('relu', name='conv_block_3_act_2_s1')(x2_3)
    x2_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_norm_2_s1'
    )(x2_3)

    # conv block 4
    x2_4 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_2_s1'
    )(x2_3)
    x2_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_pool_2_s1'
    )(x2_4)
    x2_4 = Activation('relu', name='conv_block_4_act_2_s1')(x2_4)
    x2_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_norm_2_s1'
    )(x2_4)
    model = Model(inputs=[x], outputs=[x2_4], name = 'feature_extraction_view2_s1')
    return model

def feature_extraction_view2_s2(base_weight_decay, input_shape, train_flag):
    x = Input(batch_shape=input_shape)

    x2_1 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer= l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_2_s2'
    )(x)
    x2_1 = Activation('relu', name='conv_block_1_act_2_s2')(x2_1)
    x2_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_norm_2_s2'
    )(x2_1)

    # conv block 2
    x2_2 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_2_s2'
    )(x2_1)
    x2_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_pool_2_s2'
    )(x2_2)
    x2_2 = Activation('relu', name='conv_block_2_act_2_s2')(x2_2)
    x2_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_norm_2_s2'
        )(x2_2)

    # conv block 3
    x2_3 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_2_s2'
    )(x2_2)
    x2_3 = Activation('relu', name='conv_block_3_act_2_s2')(x2_3)
    x2_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_norm_2_s2'
    )(x2_3)

    # conv block 4
    x2_4 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_2_s2'
    )(x2_3)
    x2_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_pool_2_s2'
    )(x2_4)
    x2_4 = Activation('relu', name='conv_block_4_act_2_s2')(x2_4)
    x2_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_norm_2_s2'
    )(x2_4)
    model = Model(inputs=[x], outputs=[x2_4], name = 'feature_extraction_view2_s2')

    return model


# view 3
def feature_extraction_view3_s0(base_weight_decay, input_shape, train_flag):
    x = Input(batch_shape=input_shape)

    x3_1 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer= l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_3_s0'
    )(x)
    x3_1 = Activation('relu', name='conv_block_1_act_3_s0')(x3_1)
    x3_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_norm_3_s0'
    )(x3_1)

    # conv block 2
    x3_2 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_3_s0'
    )(x3_1)
    x3_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_pool_3_s0'
    )(x3_2)
    x3_2 = Activation('relu', name='conv_block_2_act_3_s0')(x3_2)
    x3_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_norm_3_s0'
        )(x3_2)

    # conv block 3
    x3_3 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_3_s0'
    )(x3_2)
    x3_3 = Activation('relu', name='conv_block_3_act_3_s0')(x3_3)
    x3_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_norm_3_s0'
    )(x3_3)

    # conv block 4
    x3_4 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_3_s0'
    )(x3_3)
    x3_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_pool_3_s0'
    )(x3_4)
    x3_4 = Activation('relu', name='conv_block_4_act_3_s0')(x3_4)
    x3_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_norm_3_s0'
    )(x3_4)
    model = Model(inputs=[x], outputs=[x3_4], name = 'feature_extraction_view3_s0')

    return model

def feature_extraction_view3_s1(base_weight_decay, input_shape, train_flag):
    x = Input(batch_shape=input_shape)

    x3_1 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer= l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_3_s1'
    )(x)
    x3_1 = Activation('relu', name='conv_block_1_act_3_s1')(x3_1)
    x3_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_norm_3_s1'
    )(x3_1)

    # conv block 2
    x3_2 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_3_s1'
    )(x3_1)
    x3_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_pool_3_s1'
    )(x3_2)
    x3_2 = Activation('relu', name='conv_block_2_act_3_s1')(x3_2)
    x3_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_norm_3_s1'
        )(x3_2)

    # conv block 3
    x3_3 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_3_s1'
    )(x3_2)
    x3_3 = Activation('relu', name='conv_block_3_act_3_s1')(x3_3)
    x3_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_norm_3_s1'
    )(x3_3)

    # conv block 4
    x3_4 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_3_s1'
    )(x3_3)
    x3_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_pool_3_s1'
    )(x3_4)
    x3_4 = Activation('relu', name='conv_block_4_act_3_s1')(x3_4)
    x3_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_norm_3_s1'
    )(x3_4)
    model = Model(inputs=[x], outputs=[x3_4], name = 'feature_extraction_view3_s1')

    return model

def feature_extraction_view3_s2(base_weight_decay, input_shape, train_flag):
    x = Input(batch_shape=input_shape)

    x3_1 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer= l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_1_3_s2'
    )(x)
    x3_1 = Activation('relu', name='conv_block_1_act_3_s2')(x3_1)
    x3_1 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_1_norm_3_s2'
    )(x3_1)

    # conv block 2
    x3_2 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_2_3_s2'
    )(x3_1)
    x3_2 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_2_pool_3_s2'
    )(x3_2)
    x3_2 = Activation('relu', name='conv_block_2_act_3_s2')(x3_2)
    x3_2 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_2_norm_3_s2'
        )(x3_2)

    # conv block 3
    x3_3 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_3_3_s2'
    )(x3_2)
    x3_3 = Activation('relu', name='conv_block_3_act_3_s2')(x3_3)
    x3_3 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_3_norm_3_s2'
    )(x3_3)

    # conv block 4
    x3_4 = Conv2D(
        data_format='channels_last',
        trainable=train_flag,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_4_3_s2'
    )(x3_3)
    x3_4 = MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
        name='conv_block_4_pool_3_s2'
    )(x3_4)
    x3_4 = Activation('relu', name='conv_block_4_act_3_s2')(x3_4)
    x3_4 = AcrossChannelLRN(
        local_size=5,
        alpha=0.01,
        beta=0.75,
        k=1,
        name='conv_block_4_norm_3_s2'
    )(x3_4)
    model = Model(inputs=[x], outputs=[x3_4], name = 'feature_extraction_view3_s2')

    return model



# feature extraction for optical flow estimation
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
    # model = init_weights_vgg(model)
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
    # model = init_weights_vgg(model)
    return model



######################## optical flow estimation #######################################
# optical flow estimation:
def optical_flow_estimation_w12(base_weight_decay, x, trainable_flag):
    # concate = Concatenate()([x1_4_proj, x2_4_u_proj])

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
        #name='w12_conv_block_5'
    )(x)
    x1_5 = Activation('tanh'
                      #, name='w12_conv_block_5_act'
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
        #name='w12_conv_block_6'
    )(x1_5)
    x1_6 = Activation('tanh'
                      #, name='w12_conv_block_6_act'
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
        #name='w12_conv_block_7'
    )(x1_6)
    x1_7 = Activation('tanh'
                      #, name='w12_conv_block_7_act'
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
        #name='w12_conv_block_8'
    )(x1_7)
    x1_8 = Activation('tanh'
                      #, name='w12_conv_block_8_act'
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
        #name='w12_conv_block_9'
    )(x1_8)
    x1_9 = Activation('tanh'
                      #, name='w12_conv_block_9_act'
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
        #name='w12_conv_block_10'
    )(x1_9)
    # x1_10 = Activation('relu'
    #                   # , name='conv_block_7_act'
    #                   )(x1_10)
    return x1_10

def optical_flow_estimation_w13(base_weight_decay, x, trainable_flag):
    # concate = Concatenate()([x1_4_proj, x2_4_u_proj])

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
        # name='w13_conv_block_5'
    )(x)
    x1_5 = Activation('tanh'
                      #, name='w13_conv_block_5_act'
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
        # name='w13_conv_block_6'
    )(x1_5)
    x1_6 = Activation('tanh'
                      #, name='w13_conv_block_6_act'
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
        # name='w13_conv_block_7'
    )(x1_6)
    x1_7 = Activation('tanh'
                      #, name='w13_conv_block_7_act'
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
        #name='w13_conv_block_8'
    )(x1_7)
    x1_8 = Activation('tanh'
                      #, name='w13_conv_block_8_act'
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
        #name='w13_conv_block_9'
    )(x1_8)
    x1_9 = Activation('tanh'
                      #, name='w13_conv_block_9_act'
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
        # name='w13_conv_block_10'
    )(x1_9)
    # x1_10 = Activation('relu'
    #                   # , name='conv_block_7_act'
    #                   )(x1_10)
    return x1_10



# single-view dmap output
def view1_decoder_s0(base_weight_decay, x):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_5_1_s0'
    )(x)
    x1_5 = Activation('relu'
                      , name='conv_block_5_act_1_s0'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_6_1_s0'
    )(x1_5)
    x1_6 = Activation('relu'
                      , name='conv_block_6_act_1_s0'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_7_1_s0'
    )(x1_6)
    x1_7 = Activation('relu'
                      , name='conv_block_7_act_1_s0'
                      )(x1_7)
    return x1_7
def view1_decoder_s1(base_weight_decay, x):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_5_1_s1'
    )(x)
    x1_5 = Activation('relu'
                      , name='conv_block_5_act_1_s1'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_6_1_s1'
    )(x1_5)
    x1_6 = Activation('relu'
                      , name='conv_block_6_act_1_s1'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_7_1_s1'
    )(x1_6)
    x1_7 = Activation('relu'
                      , name='conv_block_7_act_1_s1'
                      )(x1_7)
    return x1_7
def view1_decoder_s2(base_weight_decay, x):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_5_1_s2'
    )(x)
    x1_5 = Activation('relu'
                      , name='conv_block_5_act_1_s2'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_6_1_s2'
    )(x1_5)
    x1_6 = Activation('relu'
                      , name='conv_block_6_act_1_s2'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_7_1_s2'
    )(x1_6)
    x1_7 = Activation('relu'
                      , name='conv_block_7_act_1_s2'
                      )(x1_7)
    return x1_7


def view2_decoder_s0(base_weight_decay, x):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_5_2_s0'
    )(x)
    x1_5 = Activation('relu'
                      , name='conv_block_5_act_2_s0'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_6_2_s0'
    )(x1_5)
    x1_6 = Activation('relu'
                      , name='conv_block_6_act_2_s0'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_7_2_s0'
    )(x1_6)
    x1_7 = Activation('relu', name='conv_block_7_act_2_s0')(x1_7)
    return x1_7
def view2_decoder_s1(base_weight_decay, x):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_5_2_s1'
    )(x)
    x1_5 = Activation('relu'
                      , name='conv_block_5_act_2_s1'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_6_2_s1'
    )(x1_5)
    x1_6 = Activation('relu'
                      , name='conv_block_6_act_2_s1'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_7_2_s1'
    )(x1_6)
    x1_7 = Activation('relu', name='conv_block_7_act_2_s1')(x1_7)
    return x1_7
def view2_decoder_s2(base_weight_decay, x):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_5_2_s2'
    )(x)
    x1_5 = Activation('relu'
                      , name='conv_block_5_act_2_s2'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_6_2_s2'
    )(x1_5)
    x1_6 = Activation('relu', name='conv_block_6_act_2_s2')(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_7_2_s2'
    )(x1_6)
    x1_7 = Activation('relu', name='conv_block_7_act_2_s2')(x1_7)
    return x1_7


def view3_decoder_s0(base_weight_decay, x):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_5_3_s0'
    )(x)
    x1_5 = Activation('relu'
                      , name='conv_block_5_act_3_s0'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_6_3_s0'
    )(x1_5)
    x1_6 = Activation('relu'
                      , name='conv_block_6_act_3_s0'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_7_3_s0'
    )(x1_6)
    x1_7 = Activation('relu'
                      , name='conv_block_7_act_3_s0'
                      )(x1_7)
    return x1_7
def view3_decoder_s1(base_weight_decay, x):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_5_3_s1'
    )(x)
    x1_5 = Activation('relu'
                      , name='conv_block_5_act_3_s1'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_6_3_s1'
    )(x1_5)
    x1_6 = Activation('relu'
                      , name='conv_block_6_act_3_s1'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_7_3_s1'
    )(x1_6)
    x1_7 = Activation('relu'
                      , name='conv_block_7_act_3_s1'
                      )(x1_7)
    return x1_7
def view3_decoder_s2(base_weight_decay, x):
    #  dmap
    # conv block 5
    x1_5 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_5_3_s2'
    )(x)
    x1_5 = Activation('relu'
                      , name='conv_block_5_act_3_s2'
    )(x1_5)

    # conv block 6
    x1_6 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_6_3_s2'
    )(x1_5)
    x1_6 = Activation('relu'
                      , name='conv_block_6_act_3_s2'
                      )(x1_6)

    # conv block 7
    x1_7 = Conv2D(
        data_format='channels_last',
        trainable=True,
        filters=1,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer='he_normal',
        padding='same',
        kernel_regularizer=l2(base_weight_decay),
        use_bias=True,
        activation=None,
        name='conv_block_7_3_s2'
    )(x1_6)
    x1_7 = Activation('relu'
                      , name='conv_block_7_act_3_s2'
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
    #model.load_weights('models/Street_fixedUnsynced_+3_fixed_fusion/09-34.2848.h5', by_name=True)  # lr = 0.0001
    return model



def scale_selection_mask(base_weight_decay, input_depth_maps):
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
        activation=None
        #name='scale_fusion1'
    )(input_depth_maps)
    #view1_scale_mask = Activation('softmax')(view1_scale)
    return  view1_scale




##################################################################################
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

    input_shape4 = (batch_size, int(768/4), int(640/4), 64)

    input_shape5 = (batch_size, int(768/4), int(640/4), 96)

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

    # encoder_view2_u_s0 = feature_extraction_view2(base_weight_decay,
    #                                          input_shape1,
    #                                          trainable_flag)
    # encoder_view2_u_s1 = feature_extraction_view2(base_weight_decay,
    #                                          input_shape2,
    #                                          trainable_flag)
    # encoder_view2_u_s2 = feature_extraction_view2(base_weight_decay,
    #                                          input_shape3,
    #                                          trainable_flag)
    #
    # encoder_view3_u_s0 = feature_extraction_view3(base_weight_decay,
    #                                          input_shape1,
    #                                          trainable_flag)
    # encoder_view3_u_s1 = feature_extraction_view3(base_weight_decay,
    #                                          input_shape2,
    #                                          trainable_flag)
    # encoder_view3_u_s2 = feature_extraction_view3(base_weight_decay,
    #                                          input_shape3,
    #                                          trainable_flag)


    encoder_view1_OP = feature_extraction_view1_OP(base_weight_decay,
                                             input_shape1,
                                             trainable_flag)
    encoder_view2_OP = feature_extraction_view2_OP(base_weight_decay,
                                             input_shape2,
                                             trainable_flag)
    encoder_view3_OP = feature_extraction_view3_OP(base_weight_decay,
                                             input_shape3,
                                             trainable_flag)

    # estimate_OP_w12 = optical_flow_estimation_w12(base_weight_decay,
    #                                               input_shape4,
    #                                               trainable_flag)
    # estimate_OP_w13 = optical_flow_estimation_w13(base_weight_decay,
    #                                               input_shape4,
    #                                               trainable_flag)

    multi_view_decoder = multi_view_fusion_decoder(base_weight_decay,
                                                  input_shape5,
                                                  trainable_flag)


    ####################### synced input feature extraction for crowd counting #########################
    # image pyramids:
    x1_s0_output = encoder_view1_s0(input_patches1_s0)
    x1_s1_output = encoder_view1_s1(input_patches1_s1)
    x1_s2_output = encoder_view1_s2(input_patches1_s2)

    # x1_s0_output = feature_extraction_view1_s0(base_weight_decay, input_patches1_s0, trainable_flag)
    # x1_s1_output = feature_extraction_view1_s1(base_weight_decay, input_patches1_s1, trainable_flag)
    # x1_s2_output = feature_extraction_view1_s2(base_weight_decay, input_patches1_s2, trainable_flag)

    # x2_s0_output = encoder_view2_s0(input_patches2_s0)
    # x2_s1_output = encoder_view2_s1(input_patches2_s1)
    # x2_s2_output = encoder_view2_s2(input_patches2_s2)
    #
    # # x2_s0_output = feature_extraction_view2_s0(base_weight_decay, input_patches2_s0, trainable_flag)
    # # x2_s1_output = feature_extraction_view2_s1(base_weight_decay, input_patches2_s1, trainable_flag)
    # # x2_s2_output = feature_extraction_view2_s2(base_weight_decay, input_patches2_s2, trainable_flag)
    #
    # x3_s0_output = encoder_view3_s0(input_patches3_s0)
    # x3_s1_output = encoder_view3_s1(input_patches3_s1)
    # x3_s2_output = encoder_view3_s2(input_patches3_s2)
    # # x3_s0_output = feature_extraction_view3_s0(base_weight_decay, input_patches3_s0, trainable_flag)
    # # x3_s1_output = feature_extraction_view3_s1(base_weight_decay, input_patches3_s1, trainable_flag)
    # # x3_s2_output = feature_extraction_view3_s2(base_weight_decay, input_patches3_s2, trainable_flag)


    ############## unsynced input for crowd counting  ################
    x2_u_s0_output = encoder_view2_s0(input_patches2_u_s0)
    x2_u_s1_output = encoder_view2_s1(input_patches2_u_s1)
    x2_u_s2_output = encoder_view2_s2(input_patches2_u_s2)

    x3_u_s0_output = encoder_view3_s0(input_patches3_u_s0)
    x3_u_s1_output = encoder_view3_s1(input_patches3_u_s1)
    x3_u_s2_output = encoder_view3_s2(input_patches3_u_s2)

    # x2_u_s0_output = feature_extraction_view2_u(base_weight_decay, input_patches2_u_s0)
    # x2_u_s1_output = feature_extraction_view2_u(base_weight_decay, input_patches2_u_s1)
    # x2_u_s2_output = feature_extraction_view2_u(base_weight_decay, input_patches2_u_s2)
    #
    # x3_u_s0_output = feature_extraction_view3_u(base_weight_decay, input_patches3_u_s0)
    # x3_u_s1_output = feature_extraction_view3_u(base_weight_decay, input_patches3_u_s1)
    # x3_u_s2_output = feature_extraction_view3_u(base_weight_decay, input_patches3_u_s2)

    ############### decoder  ################
    # x1_7 = view1_decoder_s0(base_weight_decay, x1_s0_output)
    # x2_7 = view2_decoder_s0(base_weight_decay, x2_s0_output)
    # x3_7 = view3_decoder_s0(base_weight_decay, x3_s0_output)

    x1_7 = view1_decoder_s0(base_weight_decay, x1_s0_output)
    x2_7 = view2_decoder_s0(base_weight_decay, x2_u_s0_output)
    x3_7 = view3_decoder_s0(base_weight_decay, x3_u_s0_output)

    # x1_7_s1 = view1_decoder_s1(base_weight_decay, x1_s1_output)
    # x2_7_s1 = view2_decoder_s1(base_weight_decay, x2_s1_output)
    # x3_7_s1 = view3_decoder_s1(base_weight_decay, x3_s1_output)
    #
    # x1_7_s2 = view1_decoder_s2(base_weight_decay, x1_s2_output)
    # x2_7_s2 = view2_decoder_s2(base_weight_decay, x2_s2_output)
    # x3_7_s2 = view3_decoder_s2(base_weight_decay, x3_s2_output)


    ############################## optical flow estimation and warping  #####################
    # view 1:
    x1_4_OP_s0 = encoder_view1_OP(input_patches1_s0)
    x1_4_OP_s1 = encoder_view1_OP(input_patches1_s1)
    x1_4_OP_s2 = encoder_view1_OP(input_patches1_s2)

    # view 2:
    # optical flow
    x2_4_u_OP_s0 = encoder_view2_OP(input_patches2_u_s0)
    x2_4_u_OP_s1 = encoder_view2_OP(input_patches2_u_s1)
    x2_4_u_OP_s2 = encoder_view2_OP(input_patches2_u_s2)

    # x12_OP_cat_0 = Concatenate(name='x12_OP_cat_0')([x1_4_proj_OP, x2_4_proj_OP])
    # w12_0 = estimate_OP_w12(x12_OP_cat_0)
    #### estimation #####

    # s2 (smallest scale)
    x12_OP_corr_s2 = Correlation_Layer(name='x12_OP_corr_s2', view=2, scale=2)([x1_4_OP_s2, x2_4_u_OP_s2])
    w12_s2 = optical_flow_estimation_w12(base_weight_decay,
                                         x12_OP_corr_s2,
                                         trainable_flag)

    # s1
    ## upsampling
    height0 = x1_4_OP_s1.shape[1].value
    width0 = x1_4_OP_s1.shape[2].value
    x1_4_OP_s2_up = UpSampling_layer(size=[height0, width0])([x1_4_OP_s2])

    height0 = x2_4_u_OP_s1.shape[1].value
    width0 = x2_4_u_OP_s1.shape[2].value
    x2_4_u_OP_s2_up = UpSampling_layer(size=[height0, width0])([x2_4_u_OP_s2])

    x12_OP_corr_s2_up = Correlation_Layer(name='x12_OP_corr_s2_up', view=2, scale=1)([x1_4_OP_s2_up, x2_4_u_OP_s2_up])
    x12_OP_corr_s1 = Correlation_Layer(name='x12_OP_corr_s1', view=2, scale=1)([x1_4_OP_s1, x2_4_u_OP_s1])

    x12_OP_corr_s1_subtract = Subtract()([x12_OP_corr_s1, x12_OP_corr_s2_up])
    w12_s1_res = optical_flow_estimation_w12(base_weight_decay,
                                             x12_OP_corr_s1_subtract,
                                             trainable_flag)
    ## addtion
    height0 = w12_s1_res.shape[1].value
    width0 = w12_s1_res.shape[2].value
    w12_s2_up = UpSampling_layer(size=[height0, width0])([w12_s2])
    w12_s1 = Add()([w12_s1_res, w12_s2_up])

    # # s0
    # ## upsampling
    # height0 = x1_4_OP_s0.shape[1].value
    # width0 = x1_4_OP_s0.shape[2].value
    # x1_4_OP_s1_up = UpSampling_layer(size=[height0, width0])([x1_4_OP_s1])
    #
    # height0 = x2_4_u_OP_s0.shape[1].value
    # width0 = x2_4_u_OP_s0.shape[2].value
    # x2_4_u_OP_s1_up = UpSampling_layer(size=[height0, width0])([x2_4_u_OP_s1])
    #
    # x12_OP_corr_s1_up = Correlation_Layer(name='x12_OP_corr_s1_up')([x1_4_OP_s1_up, x2_4_u_OP_s1_up])
    # x12_OP_corr_s0 = Correlation_Layer(name='x12_OP_corr_s0')([x1_4_OP_s0, x2_4_u_OP_s0])
    #
    # x12_OP_corr_s0_subtract = Subtract()([x12_OP_corr_s0, x12_OP_corr_s1_up])
    # w12_s0_res = optical_flow_estimation_w12(base_weight_decay,
    #                                          x12_OP_corr_s0_subtract,
    #                                          trainable_flag)
    # ## upsampling
    # height0 = w12_s0_res.shape[1].value
    # width0 = w12_s0_res.shape[2].value
    # w12_s1_up = UpSampling_layer(size=[height0, width0])([w12_s1])
    # w12_s0 = Add()([w12_s0_res, w12_s1_up])
    ## upsampling
    height0 = x1_4_OP_s0.shape[1].value
    width0 = x1_4_OP_s0.shape[2].value
    w12_s1_up = UpSampling_layer(size=[height0, width0])([w12_s1])
    w12_s0 = w12_s1_up

    ########################## warping #####################################
    # use the same
    # x2_4_proj_est_0 = optical_flow_warping(view=2, name='w12_0')([w12_0, fusion_v2_proj])
    x2_s0_est = optical_flow_warping(view=2, name='w12_s0')([w12_s0, x2_u_s0_output])
    x2_s1_est = optical_flow_warping(view=2, name='w12_s1')([w12_s1, x2_u_s1_output])
    x2_s2_est = optical_flow_warping(view=2, name='w12_s2')([w12_s2, x2_u_s2_output])

    # view 3:
    # optical flow
    x3_4_u_OP_s0 = encoder_view3_OP(input_patches3_u_s0)
    x3_4_u_OP_s1 = encoder_view3_OP(input_patches3_u_s1)
    x3_4_u_OP_s2 = encoder_view3_OP(input_patches3_u_s2)

    # s2 (smallest scale)
    x13_OP_corr_s2 = Correlation_Layer(name='x13_OP_corr_s2', view=3, scale=2)([x1_4_OP_s2, x3_4_u_OP_s2])
    w13_s2 = optical_flow_estimation_w13(base_weight_decay,
                                         x13_OP_corr_s2,
                                         trainable_flag)

    # s1
    ## upsampling
    height0 = x1_4_OP_s1.shape[1].value
    width0 = x1_4_OP_s1.shape[2].value
    x1_4_OP_s2_up = UpSampling_layer(size=[height0, width0])([x1_4_OP_s2])

    height0 = x3_4_u_OP_s1.shape[1].value
    width0 = x3_4_u_OP_s1.shape[2].value
    x3_4_u_OP_s2_up = UpSampling_layer(size=[height0, width0])([x3_4_u_OP_s2])

    x13_OP_corr_s2_up = Correlation_Layer(name='x13_OP_corr_s2_up', view=3, scale=1)([x1_4_OP_s2_up, x3_4_u_OP_s2_up])
    x13_OP_corr_s1 = Correlation_Layer(name='x13_OP_corr_s1', view=3, scale=1)([x1_4_OP_s1, x3_4_u_OP_s1])

    x13_OP_corr_s1_subtract = Subtract()([x13_OP_corr_s1, x13_OP_corr_s2_up])
    w13_s1_res = optical_flow_estimation_w13(base_weight_decay,
                                             x13_OP_corr_s1_subtract,
                                             trainable_flag)
    ## addtion
    height0 = w13_s1_res.shape[1].value
    width0 = w13_s1_res.shape[2].value
    w13_s2_up = UpSampling_layer(size=[height0, width0])([w13_s2])
    w13_s1 = Add()([w13_s1_res, w13_s2_up])

    # # s0
    # ## upsampling
    # height0 = x1_4_OP_s0.shape[1].value
    # width0 = x1_4_OP_s0.shape[2].value
    # x1_4_OP_s1_up = UpSampling_layer(size=[height0, width0])([x1_4_OP_s1])
    #
    # height0 = x3_4_u_OP_s0.shape[1].value
    # width0 = x3_4_u_OP_s0.shape[2].value
    # x3_4_u_OP_s1_up = UpSampling_layer(size=[height0, width0])([x3_4_u_OP_s1])
    #
    # x13_OP_corr_s1_up = Correlation_Layer(name='x13_OP_corr_s1_up')([x1_4_OP_s1_up, x3_4_u_OP_s1_up])
    # x13_OP_corr_s0 = Correlation_Layer(name='x13_OP_corr_s0')([x1_4_OP_s0, x3_4_u_OP_s0])
    #
    # x13_OP_corr_s0_subtract = Subtract()([x13_OP_corr_s0, x13_OP_corr_s1_up])
    # w13_s0_res = optical_flow_estimation_w13(base_weight_decay,
    #                                          x13_OP_corr_s0_subtract,
    #                                          trainable_flag)
    # ## upsampling
    # height0 = w13_s0_res.shape[1].value
    # width0 = w13_s0_res.shape[2].value
    # w13_s1_up = UpSampling_layer(size=[height0, width0])([w13_s1])
    # w13_s0 = Add()([w13_s0_res, w13_s1_up])
    ## upsampling
    height0 = x1_4_OP_s0.shape[1].value
    width0 = x1_4_OP_s0.shape[2].value
    w13_s1_up = UpSampling_layer(size=[height0, width0])([w13_s1])
    w13_s0 = w13_s1_up

    #### warping #####
    # use the same
    # x2_4_proj_est_0 = optical_flow_warping(view=2, name='w12_0')([w12_0, fusion_v2_proj])
    x3_s0_est = optical_flow_warping(view=3, name='w13_s0')([w13_s0, x3_u_s0_output])
    x3_s1_est = optical_flow_warping(view=3, name='w13_s1')([w13_s1, x3_u_s1_output])
    x3_s2_est = optical_flow_warping(view=3, name='w13_s2')([w13_s2, x3_u_s2_output])



    #################################### fusion #############################################
    ################# get the scale-selection mask #####################
    # view depth of image
    batch_size = x1_s0_output.shape[0].value
    height = x1_s0_output.shape[1].value
    width = x1_s0_output.shape[2].value
    num_channels = x1_s0_output.shape[3].value
    output_shape = [1, height, width, 1]

    # view1_depth = feature_scale_fusion_layer_mask(scale_number=scale_number,
    #                                               view = 1, output_shape=output_shape)
    # view2_depth = feature_scale_fusion_layer_mask(scale_number=scale_number,
    #                                               view = 2, output_shape=output_shape)
    # view3_depth = feature_scale_fusion_layer_mask(scale_number=scale_number,
    #                                               view = 3, output_shape=output_shape)

    # view1_scale = scale_selection_mask(base_weight_decay, input_depth_maps_v1)
    # view2_scale = scale_selection_mask(base_weight_decay, input_depth_maps_v2)
    # view3_scale = scale_selection_mask(base_weight_decay, input_depth_maps_v3)

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
    fusion_v1_proj = SpatialTransformer(1, [int(768/4), int(640/4)])(fusion_v1)

    # # view 2
    # ## conv
    # x2_s0_output_fusion = fusion_conv_v2(base_weight_decay, x2_s0_output)
    # x2_s1_output_fusion = fusion_conv_v2(base_weight_decay, x2_s1_output)
    # x2_s2_output_fusion = fusion_conv_v2(base_weight_decay, x2_s2_output)
    #
    # ## up sampling
    # x2_s1_output_fusion = UpSampling_layer(size=[height, width])([x2_s1_output_fusion])
    # x2_s2_output_fusion = UpSampling_layer(size=[height, width])([x2_s2_output_fusion])
    #
    # concatenated_map_v2 = Concatenate(name='cat_map_v2')(
    #     [x2_s0_output_fusion, x2_s1_output_fusion, x2_s2_output_fusion])
    # fusion_v2 = feature_scale_fusion_layer(scale_number=scale_number)([concatenated_map_v2, view2_scale_mask])
    #
    # ## proj
    # fusion_v2_proj = SpatialTransformer(2, [768/4, 640/4])(fusion_v2)
    #
    #
    # # view 3
    # ## conv
    # x3_s0_output_fusion = fusion_conv_v3(base_weight_decay, x3_s0_output)
    # x3_s1_output_fusion = fusion_conv_v3(base_weight_decay, x3_s1_output)
    # x3_s2_output_fusion = fusion_conv_v3(base_weight_decay, x3_s2_output)
    #
    # ## up sampling
    # x3_s1_output_fusion = UpSampling_layer(size=[height, width])([x3_s1_output_fusion])
    # x3_s2_output_fusion = UpSampling_layer(size=[height, width])([x3_s2_output_fusion])
    #
    # concatenated_map_v3 = Concatenate(name='cat_map_v3')(
    #     [x3_s0_output_fusion, x3_s1_output_fusion, x3_s2_output_fusion])
    #
    # fusion_v3 = feature_scale_fusion_layer(scale_number=scale_number)([concatenated_map_v3, view3_scale_mask])
    #
    # ## proj
    # fusion_v3_proj = SpatialTransformer(3, [768/4, 640/4])(fusion_v3)



    ############################ unsyced input projection and fusion: ###############

    # unsyced input: view 2
    ## conv
    x2_s0_est_fusion = fusion_conv_v2(base_weight_decay, x2_s0_est)
    x2_s1_est_fusion = fusion_conv_v2(base_weight_decay, x2_s1_est)
    x2_s2_est_fusion = fusion_conv_v2(base_weight_decay, x2_s2_est)

    ## up sampling
    x2_s1_est_fusion = UpSampling_layer(size=[height, width])([x2_s1_est_fusion])
    x2_s2_est_fusion = UpSampling_layer(size=[height, width])([x2_s2_est_fusion])

    concatenated_map_v2_est = Concatenate(name='cat_map_v2_est')(
        [x2_s0_est_fusion, x2_s1_est_fusion, x2_s2_est_fusion])

    fusion_v2_est = feature_scale_fusion_layer(scale_number=scale_number)([concatenated_map_v2_est,
                                                                         view2_scale_mask])

    ## proj
    fusion_v2_est_proj = SpatialTransformer(2, [int(768/4), int(640/4)])(fusion_v2_est)

    # unsyced input: view 3
    ## conv
    x3_s0_est_fusion = fusion_conv_v3(base_weight_decay, x3_s0_est)
    x3_s1_est_fusion = fusion_conv_v3(base_weight_decay, x3_s1_est)
    x3_s2_est_fusion = fusion_conv_v3(base_weight_decay, x3_s2_est)

    ## up sampling
    x3_s1_est_fusion = UpSampling_layer(size=[height, width])([x3_s1_est_fusion])
    x3_s2_est_fusion = UpSampling_layer(size=[height, width])([x3_s2_est_fusion])

    concatenated_map_v3_est = Concatenate(name='cat_map_v3_est')(
        [x3_s0_est_fusion, x3_s1_est_fusion, x3_s2_est_fusion])

    fusion_v3_est = feature_scale_fusion_layer(scale_number=scale_number)([concatenated_map_v3_est,
                                                                         view3_scale_mask])

    ## proj
    fusion_v3_est_proj = SpatialTransformer(3, [int(768/4), int(640/4)])(fusion_v3_est)



    ################# concatenate ################
    # concatenated_map_0 = Concatenate(name='cat_map_fusion_0')([fusion_v1_proj,
    #                                                        fusion_v2_proj,
    #                                                        fusion_v3_proj])

    concatenated_map = Concatenate(name='cat_map_fusion')([fusion_v1_proj,
                                                           fusion_v2_est_proj,
                                                           fusion_v3_est_proj])

    # x_output_0 = multi_view_decoder(concatenated_map_0)
    x_output = multi_view_decoder(concatenated_map)

    # We could use the synced frames to directly guide the synchronization module
    # or use consistency between views when synced frames are not available
    l12 =  ReduceSum_layer()([fusion_v1_proj, fusion_v2_est_proj])
    l13 =  ReduceSum_layer()([fusion_v1_proj, fusion_v3_est_proj])


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
                      outputs=[x1_7,  # x1_7_s1, x1_7_s2,
                               #x2_7, #x2_7_s1, x2_7_s2,
                               #x3_7,
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
                      outputs=[x1_7, #x1_7_s1, x1_7_s2,
                               # x2_7, #x2_7_s1, x2_7_s2,
                               # x3_7,
                               x_output,
                               l12, l13],  # x_output_0, x_output
                      name = net_name+'overall')

    print('Compiling ...')

    def zero_loss(y_true, y_pred):
        return keras.backend.mean(y_pred) - 1

    model.compile(optimizer=optimizer,
                  loss=['mse', 
                        # 'mse','mse',
                        'mse',
                        zero_loss, zero_loss],
                  loss_weights=[0.1,
                                # 0.01,
                                # 0.01,
                                1,
                                100, 100])#,

    return model