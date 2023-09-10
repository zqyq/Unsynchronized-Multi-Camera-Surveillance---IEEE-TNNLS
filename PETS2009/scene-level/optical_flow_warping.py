from keras.layers.core import Layer
import tensorflow as tf
import numpy as np
import cv2

class optical_flow_warping(Layer):
    """optical_flow_warping
    Implements a optical_flow_warping layer
    Borrowed from [2]_:
    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
    """

    def __init__(self,
                 view,
                 **kwargs):
        self.view = view
        super(optical_flow_warping, self).__init__(**kwargs)


    # def build(self, input_shape):
    #    super(SpatialTransformer, self).build(input_shape)
        # self.locnet.build(input_shape)
        # self.trainable_weights = self.locnet.trainable_weights
        # self.regularizers = self.locnet.regularizers #//NOT SUER ABOUT THIS, THERE IS NO MORE SUCH PARAMETR AT self.locnet
        # self.constraints = self.locnet.constraints

    def compute_output_shape(self, input_shape):
        feature = input_shape[1]
        return (int(feature[0]),
                int(feature[1]),
                int(feature[2]),
                int(feature[-1]))

    def call(self, inputs, mask=None):
        output = self._transform(inputs)
        return output

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        num_channels = tf.shape(image)[3]

        batch_size = image.shape[0].value
        # height = image.shape[1].value
        # width = image.shape[2].value
        num_channels = image.shape[3].value

        x = tf.cast(x , dtype='float32')
        y = tf.cast(y , dtype='float32')

        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width  = output_size[1]

        x = .5*(x + 1.0)*(width_float)
        y = .5*(y + 1.0)*(height_float)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a*pixel_values_a,
                           area_b*pixel_values_b,
                           area_c*pixel_values_c,
                           area_d*pixel_values_d])
        return output



    def _meshgrid(self, height, width):
        x_linspace = tf.linspace(0.0, width-1, width)
        y_linspace = tf.linspace(0.0, height-1, height)

        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, [-1])
        y_coordinates = tf.reshape(y_coordinates, [-1])

        ones = tf.ones_like(x_coordinates)
        # indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)

        indices_grid = tf.concat([x_coordinates, y_coordinates], 0)
        return indices_grid



    def _transform(self, inputs):
        self.w = inputs[0]
        self.feature = inputs[1]

        feature = self.feature
        w = self.w
        view = self.view

        batch_size = tf.shape(feature)[0]
        height = tf.shape(feature)[1]
        width = tf.shape(feature)[2]
        num_channels = tf.shape(feature)[3]

        batch_size = feature.shape[0].value
        height = feature.shape[1].value
        width = feature.shape[2].value
        num_channels = feature.shape[3].value

        # width = tf.cast(width, dtype='float32')
        # height = tf.cast(height, dtype='float32')

        output_height = height
        output_width = width
        indices_grid = self._meshgrid(output_height, output_width)

        indices_grid = tf.expand_dims(indices_grid, 0)
        indices_grid = tf.reshape(indices_grid, [-1])

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, (batch_size, 2, -1))

        # reshape the flow:
        # h_width = w.shape[1].value
        # w_width = w.shape[2].value
        ################################### adding the flow  ###########################
        # w = tf.reshape(w, [-1])
        # flowXY = tf.reshape(w, (batch_size, 2, -1))
        w = tf.reshape(w, [batch_size, -1, 2])
        w = tf.transpose(w, [0, 2, 1])
        flowXY = tf.reshape(w, (batch_size, 2, -1))


        # get the transformed grids: adding the flow
        transformed_grid = indices_grid + flowXY
        x_s_flatten = 2.0*transformed_grid[:,0,:]/ max(width-1,1)-1.0
        y_s_flatten = 2.0*transformed_grid[:,1,:]/ max(height-1,1)-1.0
        x_s_flatten = tf.reshape(x_s_flatten, [-1])
        y_s_flatten = tf.reshape(y_s_flatten, [-1])

        if view==1:
            view = 'view1'
            view_gp_mask = np.load('coords_correspondence/mask/view1_gp_mask.npz')
        if view==2:
            view = 'view2'
            view_gp_mask = np.load('coords_correspondence/mask/view2_gp_mask.npz')
        if view==3:
            view = 'view3'
            view_gp_mask = np.load('coords_correspondence/mask/view3_gp_mask.npz')

        view_gp_mask = view_gp_mask.f.arr_0
        view_gp_mask = cv2.resize(view_gp_mask, (width, height))
        view_gp_mask = tf.cast(view_gp_mask, 'float32')
        view_gp_mask = tf.expand_dims(view_gp_mask, axis=0)
        view_gp_mask = tf.expand_dims(view_gp_mask, axis=-1)
        view_gp_mask = tf.tile(view_gp_mask, [batch_size, 1, 1, num_channels])
        view_gp_mask = tf.cast(view_gp_mask, 'float32')

        # x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        # y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        # x_s_flatten = tf.reshape(x_s, [-1])
        # y_s_flatten = tf.reshape(y_s, [-1])

        output_size = [output_height, output_width]
        transformed_image = self._interpolate(feature,
                                              x_s_flatten,
                                              y_s_flatten,
                                              output_size)

        transformed_image = tf.reshape(transformed_image, shape=(batch_size,
                                                                output_height,
                                                                output_width,
                                                                num_channels))

        transformed_image = tf.multiply(transformed_image, view_gp_mask)
        # transformed_image = tf.multiply(transformed_image, view_norm_mask)

        # # normalization:
        # # get the sum of each channel/each image
        # input_sum = tf.reduce_sum(input_shape, [1, 2])
        # input_sum = tf.expand_dims(input_sum, axis=1)
        # input_sum = tf.expand_dims(input_sum, axis=1)
        #
        # output_sum = tf.reduce_sum(transformed_image, [1, 2])
        # output_sum = tf.expand_dims(output_sum, axis=1)
        # output_sum = tf.expand_dims(output_sum, axis=1)
        #
        # amplify_times = tf.divide(input_sum, output_sum)
        # mul_times = tf.constant([1, output_height, output_width, 1])
        # amplify_times = tf.tile(amplify_times, mul_times)
        #
        # # transformed_image = tf.image.resize_images(transformed_image,
        # #                                            [output_height/4, output_width/4])
        #
        # transformed_image_sum = tf.multiply(transformed_image, amplify_times)

        return transformed_image  # no nomalization