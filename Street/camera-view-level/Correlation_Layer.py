from __future__ import print_function
import tensorflow as tf
# assert tf.__version__.startswith('1.')
from keras.engine import Layer
import numpy as np
from keras.layers import  Multiply
from keras.layers import  MaxPooling2D



class Correlation_Layer(Layer):

    def __init__(self,
                 view,
                 scale,
                 **kwargs):

        self.view = view
        self.scale = scale
        super(Correlation_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        print('No trainable weights for correlation layer.')

    def compute_output_shape(self, input_shape):
        feature = input_shape[0]
        return (feature[0],
                feature[1],
                feature[2],
                int(feature[1]*feature[2]))


    def call(self, x):

        view = self.view
        scale = self.scale

        feature_A = x[0]
        feature_B = x[1]

        b = feature_A.shape[0].value
        h = feature_A.shape[1].value
        w = feature_A.shape[2].value
        c = feature_A.shape[3].value

        # F12 = np.asarray([[-7.57435756e-07, 1.34702976e-06, 1.93248775e-03],
        #                     [7.94753860e-06, 4.05349344e-06, - 1.70838573e-02],
        #                     [1.37677386e-04, 7.45167670e-03, 1.00000000e+00]])
        # F13 = np.asarray([[-1.24373323e-05, -1.63886765e-06, -8.10405530e-03],
        #                   [3.31829911e-05, - 6.43505269e-06, 2.11265052e-02],
        #                   [3.53938546e-03, - 2.80395645e-02, 1.00000000e+00]])
        F12 = np.asarray([[1.12293771e-07, -1.97478833e-07, -1.59809795e-04],
                          [-5.70950953e-07, 2.46095145e-06, -1.01864207e-03],
                          [2.35104222e-04, -1.47524230e-03, 1.00000000e+00]])
        F13 = np.asarray([[-6.87272674e-06, 4.08789920e-05, 2.43438971e-03],
                          [2.76138333e-05, -7.45718447e-06, -8.51229065e-02],
                          [1.33866170e-03, -1.10301766e-02, 1.00000000e+00]])

        if view==2:
            F = F12
        elif view==3:
            F = F13

        Weight = self.epipolar_weight(F, scale, h, w)

        feature_A_flatten = tf.reshape(feature_A, [b, int(h*w), c])
        feature_A_flatten = tf.transpose(feature_A_flatten, [0, 2, 1])

        feature_B_flatten = tf.reshape(feature_B, [b, int(h*w), c])

        corr_AB = tf.matmul(feature_B_flatten, feature_A_flatten)
        corr_AB = tf.reshape(corr_AB, [b, h, w, int(h*w)])

        # epipolar constraint: suppress the values far away from the epipolar lines.
        ## try the epipolar line first
        # Weight = np.zeros((h, w, h*w))
        #
        # if view == 2:
        #     F = np.asarray([[1.12293771e-07, -1.97478833e-07, -1.59809795e-04],
        #                     [-5.70950953e-07, 2.46095145e-06, -1.01864207e-03],
        #                     [2.35104222e-04, -1.47524230e-03, 1.00000000e+00]])
        # if view == 3:
        #     F = np.asarray([[-6.87272674e-06, 4.08789920e-05, 2.43438971e-03],
        #                     [ 2.76138333e-05, -7.45718447e-06, -8.51229065e-02],
        #                     [ 1.33866170e-03, -1.10301766e-02, 1.00000000e+00]])
        #
        #
        # a0, b0, c0, d0, e0, f0, g0, h0, i = F.flatten()
        # for x0 in range(w):
        #     for y0 in range(h):
        #         ii = h*x0 + y0
        #         aa = a0*x0*32+b0*y0*32+c0
        #         bb = d0*x0*32+e0*y0*32+f0
        #         cc = g0*x0*32+y0*32*h0+1
        #
        #         for x1 in range(w):
        #             for y1 in range(h):
        #                 dist = np.abs(aa*x1+bb*y1+cc/32)/np.sqrt(aa*aa+bb*bb)
        #                 wi = np.exp(-dist/20)
        #                 Weight[y1, x1, ii] = wi

        # import matplotlib.pyplot as plt
        # plt.figure()
        # for ii in range(0, h*w, 200):
        #     print(ii)
        #     plt.imshow(Weight[:, :, ii])
        #     plt.pause(1)


        Weight = tf.expand_dims(Weight.astype('float32'), axis=0)
        Weight = tf.tile(Weight, [b, 1, 1, 1])
        corr_AB = tf.multiply(corr_AB, Weight)


        # normaliztion:
        corr_AB_sum = tf.reduce_sum(corr_AB, [1, 2])
        corr_AB_sum = tf.expand_dims(corr_AB_sum, axis=1)
        corr_AB_sum = tf.expand_dims(corr_AB_sum, axis=1)

        amplify_times = tf.divide(1, corr_AB_sum + 1e-8)
        mul_times = tf.constant([1, h, w, 1])
        amplify_times = tf.tile(amplify_times, mul_times)

        corr_AB_norm = tf.multiply(corr_AB, amplify_times)

        return corr_AB_norm

    def epipolar_weight(self, F, scale, h, w):

        ratio = 2**(4+scale)
        #ratio2 = 2**scale

        d_std = 256 #640.0 #256

        Weight = np.zeros((h, w, h * w))

        a0, b0, c0, d0, e0, f0, g0, h0, i = F.flatten()
        for x0 in range(w):
            for y0 in range(h):
                ii = x0 + y0*w
                # aa = a0 * x0 * ratio + b0 * y0 * ratio + c0
                # bb = d0 * x0 * ratio + e0 * y0 * ratio + f0
                # cc = g0 * x0 * ratio + y0 * ratio * h0 + 1
                aa = (a0 * x0 * ratio + b0 * y0 * ratio + c0)*ratio
                bb = (d0 * x0 * ratio + e0 * y0 * ratio + f0)*ratio
                cc = g0 * x0 * ratio + y0 * ratio * h0 + 1

                for x1 in range(w):
                    for y1 in range(h):
                        dist = np.abs(aa * x1 + bb * y1 + cc) / np.sqrt(aa * aa + bb * bb)
                        wi = np.exp(-dist / d_std)
                        Weight[y1, x1, ii] = wi
                # if scale == 1:
                #     a = 0
                #     import matplotlib.pyplot as plt
                #     plt.imshow(Weight[:, :, ii])
        return Weight