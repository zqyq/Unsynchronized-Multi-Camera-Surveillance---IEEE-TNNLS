from __future__ import print_function
import os
import sys
import time
import h5py
from pprint import pprint, pformat
import numpy as np
np.set_printoptions(precision=6)
fixed_seed = 999
np.random.seed(fixed_seed)  # Set seed for reproducibility
import tensorflow as tf
import keras
print("Using keras {}".format(keras.__version__))
# assert keras.__version__.startswith('2.')
tf.set_random_seed(fixed_seed)

from datagen_v3_randomUnsynced import datagen_v3
from net_def_moreSupervised import build_model_FCN_model_api as build_FCNN


from keras.optimizers import Adam, Nadam
from keras.optimizers import SGD
# from MyOptimizer_keras2 import SGD_policy as SGD
from keras.callbacks import Callback, CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import load_model

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# # set enough GPU memory as needed(default, all GPU memory is used)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)



def logging_level(level='info'):
    import logging
    str_format = '%(asctime)s - %(levelname)s: %(message)8s'
    if level == 'debug':
        logging.basicConfig(level=logging.DEBUG, format=str_format, datefmt='%Y-%m-%d %H:%M:%S')
    elif level == 'info':
        logging.basicConfig(level=logging.INFO, format=str_format, datefmt='%Y-%m-%d %H:%M:%S')

    return logging


class batch_loss_callback(Callback):
    """Callback that streams epoch results to a plain txt file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, append=False):
        self.filename = filename
        self.append = append
        self.append_header = True
        self.batch_number = 0
        super(batch_loss_callback, self).__init__()

    def on_train_begin(self, logs=None):
        self.losses = list()
        # pass
        # if self.append:
        #     if os.path.exists(self.filename):
        #         with open(self.filename) as f:(
        #             self.append_header = not bool(len(f.readline()))
        #     self.textfile = open(self.filename, 'a')
        # else:
        #     self.textfile = open(self.filename, 'w')

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))
        self.batch_number += 1
        if self.batch_number % 100 == 0:
            # self.textfile.write("Batch No. {:7d}\n".format(self.batch_number))
            # self.textfile.write("    {:.6f}\n".format(logs.get('loss')))
            info_print_1 = "Batch No. {:7d}".format(self.batch_number)
            info_print_2 = "    loss: {:.6f}".format(logs.get('loss'))
            # self.textfile.write("{}\n".format(info_print_1))
            # self.textfile.write("{}\n".format(info_print_2))
            logging.info(info_print_1)
            logging.info(info_print_2)
            # self.textfile.flush()

    def on_epoch_end(self, epoch, logs=None):
        temp_name, _ = self.filename.rsplit('.', 1)
        with h5py.File('{}.h5'.format(temp_name), 'w') as f:
            f['losses'] = np.asarray(self.losses, dtype=np.float32)
        # pass
        # self.textfile.close()


def main(exp_name='FCNN', verbosity=0):
    ################################################################################
    # Experiment Settings
    ################################################################################

    # Model + Generator hyper-parameters
    optimizer = 'sgd'
    learning_rate = 0.0001   # 0.0001
    # learning_rate = 0.002  # for Nadam only
    lr_decay = 0.0001
    momentum = 0.9
    nesterov = False
    weight_decay = 0.0001 # 0.001
    save_dir = os.path.join("", exp_name)

    batch_size = 1
    epochs = 2000
    images_per_set = None
    patches_per_image = 1      #1000
    patch_dim = (288, 384, 1)
    image_shuffle = True
    patch_shuffle = True
    epoch_random_state = None  # Set this to an integer if you want the data the same in every epoch

    train_samples = 1105*10    #243243 #20490
    val_samples   = 794    #59598 #8398

    ################################################################################
    # Model Definition
    if optimizer.lower() == 'sgd':
        opt = SGD(
            lr=learning_rate,
            decay=lr_decay,
            momentum=momentum,
            nesterov=nesterov,
            # lr_policy='inv' if params_solver.get('lr_policy', None) is None else params_solver['lr_policy'],
            # step=params_solver.get('step', 10000000.),  # useful only when lr_policy == 'step'
            clipnorm=5,
            clipvalue=1)
    elif optimizer.lower() == 'adam':
        logging.info("use Adam solver")
        opt = Adam(
            lr=learning_rate,
            # decay=params_solver.get('lr_decay', 0.),
            clipnorm=5,
            clipvalue=1)
    elif optimizer.lower() == 'nadam':
        logging.info("use Nadam solver")
        opt = Nadam(
            lr=learning_rate,
            clipnorm=5,
            clipvalue=1)
    else:
        logging.error('Unrecognized solver')

    model = build_FCNN(
        batch_size=batch_size,
        patch_size=patch_dim,
        optimizer=opt,
        base_weight_decay=weight_decay,
        output_ROI_mask=False,)


    # Generator setup
    scaler_stability_factor = 1  # 100

    train_path0 = '../../dataset/PETS_2009/dmaps/'

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

    # the ground-plane GT is corresponding to the view 1 timepoints in 'dataset/dmaps/.../GP_maps/';
    # if not, use ground-plane GT in 'dataset/unsynced_GP_maps/'
    train_GP_1 = train_path0 + 'S1L3/14_17/GP_maps/PETS_S1L3_1_groundplane_dmaps_10.h5'
    train_GP_2 = train_path0 + 'S1L3/14_33/GP_maps/PETS_S1L3_2_groundplane_dmaps_10.h5'
    train_GP_3 = train_path0 + 'S2L2/14_55/GP_maps/PETS_S2L2_1_groundplane_dmaps_10.h5'
    train_GP_4 = train_path0 + 'S2L3/14_41/GP_maps/PETS_S2L3_1_groundplane_dmaps_10.h5'

    h5file_train_view1 = [train_view1_1, train_view1_2, train_view1_3, train_view1_4]
    h5file_train_view2 = [train_view2_1, train_view2_2, train_view2_3, train_view2_4]
    h5file_train_view3 = [train_view3_1, train_view3_2, train_view3_3, train_view3_4]
    h5file_train_GP = [train_GP_1, train_GP_2, train_GP_3, train_GP_4]

    train_gen = datagen_v3(
        h5file_view1=h5file_train_view1,
        h5file_view2=h5file_train_view2,
        h5file_view3=h5file_train_view3,
        h5file_GP=h5file_train_GP,

        batch_size=batch_size,
        images_per_set=images_per_set,
        patches_per_image=patches_per_image,
        patch_dim=patch_dim[:2],
        density_scaler=scaler_stability_factor,
        image_shuffle=image_shuffle,
        patch_shuffle=patch_shuffle,
        random_state=epoch_random_state
    )

    test_path0 = train_path0

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

    # the ground-plane GT is corresponding to the view 1 timepoints in 'dataset/dmaps/.../GP_maps/';
    # if not, use ground-plane GT in 'dataset/unsynced_GP_maps/'
    test_GP_1 = test_path0 + 'S1L1/13_57/GP_maps/PETS_S1L1_1_groundplane_dmaps_10.h5'
    test_GP_2 = test_path0 + 'S1L1/13_59/GP_maps/PETS_S1L1_2_groundplane_dmaps_10.h5'
    test_GP_3 = test_path0 + 'S1L2/14_06/GP_maps/PETS_S1L2_1_groundplane_dmaps_10.h5'
    test_GP_4 = test_path0 + 'S1L2/14_31/GP_maps/PETS_S1L2_2_groundplane_dmaps_10.h5'

    h5file_test_GP = [test_GP_1, test_GP_2, test_GP_3, test_GP_4]

    h5file_test_view1 = [test_view1_1, test_view1_2, test_view1_3, test_view1_4]
    h5file_test_view2 = [test_view2_1, test_view2_2, test_view2_3, test_view2_4]
    h5file_test_view3 = [test_view3_1, test_view3_2, test_view3_3, test_view3_4]

    val_gen = datagen_v3(
        h5file_view1=h5file_test_view1,
        h5file_view2=h5file_test_view2,
        h5file_view3=h5file_test_view3,
        h5file_GP=h5file_test_GP,

        batch_size=batch_size,
        images_per_set=images_per_set,
        patches_per_image=1,  # 1000,
        patch_dim=patch_dim[:2],
        density_scaler=scaler_stability_factor,
        image_shuffle=image_shuffle,
        patch_shuffle=patch_shuffle,
        random_state=epoch_random_state
    )

    # Model Training
    # Save directory
    if not os.path.exists(save_dir):
        logging.info(">>>> save dir: {}".format(save_dir))
        os.makedirs(save_dir)
    callbacks = list()
    callbacks.append(CSVLogger(
        filename=os.path.join(save_dir, 'train_val.csv'),
        separator=',',
        append=False,  # useful if it's resumed from the latest checkpoint
    ))
    callbacks.append(ModelCheckpoint(
        filepath=os.path.join(save_dir, '{epoch:02d}-{val_loss:.4f}.h5'),
        monitor='val_loss',
        verbose=1,
        save_weights_only=True,
        save_best_only=False,
    ))
    callbacks.append(ModelCheckpoint(
        filepath=os.path.join(save_dir, '{epoch:02d}-{val_loss:.4f}-better.h5'),
        monitor='val_loss',
        verbose=1,
        save_weights_only=True,
        save_best_only=True,  # this will save all the improved models
    ))
    #callbacks.append(EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto'))
    # callbacks.append(TensorBoard(
    #     log_dir=os.path.join(save_dir, 'TensorBoard_info'),
    #     histogram_freq=1,
    #     write_graph=True,
    #     write_images=True,
    #     embeddings_freq=0,  # default, new feature only in latest keras and tensorflow
    #     embeddings_layer_names=None,  # default
    #     embeddings_metadata=None,  # default
    # ))
    if verbosity == 0:
        callbacks.append(batch_loss_callback(
            filename=os.path.join(save_dir, 'train_val_loss_batch.log'),
            append=False,  # useful if it's resumed from the latest checkpoint
        ))

    logging.info('Begin training...')
    start_time = time.time()
    # train the network from here
    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        verbose=verbosity,
        callbacks=callbacks,
        validation_data=val_gen,
        validation_steps=val_samples // batch_size,
        max_q_size=20,
        workers=1,
        pickle_safe=False)

    # YY = model.layers[60].output
    # print(sum(YY.flatten))

    logging.info('----- {:.2f} seconds -----'.format(time.time() - start_time))

    # # Save model history
    # sys.setrecursionlimit(100000)
    # with open('{}/training.history'.format(save_dir), 'w') as f:
    #     # does not work for Keras2. Also not convinient to use
    #     pickle.dump(history, f)



if __name__ == '__main__':
    logging = logging_level('debug')
    logging.debug('use debug level logging setting')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp_name",
        type=str,
        default='models/PETS_moreSupervised_fusion_sync_syncloss',
        action="store")
    parser.add_argument(
        "-v", "--verbosity",
        type=int,
        default=0,
        choices=[0, 1, 2],
        action="store")
    args = parser.parse_args()
    main(exp_name=args.exp_name, verbosity=args.verbosity)
