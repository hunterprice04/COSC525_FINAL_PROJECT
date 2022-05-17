import os
import tensorflow as tf


def create_callbacks(base_dir, model, defaults: list = None):
    dir_models = os.path.join(base_dir, model.name)
    os.makedirs(dir_models, exist_ok=True)
    print(f'# Model Dir: {dir_models}')

    path_csv = os.path.join(dir_models, 'history.csv')
    print(" - History Path:", path_csv)

    path_ckp = os.path.join(dir_models, 'checkpoints.h5')
    print(" - Checkpoint Path:", path_ckp)

    path_tb = os.path.join(dir_models)
    tb_file_writer = tf.summary.create_file_writer(path_tb)
    print(" - TB Path:", path_tb)

    callbacks = [] if defaults is None else defaults
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=path_tb, histogram_freq=1))
    callbacks.append(tf.keras.callbacks.CSVLogger(path_csv, separator=",", append=True))
    # callbacks.append(tf.keras.callbacks.ModelCheckpoint(path_ckp,
    #                                                     monitor='loss',
    #                                                     save_best_only=True,
    #                                                     mode='auto',
    #                                                     verbose=0))
    return callbacks, tb_file_writer
