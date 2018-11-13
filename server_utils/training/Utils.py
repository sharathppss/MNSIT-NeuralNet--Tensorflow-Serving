import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist


def load_hyperparameters(config_file):
    """ Load api_keys for all geoservices from a file at APIKEYS_FILE.
    :config_file: config file to use.
    :return: dict of keys if successful else None
    raise exception on unable to load
    """
    result = {}
    try:
        with open(config_file, 'r', encoding='utf-8') as file:
            lines = file.read().splitlines()
        for line in lines:
            vals = line.split('=')
            result[vals[0]] = vals[1]

    except Exception as error:
        raise EnvironmentError("load_hyperparameters: Exception loading hyperparams: {0}".format(error))

    return result


def load_mnsit_training_set():
    """
    Load the MNSIT image data set.
    :return: data and labels.
    raise exception on unable to load
    """
    try:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        train_data = np.asarray(x_train, dtype=np.float32)
        eval_data = np.asarray(x_test, dtype=np.float32)
        train_labels = np.asarray(y_train, dtype=np.int32)
        eval_labels = np.asarray(y_test, dtype=np.int32)
        return train_data, eval_data, train_labels, eval_labels

    except Exception as error:
        raise EnvironmentError("load_mnsit_training_set: Exception loading MNSIT data: {0}".format(error))


def get_loss_function(loss):
    """
    retrieve loss function for neural net.
    :param loss: loss string value to be used.
    :return: loss function to use.
    raise exception on invalid value input
    """
    try:

        loss_func_map = {"sparse_softmax_cross_entropy": tf.losses.sparse_softmax_cross_entropy,
                         "sigmoid_cross_entropy": tf.losses.sigmoid_cross_entropy,
                         "softmax_cross_entropy": tf.losses.softmax_cross_entropy}

    except Exception as error:
        raise EnvironmentError("get_loss_function: Exception getting loss function: {0}".format(error))

    return loss_func_map[loss]


def get_optimizer(optimizer, learning_rate):
    """
    retrieve optimizer for neural net.

    :param optimizer: optimizer to use
    :param learning_rate: learning for optimizer to use
    :return: expected optimizer
    raise exception on invalid value input
    """
    try:
        loss_func_map = {"AdamOptimizer": tf.train.AdamOptimizer(learning_rate=learning_rate),
                         "AdadeltaOptimizer": tf.train.AdadeltaOptimizer(learning_rate=learning_rate),
                         "AdagradOptimizer": tf.train.AdagradOptimizer(learning_rate=learning_rate)}

    except Exception as error:
        raise EnvironmentError("get_optimizer: Exception getting optimizer function: {0}".format(error))

    return loss_func_map[optimizer]
