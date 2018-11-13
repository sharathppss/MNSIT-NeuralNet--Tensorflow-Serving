
import numpy as np
import tensorflow as tf


def cnn_postmates_model_fn(features, labels, mode, params):
    """Postmates Model function for CNN."""
    # Input Layer
    input_layer = features[params["feature_name"]]

    # 1 convolutional Layer and pooling layer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=None)
    batch_norm1 = tf.layers.batch_normalization(conv1)
    relu1 = tf.nn.relu(batch_norm1)
    #pool1 = tf.layers.max_pooling2d(inputs=relu1, pool_size=[2, 2], strides=2)

    # 2.1 convolutional Layer and pooling layer
    conv21 = tf.layers.conv2d(
        inputs=relu1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=None)
    batch_norm21= tf.layers.batch_normalization(conv21)
    relu21 = tf.nn.relu(batch_norm21)
    pool21 = tf.layers.max_pooling2d(inputs=relu21, pool_size=[2, 2], strides=2)

    # 2.2 convolutional Layer and pooling layer
    conv22 = tf.layers.conv2d(
        inputs=relu1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=None)
    batch_norm22= tf.layers.batch_normalization(conv22)
    relu22 = tf.nn.relu(batch_norm22)
    pool22 = tf.layers.max_pooling2d(inputs=relu22, pool_size=[2, 2], strides=2)

    # 3.1 convolutional Layer and pooling layer
    conv31 = tf.layers.conv2d(
        inputs=pool21,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=None)
    batch_norm31 = tf.layers.batch_normalization(conv31)
    relu31 = tf.nn.relu(batch_norm31)
    pool31 = tf.layers.max_pooling2d(inputs=relu31, pool_size=[2, 2], strides=2)

    # 3.2  convolutional Layer and pooling layer
    conv32 = tf.layers.conv2d(
        inputs=pool22,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=None)
    batch_norm32 = tf.layers.batch_normalization(conv32)
    relu32 = tf.nn.relu(batch_norm32)
    pool32 = tf.layers.max_pooling2d(inputs=relu32, pool_size=[2, 2], strides=2)

    comb3 = tf.concat([pool31, pool32], axis=2)

    # Flatten tensor into a batch of vectors
    comb3_flat = tf.layers.flatten(comb3)

    # 4 Dense Layer
    dense4 = tf.layers.dense(inputs=comb3_flat, units=1000, activation=tf.nn.relu)

    # 5 Dense Layer
    dense5 = tf.layers.dense(inputs=dense4, units=500, activation=tf.nn.relu)

    # Logits layer
    logits = tf.layers.dense(inputs=dense5, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = params["loss_func"](labels=labels, logits=logits)

    # Configure the training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params["optimizer"]
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
