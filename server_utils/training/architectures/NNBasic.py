import numpy as np
import tensorflow as tf


def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    input_layer = features[params["feature_name"]]

    # First convolutional Layer and pooling layer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=None)
    batch_norm1 = tf.layers.batch_normalization(conv1)
    relu1 = tf.nn.relu(batch_norm1)
    pool1 = tf.layers.max_pooling2d(inputs=relu1, pool_size=[2, 2], strides=2)

    # Second convolutional Layer and pooling layer
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=None)
    batch_norm2 = tf.layers.batch_normalization(conv2)
    relu2 = tf.nn.relu(batch_norm2)
    pool2 = tf.layers.max_pooling2d(inputs=relu2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    pool2_flat = tf.layers.flatten(pool2)

    # Dense Layer
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits layer
    logits = tf.layers.dense(inputs=dropout, units=NUM_CLASSES)

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
