
import argparse
import os
import sys
import time
import numpy as np
import tensorflow as tf

from architectures.NNPostmateArch import cnn_postmates_model_fn
from Utils import *
mnist = tf.keras.datasets.mnist

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string('model_dir', "model_dir", "Directory path where model will be stored")
tf.app.flags.DEFINE_string('config_file', "-", "File path to config file")
tf.app.flags.DEFINE_string('result_dir', "evaluation_results", "Directory path where evaluation result will be stored")
FLAGS = tf.app.flags.FLAGS


NUM_CLASSES = 10
INPUT_FEATURE = "image"
SAVE_STEP_PERIOD = 20

# TODO: Move to utils. Figure out better way to use global param INPUT_FEATURE
def serving_input_receiver_fn():
    """
    This is used to define inputs to serve the model.
    :return: ServingInputReciever
    """
    reciever_tensors = {
        # The size of input image is flexible.
        INPUT_FEATURE: tf.placeholder(tf.float32, [None, None, None, 1]),
    }

    # Convert give inputs to adjust to the model.
    features = {
        # Resize given images.
        INPUT_FEATURE: tf.image.resize_images(reciever_tensors[INPUT_FEATURE], [28, 28]),
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors,
                                                    features=features)


def main(_):
    train_data, eval_data, train_labels, eval_labels = load_mnsit_training_set()
    params = load_hyperparameters(FLAGS.config_file)
    loss_func = get_loss_function(params["loss_func"])
    optimizer = get_optimizer(params["optimizer"], float(params["learning_rate"]))

    # reshape images
    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    eval_data = eval_data.reshape(eval_data.shape[0], 28, 28, 1)

    # Create the Estimator
    training_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        save_summary_steps = SAVE_STEP_PERIOD,
        save_checkpoints_steps = SAVE_STEP_PERIOD
    )

    classifier = tf.estimator.Estimator(
        model_fn=cnn_postmates_model_fn,
        params={"loss_func": loss_func,
                "optimizer": optimizer,
                "classes_count": NUM_CLASSES,
                "feature_name": INPUT_FEATURE},
        model_dir=FLAGS.model_dir,
        config=training_config)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: train_data},
        y=train_labels,
        batch_size=int(params["batch_size"]),
        num_epochs=int(params["epochs"]),
        shuffle=(True if params["shuffle_train"] == "1" else False))
    classifier.train(
        input_fn=train_input_fn,
        steps=int(params["steps"]))

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: eval_data},
        y=eval_labels,
        num_epochs=int(params["epochs"]),
        shuffle=False)

    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    cur_time = str(time.time())

    # write result to file
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)
    with open(os.path.join(FLAGS.result_dir, cur_time), "w+") as fp:
        fp.write(str(eval_results))

    # Save the model
    classifier.export_savedmodel(os.path.join(FLAGS.model_dir, "output"),
                                serving_input_receiver_fn=serving_input_receiver_fn)


if __name__ == '__main__':

    tf.app.run()

