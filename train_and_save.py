from __future__ import print_function

import glob
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disabilita GPU

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

tf.app.flags.DEFINE_string('work_dir', './temp', 'Working directory')
FLAGS = tf.app.flags.FLAGS

mnist_dataframe = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv",
    sep=",",
    header=None)

mnist_dataframe = mnist_dataframe.head(10000)
# mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))
mnist_dataframe.head()

mnist_dataframe.loc[:, 72:72]


def parse_labels_and_features(dataset):
    labels = dataset[0]
    features = dataset.loc[:, 1:784]
    features = features / 255
    return labels, features


def construct_feature_columns():
    return set([tf.feature_column.numeric_column('pixels', shape=784)])


def create_training_input_fn(features, labels, batch_size, num_epochs=None, shuffle=True):
    def _input_fn(num_epochs=None, shuffle=True):
        idx = np.random.permutation(features.index)
        raw_features = {"pixels": features.reindex(idx)}
        raw_targets = np.array(labels[idx])

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)

        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


def create_predict_input_fn(features, labels, batch_size):
    def _input_fn():
        raw_features = {"pixels": features.values}
        raw_targets = np.array(labels)

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))
        ds = ds.batch(batch_size)

        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


# Neural network

def create_classifier(
        learning_rate,
        hidden_units,
):
    feature_columns = [tf.feature_column.numeric_column("pixels", shape=784)]

    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        hidden_units=hidden_units,
        optimizer=optimizer,
        config=tf.estimator.RunConfig(keep_checkpoint_max=1),
        model_dir=FLAGS.work_dir
    )

    return classifier


def train_nn_classification_model(
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets
):
    periods = 1  # 10
    steps_per_period = steps / periods
    predict_training_input_fn = create_predict_input_fn(training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(training_examples, training_targets, batch_size)

    classifier = create_classifier(learning_rate, hidden_units)

    print("Training model...")
    print("LogLoss error (on validation data): ")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        classifier.train(input_fn=training_input_fn, steps=steps_per_period)

        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)

        print("  period %02d : %0.2f" % (period, validation_log_loss))

        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)

    print("Model training finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents')))

    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return classifier


# Export
def serving_input_receiver_fn():
    receiver_tensors = {
        # The size of input image is flexible.
        'pixels': tf.placeholder(tf.float32, [None, None, None, 1]),
    }
    # Convert give inputs to adjust to the model.
    features = {
        # Resize given images.
        'pixels': tf.image.resize_images(receiver_tensors['pixels'], [28, 28]),
    }
    # feature_spec = tf.feature_column.make_parse_example_spec(features)
    # return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors,
                                                    features=features)


    # serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[784], name='pixels')
    # receiver_tensors = {"pixels": serialized_tf_example}
    # feature_spec = tf.feature_column.make_parse_example_spec(
    #     [tf.feature_column.numeric_column("pixels", shape=784)])
    # features = tf.parse_example(serialized_tf_example, feature_spec)
    # return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def export(classifier):
    classifier.export_savedmodel(FLAGS.work_dir, serving_input_receiver_fn,
                                 strip_default_attrs=True)


def restore(learning_rate, hidden_units):
    classifier = create_classifier(learning_rate, hidden_units)
    return classifier


def main(_):
    # TODO: Executes train
    training_targets, training_examples = parse_labels_and_features(mnist_dataframe[:7500])
    validation_targets, validation_examples = parse_labels_and_features(mnist_dataframe[7500:10000])

    classifier = train_nn_classification_model(
        learning_rate=0.05,
        steps=1000,
        batch_size=30,
        hidden_units=[100, 100],
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

    print('Done training!')

    # TODO Restart from the previous state
    # classifier = restore(0.05, [100, 100])

    print('Start prediction test')
    predict_target, predict_example = parse_labels_and_features(mnist_dataframe[:1])
    predict_test_value_input_fn = create_predict_input_fn(
        predict_example, predict_target, 10)
    prediction = classifier.predict(input_fn=predict_test_value_input_fn, yield_single_examples=True)
    print("Predicted Value", [item['class_ids'][0] for item in prediction])
    first_image = np.array(predict_example, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

    # print('Start exporting')
    # TODO: Export model
    # export(classifier)
    # print('Done exporting!')


if __name__ == '__main__':
    tf.app.run()
