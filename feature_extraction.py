import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.utils import np_utils

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    num_classes = 43
    model = Sequential()
    model.add(Dense(num_classes, input_shape=X_train.shape[1:]))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.summary()
    

    # TODO: train your model here
    #one-hot-encode the labels
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_val   = np_utils.to_categorical(y_val,   num_classes)

    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                    batch_size=256, nb_epoch=50,
                    verbose=1, validation_data=(X_val, y_val), shuffle=True)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
