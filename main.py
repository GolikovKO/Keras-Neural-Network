from tensorflow import keras
from keras.utils import to_categorical
from keras.datasets import imdb, mnist, reuters, boston_housing
from keras import layers, models
import numpy as np

from functions import loses, relu_tanh, batch_1, neuron, lay_1, epochs_count, lay_2, batch_2, graph_1, graph_2


def first_exercise():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    network = keras.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    network.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=0)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print(test_acc)

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    print('___________________________')
    lay_1(1, x_train, y_train, x_test, y_test)
    lay_1(2, x_train, y_train, x_test, y_test)
    lay_1(3, x_train, y_train, x_test, y_test)
    print('___________________________')
    neuron(32, 32, x_train, y_train, x_test, y_test)
    neuron(32, 64, x_train, y_train, x_test, y_test)
    neuron(64, 64, x_train, y_train, x_test, y_test)
    neuron(64, 96, x_train, y_train, x_test, y_test)
    neuron(96, 128, x_train, y_train, x_test, y_test)
    print('___________________________')
    loses('binary_crossentropy', x_train, y_train, x_test, y_test)
    loses('mse', x_train, y_train, x_test, y_test)
    print('___________________________')
    relu_tanh('relu', x_train, y_train, x_test, y_test)
    relu_tanh('tanh', x_train, y_train, x_test, y_test)
    print('___________________________')
    batch_1(32, x_train, y_train, x_test, y_test)
    batch_1(64, x_train, y_train, x_test, y_test)
    batch_1(96, x_train, y_train, x_test, y_test)
    batch_1(128, x_train, y_train, x_test, y_test)
    batch_1(256, x_train, y_train, x_test, y_test)
    batch_1(512, x_train, y_train, x_test, y_test)
    print('////////////////////////////////////////////////////')


def second_exercise():
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]

    history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val),
                        verbose=0)

    graph_1(history)
    graph_2(history)

    print('___________________________')
    epochs_count(4, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    epochs_count(9, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    epochs_count(32, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    epochs_count(64, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    epochs_count(128, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    print('___________________________')
    lay_2(1, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    lay_2(2, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    lay_2(3, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    print('___________________________')
    batch_2(50, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    batch_2(100, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    batch_2(200, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    batch_2(350, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    batch_2(512, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    batch_2(600, partial_x_train, partial_y_train, x_test, one_hot_test_labels, x_val, y_val)
    print('////////////////////////////////////////////////////')


def third_exercise():
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std

    def build_model():
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 100
    all_scores = []

    for i in range(k):
        print('Processing fold #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)

        model = build_model()
        model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
        print(all_scores)
        print(np.mean(all_scores))


def main():
    # first_exercise()
    # second_exercise()
    third_exercise()


if __name__ == '__main__':
    main()
