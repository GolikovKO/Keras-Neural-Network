from tensorflow import keras
from keras.utils import to_categorical
from keras.datasets import imdb, mnist, reuters, boston_housing
from keras import layers, models
import numpy as np

from functions import loses, relu_tanh, batch_1, neuron, lay_1, epochs_count, lay_2, batch_2, graph_1, graph_2, epochs_3


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

    k = 4
    num_val_samples = len(train_data) // k

    print('___________________________')
    batch = 1
    layers_count = 2
    print('Количество эпох - 25')
    epochs_3(25, layers_count, batch, k, train_data, num_val_samples, train_targets)
    print('___________________________')
    print('Количество эпох - 50')
    epochs_3(50, layers_count, batch, k, train_data, num_val_samples, train_targets)
    print('___________________________')
    print('Количество эпох - 75')
    epochs_3(75, layers_count, batch, k, train_data, num_val_samples, train_targets)
    print('___________________________')
    print('Количество эпох - 100')
    epochs_3(100, layers_count, batch, k, train_data, num_val_samples, train_targets)
    print('___________________________')

    print('Количество слоёв - 1')
    epochs_3(100, 1, batch, k, train_data, num_val_samples, train_targets)
    print('Количество слоёв - 2')
    epochs_3(100, 2, batch, k, train_data, num_val_samples, train_targets)
    print('Количество слоёв - 3')
    epochs_3(100, 3, batch, k, train_data, num_val_samples, train_targets)
    print('___________________________')

    batch = 20
    print('Выборка - 20')
    epochs_3(100, 2, batch, k, train_data, num_val_samples, train_targets)
    batch = 50
    print('Выборка - 50')
    epochs_3(100, 2, batch, k, train_data, num_val_samples, train_targets)
    batch = 70
    print('Выборка - 70')
    epochs_3(100, 2, batch, k, train_data, num_val_samples, train_targets)
    print('___________________________')

    print('Функция активации relu - 1 слой')
    epochs_3(100, 1, batch, k, train_data, num_val_samples, train_targets)
    print('Функция активации relu - 2 слоя')
    epochs_3(100, 2, batch, k, train_data, num_val_samples, train_targets)
    print('Функция активации relu - 3 слоя')
    epochs_3(100, 3, batch, k, train_data, num_val_samples, train_targets)
    print('////////////////////////////////////////////////////')


def main():
    first_exercise()
    second_exercise()
    third_exercise()


if __name__ == '__main__':
    main()
